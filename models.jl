using Statistics
import FileIO

using Knet: Data

include("nn.jl")
include("utils/parse_config.jl")
include("utils/utils.jl")
include("cfg/hyper_params.jl")

import .NN

function create_modules(module_defs, img_size; atype=Knet.atype())
    img_size = isa(img_size, Int) ? (img_size, img_size) : img_size

    filters = module_defs[1]["channels"]
    output_filters = [filters]

    module_list = []
    routes = []
    yolo_index = 0

    for (i, mdef) in enumerate(module_defs[2:end])
        modules = NN.Chain()

        if mdef["type"] == "convolutional"
            bn = mdef["batch_normalize"]
            filters = mdef["filters"]
            k = mdef["size"]
            stride = haskey(mdef, "stride") ? mdef["stride"] : (mdef["stride_y"], mdef["stride_x"])
            padding = haskey(mdef, "pad") ? floor(Integer, k/2) : 0

            push!(
                modules,
                NN.Conv2d(
                    k,
                    output_filters[end],
                    filters;
                    stride=stride,
                    bias=!bn,
                    padding=padding
                )
            )

            if bn
                push!(
                    modules,
                    NN.BatchNorm2d(filters; momentum=0.03, eps=1e-4)
                )
            else
                push!(routes, i)  # detection output (goes into yolo layer)
            end

            if mdef["activation"] == "leaky"
                push!(modules, NN.LeakyReLU(0.1))
            end

        elseif mdef["type"] == "upsample"
            modules = NN.Upsample2d(mdef["stride"])

        elseif mdef["type"] == "route"
            layers = mdef["layers"]
            filters = sum([output_filters[l > 1 ? l + 1 : length(output_filters) + l] for l in layers])
            append!(routes, [l < 1 ? i + l - 1 : l for l in layers])
            modules = NN.FeatureConcat(layers)

        elseif mdef["type"] == "shortcut"
            layers = mdef["from"]
            filters = output_filters[end]
            append!(routes, [l < 1 ? i + l - 1 : l for l in layers])

            modules = NN.WeightedFeatureFusion(layers)

        elseif mdef["type"] == "yolo"
            yolo_index += 1
            stride = [32, 16, 8]  # P5, P4, P3 strides
            layers = get(mdef, "from", [])
            anchors = mdef["anchors"][mdef["mask"], :]
            modules = NN.YOLOLayer(
                anchors,
                mdef["classes"],
                img_size,
                yolo_index,
                layers,
                stride[yolo_index]
            )

        else
            println("Warning: Unrecognized Layer Type: ", mdef["type"])
        end

        push!(module_list, modules)
        push!(output_filters, filters)
    end

    routes_bin = Array{Bool}([false for _ in 1:length(module_list)])
    routes_bin[routes] .= true

    return module_list, routes_bin
end


struct Darknet
    module_defs
    module_list
    routes
    yolo_layers
    verbose
    atype

    function Darknet(
        cfg;
        img_size=(416, 416),
        verbose=false,
        atype=Knet.atype()
    )
        module_defs = parse_model_cfg(cfg)
        module_list, routes = create_modules(module_defs, img_size; atype=atype)

        yolo_layers = [i for (i, m) in enumerate(module_list) if typeof(m) == NN.YOLOLayer]

        return new(
            module_defs,
            module_list,
            routes,
            yolo_layers,
            verbose,
            atype
        );
    end
end


function (c::Darknet)(x; training=true)
    img_size = size(x)[1:2]
    yolo_out, out = [], []

    if c.verbose; println("0 ", size(x)); end

    for (i, layer) in enumerate(c.module_list)
        layer_type = typeof(layer)

        if layer_type in [NN.FeatureConcat, NN.WeightedFeatureFusion]
            x = layer(x, out)

        elseif layer_type == NN.YOLOLayer
            push!(yolo_out, layer(x, out))
        else
            x = layer(x)
        end

        push!(out, c.routes[i] ? x : [])

        if c.verbose
            println(typeof(x))
            println("$i /$(length(c.module_list)) $layer_type ", size(x))
        end
    end

    if training
        return yolo_out
    else

        results = nothing

        for (i, out) in enumerate(yolo_out)
            layer_id = c.yolo_layers[i]

            no, ny, nx, na, bs = size(out)

            yv, xv = meshgrid(0:ny-1, 0:nx-1)
            grid = cat(yv, xv, dims=3)
            grid = permutedims(grid, (3, 1, 2))
            grid = reshape(grid, (2, ny, nx, 1, 1))
            grid = convert(c.atype, grid)

            io = deepcopy(out)
            io[1:2,:,:,:,:] = sigm.(io[1:2,:,:,:,:]) .+ grid
            io[3:4,:,:,:,:] = exp.(io[3:4,:,:,:,:]) .* c.module_list[layer_id].anchor_wh
            temp = deepcopy(io[1,:,:,:,:])
            io[1,:,:,:,:] = io[2,:,:,:,:]
            io[2,:,:,:,:] = temp
            io[1:4,:,:,:,:] = io[1:4,:,:,:,:] .* c.module_list[layer_id].stride
            io[5:end,:,:,:,:] = sigm.(io[5:end,:,:,:,:])

            r = reshape(io, (no, :, bs))

            if results == nothing
                results = r
            else
                results = cat(results, r, dims=2)
            end
        end

        return results

    end
end


function (model::Darknet)(x, y; training::Bool=true)
    # p = yolo_out, targets = y, pi = out
    yolo_out = model(x)  # p
    losses = convert(model.atype, zeros(3))
    lcls, lbox, lobj = losses[1], losses[2], losses[3]
    tcls, tbox, indices, anchors = build_targets(yolo_out, y, model)  # targets

    red = "mean"  # Loss reduction (sum or mean)

    nt = 0
    for (i, out) in enumerate(yolo_out)
        b, a, gj, gi = indices[i]

        # TODO: convert to knet array
        tobj = zeros(Integer, size(out)[2:end])  # target object

        nb = size(b)[1]
        if nb > 0
            nt += nb
            ps = out[:, gj[1], gi[1], a[1], b[1]]  # gj for y, gi for x

            for psi in 2:length(gj)
                ps = hcat(ps, out[:, gj[psi], gi[psi], a[psi], b[psi]])
            end

            pxy = sigm.(ps[1:2, :])
            pwh = clamp_.(exp.(ps[3:4, :]), 0, 1000) .* anchors[i]'
            pbox = cat(pxy, pwh; dims=1)  # vcat

            giou = bbox_giou(
                pbox,
                convert(model.atype, tbox[i]);
                x1y1x2y2=false
            )

            for ti in 1:length(gj)
                tobj[gj[ti], gi[ti], a[ti], b[ti]] = 1
            end

            lbox += (sum(1.0 .- giou) ./ nb)

            lcls += nll(ps[6:end, :], tcls[i]; average=red=="mean")

        end

        lobj += bce(out[5, :, :, :, :][:], tobj[:]; average=red=="mean")
    end

    lbox *= PARAM_GIOU
    lobj *= PARAM_OBJ
    lcls *= PARAM_CLS

    loss = lbox + lobj + lcls

    return loss

end

(c::Darknet)(d::Data) = mean(c(x,y) for (x,y) in d)  # Batch loss

function save_model(model::Darknet, filename::String)
    FileIO.save(filename, "darknet59", model)
    @info "Saved to: $filename."
end

function load_model(filename::String)
    model_d = FileIO.load(filename)
    @info "Loaded: $filename."
    return model_d["darknet59"]
end



# Hyperparameters
# hyp = {'giou': 3.54,  # giou loss gain
#        'cls': 37.4,  # cls loss gain
#        'cls_pw': 1.0,  # cls BCELoss positive_weight
#        'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
#        'obj_pw': 1.0,  # obj BCELoss positive_weight
#        'iou_t': 0.20,  # iou training threshold
#        'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
#        'lrf': 0.0005,  # final learning rate (with cos scheduler)
#        'momentum': 0.937,  # SGD momentum
#        'weight_decay': 0.0005,  # optimizer weight decay
#        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
#        'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
#        'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
#        'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
#        'degrees': 1.98 * 0,  # image rotation (+/- deg)
#        'translate': 0.05 * 0,  # image translation (+/- fraction)
#        'scale': 0.05 * 0,  # image scale (+/- gain)
#        'shear': 0.641 * 0}  # image shear (+/- deg)
