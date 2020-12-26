using Statistics

using Knet: Data, sigm
import Knet: sigm

include("nn.jl")
include("utils/parse_config.jl")
include("utils/utils.jl")

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

            # println(i, layers, filters, [l < 1 ? i + l - 1 : l for l in layers])
            # println(length(output_filters), [l > 1 ? l + 1 : length(output_filters) + l for l in layers])

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
            modules = YOLOLayer(
                anchors,
                mdef["classes"],
                img_size,
                yolo_index,
                layers,
                stride[yolo_index]
            )



            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            #=try
                j = get(mdef, "from", 0)
                j = j < 1 ? j + length(module_list) : j

                bias_ = value(module_list[j].layers[1].b)
                bias = bias_[:, :, 1:modules.no * modules.na, :]
                bias = reshape(bias, (modules.na, :))
                bias[:, 5] = bias[:, 5] .- 4.5
                bias[:, 6:end] = bias[:, 6:end] .+ log(0.6 / (modules.nc - 0.99))
                bias = reshape(bias, (1, 1, :, 1))
                module_list[j].layers[1].b = Param(convert(atype, bias))

            catch y
                println("WARNING: smart bias initialization failure. ", y)
            end=#

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

        yolo_layers = [i for (i, m) in enumerate(module_list) if typeof(m) == YOLOLayer]

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

        elseif layer_type == YOLOLayer
            push!(yolo_out, layer(x, out; training=training))
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
        return x
    end
end

function (model::Darknet)(x, y; training::Bool=true)
    # TODO: Implement Loss
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
        tobj = zeros(size(out)[2:end])  # target object

        nb = size(b)[1]
        if nb > 0
            nt += nb
            ps = out[:, gj[1], gi[1], a[1], b[1]]  # gj for y, gi for x

            # ps = Param(convert(Knet.atype(), ps))

            for psi in 2:length(gj)
                ps = hcat(ps, out[:, gj[psi], gi[psi], a[psi], b[psi]])
            end
            # pxy = tanh.(ps[1:2, :])
            pxy = sigm.(ps[1:2, :])

            # pxy = ps[1:2, :]
            pwh = clamp.(exp.(ps[3:4, :]), 0, 1000) .* anchors[i]'
            pbox = cat(pxy, pwh; dims=1)  # vcat

            giou = bbox_giou(
                pbox,
                convert(model.atype, tbox[i]);
                x1y1x2y2=false
            )

            lbox = lbox + (sum(1.0 .- giou) ./ nb)
        end
    end

    loss = lbox + lobj + lcls

    return loss

end

# Batch loss
# function (model::Darknet)(d::Data)
#     loss = 0.0

#     for (x, y) in d
#         l = model(x, y)

#         loss += l
#     end
# end
(c::Darknet)(d::Data) = mean(c(x,y) for (x,y) in d)  # Batch loss



mutable struct YOLOLayer
    anchors
    index
    layers
    stride
    nl
    na
    nc
    no
    nx; ny; ng
    anchor_vec
    anchor_wh

    function YOLOLayer(
        anchors,
        nc,
        img_size,
        yolo_index,
        layers,
        stride
    )
        na = size(anchors)[1]
        ns = yolo_index * 13
        anchor_vec = anchors ./ stride
        return new(
            anchors,
            yolo_index,
            layers,
            stride,
            length(layers),
            na,
            nc,
            nc + 5,
            ns, ns, (ns, ns),
            anchor_vec,
            reshape(anchor_vec, (1, na, 1, 1, 2))
        )
    end
end

function (c::YOLOLayer)(p, out; training=true)
    ny, nx, _, bs = size(p)

    r = reshape(p, (ny, nx, c.no, c.na, bs))
    r = permutedims(r, (3, 1, 2, 4, 5))

    if training
        return r
    end
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
