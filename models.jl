include("nn.jl")
include("utils/parse_config.jl")

import .NN

function create_modules(module_defs, img_size)
    img_size = isa(img_size, Int) ? (img_size, img_size) : img_size

    filters = module_defs[1]["channels"]
    output_filters = [filters]

    module_list = []
    routes = []
    yolo_index = -1

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
            # TODO: implement:
            println("yolo")
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

    function Darknet(
        cfg;
        img_size=(416, 416),
        verbose=false
    )
        module_defs = parse_model_cfg(cfg)
        module_list, routes = create_modules(module_defs, img_size)
        yolo_layers = nothing

        return new(
            module_defs,
            module_list,
            routes,
            yolo_layers
        );
    end
end

function (c::Darknet)(x; verbose=false)
    img_size = size(x)[1:2]
    yolo_out, out = [], []

    if verbose; println("0 ", size(x)); end

    for (i, layer) in enumerate(c.module_list)
        layer_type = typeof(layer)

        if layer_type in [NN.FeatureConcat, NN.WeightedFeatureFusion]
            x = layer(x, out)
        elseif layer_type == YOLOLayer
            continue
        else
            x = layer(x)
        end

        push!(out, c.routes[i] ? x : [])

        if verbose
            println(x[:,:,:,1])
            println("$i /$(length(c.module_list)) $layer_type ", size(x))
        end
    end

    return x
end
