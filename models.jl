include("nn.jl")

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
            filters = sum([output_filters[l > 0 ? l + 2 : length(output_filters) + l + 1] for l in layers])
            append!(routes, [l < 0 ? i + l : l + 1 for l in layers])

            # TODO: impolement this:
            modules = NN.FeatureConcat(layers)

        elseif mdef["type"] == "shortcut"
            layers = mdef["from"]
            filters = output_filters[end]
            append!(routes, [l < 0 ? i + l : l + 1 for l in layers])

            # TODO: implement this:
            modules = NN.WeightedFeatureFusion(layers)

        elseif mdef["type"] == "yolo"
            # TODO: implement:
            continue
        else
            println("Warning: Unrecognized Layer Type: ", mdef["type"])
        end

        push!(module_list, modules)
        push!(output_filters, filters)
    end

    return module_list, routes
end