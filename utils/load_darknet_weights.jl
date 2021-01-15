function load_darknet_weights(model, weights, cutoff=-1)
    f = open(weights)

    version = Array{Int32}(UndefInitializer(), 3);
    read!(f, version);

    seen = Array{Int64}(UndefInitializer(), 1);
    read!(f, seen);

    for (i, mdef) in enumerate(model.module_defs[2:end+cutoff])
        if mdef["type"] == "convolutional"
            conv = model.module_list[i].layers[1];

            if mdef["batch_normalize"]
                bn = model.module_list[i].layers[2];

                nb = length(_bnbias(bn.params));
                moment_size = _wsize(conv.w);

                bn_bias = Array{Float32}(UndefInitializer(), nb);
                bn_weight = Array{Float32}(UndefInitializer(), nb);
                bn_running_mean = Array{Float32}(UndefInitializer(), nb);
                bn_running_var = Array{Float32}(UndefInitializer(), nb);

                read!(f, bn_bias);
                read!(f, bn_weight);
                read!(f, bn_running_mean);
                read!(f, bn_running_var);

                bn.params[div(length(bn.params), 2)+1:end] = convert(typeof(bn.params), bn_bias);
                bn.params[1:div(length(bn.params), 2)] = convert(typeof(bn.params), bn_weight);
                bn.moments.mean = convert(Knet.atype(), reshape(bn_running_mean, moment_size));
                bn.moments.var = convert(Knet.atype(), reshape(bn_running_var, moment_size));
            else
                nb = length(conv.b)

                conv_b = Array{Float32}(UndefInitializer(), nb);
                read!(f, conv_b);

                conv_b = reshape(conv_b, size(conv.b));

                conv.b = Param(convert(Knet.atype(), conv_b));
            end

            nw = length(conv.w)

            conv_w = Array{Float32}(UndefInitializer(), nw);
            read!(f, conv_w);

            conv_w = reshape(conv_w, size(conv.w));
            conv_w = permutedims(conv_w, (2, 1, 3, 4));
            conv_w = flipkernel(conv_w);

            conv.w = Param(convert(Knet.atype(), conv_w));
        end
    end

    close(f)
end


_wsize(y) = ((1 for _=1:ndims(y)-2)..., size(y)[end], 1)
_bnscale(param) = param[1:div(length(param), 2)]
_bnbias(param) = param[div(length(param), 2)+1:end]
flipkernel(x) = x[end:-1:1, end:-1:1, :, :]
