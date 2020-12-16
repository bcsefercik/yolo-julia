module NN

using Knet


# Leaky ReLU
function lrelu(x, alpha=0.1)
    pos = relu(x)
    neg = relu(-x)
    return pos + alpha * neg
end


# 2D convolution
struct Conv2d
    w; b; stride; padding;

    function Conv2d(
        in_channels::Int,
        out_channels::Int,
        kernel_size;
        stride=1,
        padding=1,
        bias=true,
        atype=Knet.atype()
    )

        kernel_size = kernel_size isa Tuple ? kernel_size : (kernel_size, kernel_size)

        return new(
            param(
                kernel_size[1],
                kernel_size[2],
                in_channels,
                out_channels;
                atype=atype
            ),
            bias ? param0(1, 1, out_channels, 1; atype=atype) : nothing,
            stride,
            padding
        )
    end
end

function (c::Conv2d)(x)
    result = conv4(c.w, x; stride=c.stride, padding=c.padding)

    if c.b != nothing
        result = result .+ c.b
    end

    return result
end


# 2D Batch Normalization
struct BatchNorm2d
    moments; params; eps

    function BatchNorm2d(
        num_features;
        eps=1e-4,
        momentum=0.03,
        atype=Knet.atype()
    )

        return new(
            bnmoments(momentum=momentum),
            convert(Knet.atype(), bnparams(num_features)),
            eps
        )
    end
end

function (c::BatchNorm2d)(x)
    return batchnorm(x, c.moments, c.params; eps=c.eps)
end


# 2D Upsample
struct Upsample2d
    scale_factor; mode

    function Upsample2d(
        scale_factor;
        mode="nearest"
    )
        return new(
            scale_factor,
            mode
        )
    end
end

function (c::Upsample2d)(x)
    xs = size(x)
    sf = c.scale_factor

    r = convert(
        typeof(x),
        zeros(
            xs[1] * sf,
            xs[2] * sf,
            xs[3],
            xs[4]
        )
    )
    rs = size(r)

    if c.mode == "nearest"
        for j in 1:sf
            for i in 1:sf
                r[i:sf:rs[1], j:sf:rs[2], :, :] = x
            end
        end
    end

    return r
end


#=struct Dense
    w; b; f; p;

    function Dense(inputsize::Int, outputsize::Int, f=relu;
            pdrop=0, atype=dtype())

        return new(
            param(outputsize, inputsize; atype=atype),
            param0(outputsize; atype=atype),
            f,
            pdrop
        )
    end
end

(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b)
=#
end  ## NN module end