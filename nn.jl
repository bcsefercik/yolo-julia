using CUDA
using Knet

function Knet.KnetArray(x::CuArray{T,N}) where {T,N}
    p = Base.bitcast(Knet.Cptr, pointer(x))
    k = Knet.KnetPtr(p, sizeof(x), Int(CUDA.device().handle), x)
    KnetArray{T,N}(k, size(x))
end

module NN

import Base: push!

using Knet
using Knet: conv4, pool, mat, sigm, KnetArray, nll, zeroone, progress, adam!, sgd!, param, param0, dropout, relu, minibatch, Data

using Statistics


struct Dense
    w; b; f; p;

    function Dense(inputsize::Int, outputsize::Int, f=lrelu;
            pdrop=0, atype=Knet.atype())

        return new(
            param(outputsize, inputsize; atype=atype),
            param0(outputsize; atype=atype),
            f,
            pdrop
        )
    end
end

(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)


struct Chain
    layers
    Chain(layers...) = new(convert(Array{Any}, [l for l in layers]))
end

(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)  # Forward pass

function push!(c::Chain, layer)
    return push!(c.layers, layer)
end


function (c::Chain)(x, y; accuracy::Bool=false)
    y_pred = c(x)

    if accuracy
        correct = 0.0

        for i=1:length(y)
            correct += y[i] == findmax(y_pred[:, i]; dims=1)[2][1] ? 1.0 : 0.0
        end

        return correct / length(y)
    else

        #=y_pred1 = reshape(y_pred, (1,10, :))
        y_pred1 = permutedims(y_pred1, (3,1,2))
        y_pred = reshape(y_pred, (10, :))=#
        return nll(y_pred, y)
    end
end

(c::Chain)(d::Data; accuracy::Bool=false) = mean(c(x,y; accuracy=accuracy) for (x,y) in d)  # Batch loss


# Leaky ReLU
struct LeakyReLU
    alpha
    function LeakyReLU(alpha=0.1, atype=Knet.atype())
        return new(convert(atype, [alpha]))
    end
end

function (c::LeakyReLU)(x)
    return max.(x, c.alpha .* x)
end

function leaky_relu(x, alpha=0.02)
    pos = relu(x)
    neg = -relu(-x)
    return pos + oftype(x, alpha) * neg
end


# 2D convolution
mutable struct Conv2d
    w; b; stride; padding;

    function Conv2d(
        kernel_size,
        in_channels::Int,
        out_channels::Int;
        stride=1,
        padding=0,
        bias=false,
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
mutable struct BatchNorm2d
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
    return unpool(x; window=c.scale_factor)
end


struct FeatureConcat
    layers; multiple

    function FeatureConcat(
        layers
    )
        return new(
            layers,
            length(layers) > 1
        )
    end
end

function (c::FeatureConcat)(x, outputs)
    layers = [l < 1 ? l + length(outputs) : l for l in c.layers]

    return c.multiple ? cat(outputs[layers]..., dims=3) : outputs[layers[1]]
end


struct WeightedFeatureFusion
    layers; n

    function WeightedFeatureFusion(
        layers
    )
        return new(
            layers,
            length(layers) + 1
        )
    end
end

function (c::WeightedFeatureFusion)(x, outputs)
    # Next version: implement weighted version
    layers = [l < 1 ? l + length(outputs) : l for l in c.layers]

    for l in layers
        x = x .+ outputs[l]
    end

    return x
end


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
    atype

    function YOLOLayer(
        anchors,
        nc,
        img_size,
        yolo_index,
        layers,
        stride;
        atype=Knet.atype()
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
            convert(atype, reshape(anchor_vec', (2, 1, 1, na, 1))),
            atype
        )
    end
end

function (c::YOLOLayer)(p, out)
    ny, nx, _, bs = size(p)

    r = reshape(p, (ny, nx, c.no, c.na, bs))
    r = permutedims(r, (3, 1, 2, 4, 5))

    return r
end


end  ## NN module end
