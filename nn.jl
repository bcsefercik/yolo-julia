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




# Define chain of layers
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
struct Conv2d
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

#=function (c::FeatureConcat)(x, outputs)
    # TODO: implement this
    # return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]

end
=#
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


#=# TODO
class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.n = len(layers) + 1  # number of layers

    def forward(self, x, outputs):
        # Weights
        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            print("nx==na", nx == na)

            if nx != na:
                raise Exception

            # Adjust channels
            x = x + a

        return x=#


#=function (c::Upsample2d)(x)
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
end=#


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