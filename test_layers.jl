import Pkg

Pkg.activate("Project.toml")

using CUDA
using Statistics
using Random
using Test
import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail
import .Iterators: cycle, Cycle, take
using Plots; default(fmt=:png,ls=:auto)

import Knet
using Knet: deconv4, conv4, unpool, pool, mat, sigm, KnetArray, nll, zeroone, progress, adam!, sgd!, param, param0, dropout, relu, minibatch, Data
import Knet: train!

using MLDatasets: MNIST

function Knet.KnetArray(x::CuArray{T,N}) where {T,N}
    p = Base.bitcast(Knet.Cptr, pointer(x))
    k = Knet.KnetPtr(p, sizeof(x), Int(CUDA.device().handle), x)
    KnetArray{T,N}(k, size(x))
end


include("utils/parse_config.jl")
include("models.jl")

# darknet = Darknet("mnist.cfg"; verbose=true);
# include("nn.jl")

import .NN

function train!(model::NN.Chain, train_data::Data, test_data::Data;
                  period::Int=4, iters::Int=100, lr=0.15, optimizer=sgd!)  # or optimizer=adam!

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for i in 0:period:iters
        push!(train_loss, model(train_data))
        push!(test_loss, model(test_data))

        push!(train_acc, model(train_data; accuracy=true))
        push!(test_acc, model(test_data; accuracy=true))

        optimizer(model, take(cycle(train_data), period); lr=lr)

	if i%10 == 0
	   println(i, " ", train_loss[end], " ", train_acc[end], " ", test_loss[end], " ", test_acc[end])
	end

            println("Iter: ", i)
    end

    return 0:period:iters, train_loss, train_acc, test_loss, test_acc
end

Random.seed!(1)

model = NN.Chain(
    NN.Conv2d(1, 1, 5),
    NN.LeakyReLU(),
    NN.Upsample2d(2),
    NN.Dense(2888000,10,identity),
)


# model = NN.Chain(
#     darknet,
#     NN.Dense(587520,10,identity)
# )

xtrn, ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10;
xtst, ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10;
xtrn = reshape(xtrn, (28, 28, 1, :))
xtrn = xtrn[:,:,:,1:20]
xtrn = unpool(xtrn; stride=14)
xtst = reshape(xtst, (28, 28, 1, :))
xtst = xtst[:,:,:,1:8]
xtst = unpool(xtst; stride=14)
dtrn = minibatch(xtrn, ytrn[1:20], 2; xsize = (380,380,1,:), xtype=Knet.atype(), shuffle=true);
dtst = minibatch(xtst, ytst[1:8], 2; xsize = (380,380,1,:), xtype=Knet.atype());


darknet = Darknet("mnist.cfg"; verbose=true);

model = NN.Chain(
    darknet,
    NN.Dense(587520,10,identity)
)


iters, trnloss, trnacc, tstloss, tstacc = train!(
    model, dtrn, dtst;
    period=1, iters=1004, lr=1e-4, optimizer=sgd!);