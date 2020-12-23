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


include("nn.jl")

import .NN

function train!(model, train_data::Data, test_data::Data;
                  period::Int=4, iters::Int=100, lr=0.15, optimizer=sgd!)  # or optimizer=adam!

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for i in 0:period:iters
        push!(train_loss, model(train_data))
        push!(test_loss, model(test_data))

        optimizer(model, take(cycle(train_data), period); lr=lr)


        println("Iter: ", i, " ", train_loss[end])
    end

    return 0:period:iters, train_loss, train_acc, test_loss, test_acc
end



include("utils/parse_config.jl")
include("models.jl")
Random.seed!(1)

darknet = Darknet("mnist.cfg"; verbose=false);


model = darknet

xtrn, ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10;
xtst, ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10;
xtrn = reshape(xtrn, (28, 28, 1, :))
xtrn = xtrn[:,:,:,1:6]
xtrn = unpool(xtrn; window=15)
xtrn = xtrn[3:418, 3:418, :, :]
xtst = reshape(xtst, (28, 28, 1, :))
xtst = xtst[:,:,:,1:4]
xtst = unpool(xtst; window=15)
xtst = xtst[3:418, 3:418, :, :]
# ytrn = rand(1:85, (1, size(yolo_re)[2:end]...))


dtrn = minibatch(xtrn, ytrn[1:6], 2; xsize = (416,416,1,:), xtype=Knet.atype(), shuffle=true);
dtst = minibatch(xtst, ytst[1:4], 2; xsize = (416,416,1,:), xtype=Knet.atype());


iters, trnloss, trnacc, tstloss, tstacc = train!(
    model, dtrn, dtst;
    period=1, iters=52, lr=1e-2, optimizer=sgd!);

# @time train!(
#     model, dtrn, dtst;
#     period=1, iters=10, lr=0.15, optimizer=sgd!);


