import Pkg

Pkg.activate("Project.toml")

import FileIO
import ProgressMeter
using ArgParse
using Knet

include("models.jl")
include("utils/train_utils.jl")
include("coco2014.jl")


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--model-out"
            help = "Model file path to save the trained model."
            arg_type = String
            required = true
        "--results"
            help = "Result file."
            arg_type = String
            default = "results.jld2"
        "--model-config"
            help = "Network config file."
            arg_type = String
            default = "cfg/yolov3.cfg"
        "--preload"
            help = "Pre-trained model file."
            arg_type = String
            default = nothing
        "--trndata"
            help = "COCO2014Data files for training."
            action = :append_arg
            nargs = '+'
            required = true
        "--valdata"
            help = "COCO2014Data files for validation."
            action = :append_arg
            nargs = '+'
            required = true
        "--epoch"
            help = "Number of epochs."
            arg_type = Int
            default = 100
        "--iepoch"
            help = "Number of instance epochs."
            arg_type = Int
            default = 2
        "--lr"
            help = "Learning rate"
            arg_type = Float64
            default = 0.001
        "--period"
            help = "Status printing period."
            arg_type = Int
            default = 10
        "--bs"
            help = "Batch size"
            arg_type = Int
            default = 8

    end

    return parse_args(s)
end


function main()
    args = parse_commandline()

    model = nothing
    if args["preload"] != nothing
        model = load_model(args["preload"]);
        @info "Preloaded an existing model."
    else
        model = Darknet(args["model-config"]; verbose=false);
        @info "Initialized a new model."
    end

    trndata_paths = args["trndata"][1]

    valdata = merge_data(args["valdata"][1])
    dval = minibatch(
        valdata.x,
        valdata.y,
        2;
        xsize = (416,416,3,:),
        xtype=Knet.atype()
    );
    @info "Loaded all validation data."

    instance_epoch = min(args["epoch"], args["iepoch"])
    outer_epoch = Int(ceil(args["epoch"]/instance_epoch))

    trn_loss, trn_map, val_loss, val_map = [], [], [], []
    best_val_loss = Float32(1e10)

    @info "Starting training."
    for ep in 1:outer_epoch
        for (ti, tpath) in enumerate(trndata_paths)
            trndata, dtrn = nothing, nothing

            trndata = load_data(tpath)
            dtrn = minibatch(
                trndata.x,
                trndata.y,
                args["bs"];
                xsize = (416,416,3,:),
                xtype=Knet.atype()
            );

            itl, itmap, ivl, ivmap, best_val_loss = train!(
                model, dtrn, dval;
                period=args["period"], epoch=instance_epoch, lr=args["lr"],
                optimizer=adam, filename=args["model-out"], bestloss=best_val_loss,
                results_filename="$(tpath)_$(args["results"])"
            )

            append!(trn_loss, itl)
            append!(trn_map, itmap)
            append!(val_loss, ivl)
            append!(val_map, ivmap)

            FileIO.save(
                args["results"],
                "trn_loss", trn_loss,
                "trn_map", trn_map,
                "val_loss", val_loss,
                "val_map", val_map
            )
        end
    end

end

main()
