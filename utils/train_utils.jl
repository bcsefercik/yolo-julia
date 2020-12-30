import FileIO
import IterTools: ncycle

using Knet
import Knet: train!


function train!(model, train_data::Data, val_data::Data;
                period::Int=5, epoch::Int=10, lr=0.001,
                optimizer=adam, filename::String="", bestloss=nothing,
                results_filename::String=""
)

    if bestloss == nothing
        bestloss = model(val_data)
    end

    trn_loss = []
    trn_map = []
    val_loss = []
    val_map = []

    progress!(optimizer(model, ncycle(train_data, epoch), lr=lr), steps=period) do y
        push!(trn_loss, model(train_data))
        push!(val_loss, model(val_data))

        if val_loss[end] < bestloss
            bestloss = val_loss[end];

            if filename != ""
                save_model(model, filename)
            end
        end

        if results_filename != ""
            save_results(results_filename, trn_loss, trn_map, val_loss, val_map, bestloss)
        end

        return "tl: $(round(trn_loss[end], digits=2)), vl: $(round(val_loss[end], digits=2)), bl: $(round(bestloss, digits=2))"
    end

    return trn_loss, trn_map, val_loss, val_map, bestloss
end


function save_results(filename, results...)
    FileIO.save(
        filename,
        "trn_loss", results[1],
        "trn_map", results[2],
        "val_loss", results[3],
        "val_map", results[4],
        "bestloss", results[5]
    )
end

