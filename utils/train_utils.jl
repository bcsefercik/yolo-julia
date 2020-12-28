import IterTools: ncycle

using Knet
import Knet: train!


function train!(model, train_data::Data, val_data::Data;
                period::Int=5, epoch::Int=10, lr=0.001,
                optimizer=adam, filename::String=""
)

    bestloss = model(val_data)

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

        return "tl: $(round(trn_loss[end], digits=2)), vl: $(round(val_loss[end], digits=2)), bl: $(round(bestloss, digits=2))"
    end

    return trn_loss, trn_map, val_loss, val_map
end
