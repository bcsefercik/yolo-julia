function build_targets(p, targets, model; atype=Knet.atype())
    # p: yolo_outs

    targets_reshaped = hcat(targets[1]...)
    targets_reshaped = vcat(ones(1, length(targets[1])), targets_reshaped)

    for (i, target) in enumerate(targets[2:end])
        targets_reshaped = hcat(
            targets_reshaped,
            vcat(
                ones(1, length(target)) * (i+1),
                hcat(target...)
            )
        )
    end

    nt = size(targets_reshaped)[2]
    tcls, tbox, indices, anch = [], [], [], []
    gain = convert(atype, ones(6))

    for (i, j) in enumerate(model.yolo_layers)
        # i: yolo_layer intra index (1, 2, 3)
        # j: yolo_layer darknet module index
        anchors = model.module_list[j].anchor_vec
        gain[3:end] .= size(p[i])[[3, 2, 3, 2]]
        na = size(anchors)[1]
        at = repeat(reshape(1:na, (na, 1)), 1, nt)

        a, t, offsets = [], targets_reshaped .* gain, 0

        if nt > 0

        end

        return targets_reshaped, gain
    end

    return tcls, tbox, indices, anch
end
