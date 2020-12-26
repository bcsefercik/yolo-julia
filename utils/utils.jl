include("../hyper_params.jl")

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

    targets_reshaped = transpose(targets_reshaped)

    nt = size(targets_reshaped)[1]
    tcls, tbox, indices, anch = [], [], [], []
    gain = convert(atype, ones(1, 6))

    for (i, j) in enumerate(model.yolo_layers)
        # i: yolo_layer intra index (1, 2, 3)
        # j: yolo_layer darknet module index
        anchors = model.module_list[j].anchor_vec
        gain[1, 3:end] .= size(p[i])[[3, 2, 3, 2]]  # xyxy gain
        na = size(anchors)[1]
        at = repeat(reshape(1:na, (na, 1)), 1, nt)

        a, t, offsets = [], targets_reshaped .* gain, 0.0

        if nt > 0
            j = wh_iou(anchors, t[:, 5:6]) .> IOU_T
            tt = repeat(t, na)
            jj = vcat(j'...)
            a, t = at'[j'], tt[jj, :]

        end

        b = Integer.(t[:, 1])
        c = Integer.(t[:, 2])
        gxy = t[:, 3:4]  # xy
        gwh = t[:, 5:6]  # wh
        gij = ceil.(Integer, gxy .- offsets)  # xy indices starts from 1 in julia
        gi = gij[:, 1]  # x
        gj = gij[:, 2]  # y

        push!(
            indices,
            (
                b,
                a,
                Integer.(clamp.(gj, 1, gain[4])),
                Integer.(clamp.(gi, 1, gain[3]))
            )
        )

        push!(
            tbox,
            cat(gxy - gij .+ 1.0, gwh; dims=2)  # x (in cell), y (in cell), w, h
        )

        push!(anch, anchors[a, :])
        push!(tcls, c)
    end

    return tcls, tbox, indices, anch
end

function wh_iou(wh1, wh2)
     # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    N, _ = size(wh1)
    M, _ = size(wh2)
    wh1r = reshape(wh1, (:, 1, 2))  # [N,1,2]
    wh2r = reshape(wh2, (1, :, 2))  # [1,M,2]

    inter = reshape(
        prod(min.(wh1r, wh2r), dims=3),
        (N, M)
    )  # [N,M]

    uni = reshape(prod(wh1r, dims=3), (:, 1)) .+ reshape(prod(wh2r, dims=3), (1, :))
    uni = uni .- inter  # [N,M]

    return inter ./ uni


end