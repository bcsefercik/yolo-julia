using Knet

include("../hyper_params.jl")

function build_targets(p, targets, model)
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
    gain = ones(1, 6)

    for (i, j) in enumerate(model.yolo_layers)
        # i: yolo_layer intra index (1, 2, 3)
        # j: yolo_layer darknet module index
        anchors = model.module_list[j].anchor_vec
        gain[1, 3:end] .= size(p[i])[[3, 2, 3, 2]]  # xyxy gain
        na = size(anchors)[1]
        at = repeat(reshape(1:na, (na, 1)), 1, nt)

        a, t, offsets = [], targets_reshaped .* gain, 0.0

        if nt > 0
            j = wh_iou(anchors, t[:, 5:6]) .> PARAM_IOU_T
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


function bbox_giou(box1, box2; x1y1x2y2=false)
    box2 = convert(Knet.atype(), transpose(box2))

    b1_x1 = box1[1, :] .- box1[3, :] ./ 2
    b1_x2 = box1[1, :] .+ box1[3, :] ./ 2
    b1_y1 = box1[2, :] .- box1[4, :] ./ 2
    b1_y2 = box1[2, :] .+ box1[4, :] ./ 2

    b2_x1 = box2[1, :] .- box2[3, :] ./ 2
    b2_x2 = box2[1, :] .+ box2[3, :] ./ 2
    b2_y1 = box2[2, :] .- box2[4, :] ./ 2
    b2_y2 = box2[2, :] .+ box2[4, :] ./ 2

    inter = clamp.(min.(b1_x2, b2_x2) - max.(b1_x1, b2_x1), 0, 1000000) .*
            clamp.(min.(b1_y2, b2_y2) - max.(b1_y1, b2_y1), 0, 1000000)

    w1, h1 = box1[3, :], box1[4, :]
    w2, h2 = box2[3, :], box2[4, :]

    uni = (w1 .* h1 .+ 1e-16) + w2 .* h2 .- inter

    iou = inter ./ uni

    # GIoU
    # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
    cw = max.(b1_x2, b2_x2) .- min.(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = max.(b1_y2, b2_y2) .- min.(b1_y1, b2_y1)  # convex height

    c_area = cw .* ch .+ 1e-16  # convex area

    return iou .- ((c_area .- uni) ./ c_area)
end
