using Knet

include("../cfg/hyper_params.jl")


function clamp_(x, lo, hi)
    return min.(max.(x, lo), hi)
end


function build_targets(p, targets, model)
    # p: yolo_outs

    tcls, tbox, indices, anch = [], [], [], []

    if length(targets) == 0
        return tcls, tbox, indices, anch
    end

    targets = [target for target in targets if length(target) != 0]

    if length(targets) == 0
        return tcls, tbox, indices, anch
    end

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
        # t = convert(model.atype, t)
        if nt > 0
            j = wh_iou(anchors, t[:, 5:6]) .> PARAM_IOU_T
            tt = repeat(t, na)
            j = convert(Array{Bool}, j)
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
                Integer.(clamp_.(gj, 1, gain[4])),
                Integer.(clamp_.(gi, 1, gain[3]))
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
    wh2r = convert(Knet.atype(), reshape(wh2, (1, :, 2)))  # [1,M,2]

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

    inter = clamp_.(min.(b1_x2, b2_x2) - max.(b1_x1, b2_x1), 0, 1000000) .*
            clamp_.(min.(b1_y2, b2_y2) - max.(b1_y1, b2_y1), 0, 1000000)

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


function xywh2xyxy_p(x)
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = convert(typeof(x), zeros(size(x)))

    y[1, :] = x[1, :] - (x[3, :] ./ 2)  # top left x
    y[2, :] = x[2, :] - (x[4, :] ./ 2)  # top left y
    y[3, :] = x[1, :] + (x[3, :] ./ 2)  # bottom right x
    y[4, :] = x[2, :] + (x[4, :] ./ 2)  # bottom right y

    return y
end


function meshgrid(x, y)
   X = [i for i in x, j in 1:length(y)]
   Y = [j for i in 1:length(x), j in y]

   return X, Y
end

function nms(prediction; conf_thres=0.05, iou_thres=0.6)
    min_wh, max_wh = 2, 4096
    bs = size(prediction)[3]
    n = size(prediction)[1]
    nc = n - 5

    min_wh, max_wh = 2, 4096

    results = []

    for xi in 1:bs
        x = prediction[:, :, xi]

        x = x[:, x[5, :] .> conf_thres]
        x = x[:, x[3, :] .> min_wh]
        x = x[:, x[4, :] .> min_wh]
        x = x[:, x[3, :] .< max_wh]
        x = x[:, x[4, :] .< max_wh]

        result = []

        if length(x) == 0
            push!(results, result)
            continue
        end

        x[6:end, :] = x[6:end, :] .* x[5:5, :]  # conf = obj_conf * cls_conf


        x[1:4, :] = xywh2xyxy_p(x[1:4, :])


        dets = Dict{Integer, Any}()

        class_candidates = findall(
            o->o==1,
            x[6:end, :] .> conf_thres
        )


        # class_candidates = argmax(x[6:end, :], dims=1)
        # println(class_candidates)
        # return class_candidates

        # distribute detections to classes
        for cc in class_candidates
            if !haskey(dets, cc[1])
                dets[cc[1]] = x[1:5, cc[2]]
            else
                dets[cc[1]] = cat(
                    dets[cc[1]],
                    x[1:5, cc[2]],
                    dims=2
                )
            end
        end

        for (k, v) in dets
            nmsed = nms_class(v, iou_thres=iou_thres)

            nmsed = cat(reshape(repeat([k], size(nmsed)[2]), (1, :)), nmsed,  dims=1)

            for jj in 1:size(nmsed)[2]
                push!(result, nmsed[:, jj])
            end
        end

        push!(results, result)
    end

    return results
end

function nms_class(p; iou_thres=0.6)
    # xyxyc

    x1 = p[1, :]
    y1 = p[2, :]
    x2 = p[3, :]
    y2 = p[4, :]

    area = (x2 - x1 .+ 1) .* (y2 - y1 .+ 1)

    idxs = sortperm(p[5, :])
    pick = Array{Integer}([])
    bboxes = nothing

    while length(idxs) > 0
        lst = length(idxs)
        i = idxs[lst]
        push!(pick, i)

        xx1 = max.(x1[i], x1[idxs[1:lst-1]])
        yy1 = max.(y1[i], y1[idxs[1:lst-1]])
        xx2 = min.(x2[i], x2[idxs[1:lst-1]])
        yy2 = min.(y2[i], y2[idxs[1:lst-1]])

        w = max.(0, xx2 - xx1 .+ 1)
        h = max.(0, yy2 - yy1 .+ 1)

        overlap = (w .* h) ./ area[idxs[1:lst-1]]

        # to_be_merged = cat(
        #                     [i for (i, v) in enumerate(overlap) if v > iou_thres],
        #                     lst,
        #                     dims=1
        #                 )
        # merged = merge_bboxes(p[:, to_be_merged])
        # if bboxes == nothing
        #     bboxes = merged
        # else
        #     bboxes = cat(bboxes, merged, dims=2)
        # end

        deleteat!(
            idxs,
            cat(
                [i for (i, v) in enumerate(overlap) if v > iou_thres],
                lst,
                dims=1
            )
        )

    end

    return p[1:end, pick]
    # return bboxes
end

function merge_bboxes(bboxes)
    result = convert(Array{Float32}, zeros(size(bboxes)[1], 1))
    result[1, 1] = minimum(bboxes[1, :])
    result[2, 1] = minimum(bboxes[2, :])
    result[3, 1] = maximum(bboxes[3, :])
    result[4, 1] = maximum(bboxes[4, :])
    result[5, 1] = sum(bboxes[5, :]) ./ size(bboxes)[2]
    return result
end
