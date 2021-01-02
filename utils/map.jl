using Knet
import Knet: Data

include("utils.jl")
include("constants.jl")
include("../models.jl")
include("../coco2014.jl")

# Adopted from: https://medium.com/analytics-vidhya/understanding-the-map-mean-average-precision-evaluation-metric-for-object-detection-432f5cca53b7
# Faster RCNN version

MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)

function compute_mAP(preds, labels, iou_thres=MINOVERLAP)
    NUM_CLASSES = length(CLASS_MAP)

    gt_counter_per_class = Dict{Integer, Integer}()

    gt_classified = Dict()
    pred_classified = Dict()

    for pred_ins in preds
        for pred in pred_ins
            if !haskey(pred_classified, Integer(pred[1]))
                pred_classified[Integer(pred[1])] = []
                gt_classified[Integer(pred[1])] = []
            end

            push!(
                pred_classified[Integer(pred[1])],
                pred
            )
        end
    end

    for label_ins in labels
        for label in label_ins
            if !haskey(gt_classified, Integer(label[1]))
                gt_classified[Integer(label[1])] = []
                pred_classified[Integer(label[1])] = []
            end

            push!(
                gt_classified[Integer(label[1])],
                label
            )
        end
    end

    sum_AP = 0.0
    cls_APs = Dict{Integer, Float32}()
    for (cls, _) in pred_classified
        cls_APs[cls], _, _ = compute_class_ap(
            pred_classified[cls],
            gt_classified[cls],
            iou_thres
        )

        sum_AP += cls_APs[cls]
    end

    mAP = sum_AP / length(cls_APs)

    return mAP, cls_APs
end


function compute_class_ap(predictions_data, labels, iou_thres=MINOVERLAP)
    nd = length(predictions_data)
    tp = zeros(nd)
    fp = zeros(nd)

    sort!(predictions_data, by=o->o[6], rev=true)
    labels = _get_x1y1x2y2.(labels)

    for (idx, pred) in enumerate(predictions_data)
        ovmax = -1
        gt_match = -1

        x1, y1, x2, y2 = pred[2], pred[3], pred[4], pred[5]

        for (idgt, obj) in enumerate(labels)
            xx1 = max.(x1, obj[2])
            yy1 = max.(y1, obj[3])
            xx2 = min.(x2, obj[4])
            yy2 = min.(y2, obj[5])

            iw = max.(0, xx2 - xx1 .+ 1)
            ih = max.(0, yy2 - yy1 .+ 1)



            if iw > 0 && ih > 0
                uni = (x2 - x1 + 1) * (y2 - y1 + 1) + (obj[4] - obj[2] + 1) * (obj[5] - obj[3] + 1) - iw * ih
                ov = (iw .* ih) ./ uni

                if ov > ovmax
                    ovmax = ov
                    gt_match = idgt
                end
            end
        end

        if ovmax >= iou_thres
            tp[idx] = 1
            deleteat!(labels, gt_match)
        else
            fp[idx] = 1
        end

    end

    csum = 0
    for (idx, val) in enumerate(fp)
        fp[idx] += csum
        csum += val
    end

    csum = 0
    for (idx, val) in enumerate(tp)
        tp[idx] += csum
        csum += val
    end

    rec = length(labels) > 0 ? tp ./ length(labels) : tp .* 0
    prec = tp ./ (fp .+ tp)

    ap, mrec, mprec = compute_voc_ap(rec, prec)

    return ap, mrec, mprec
end


function compute_voc_ap(rec, prec)

    if sum(prec) < 1e-5
        return 0, rec, prec
    end

    insert!(rec, 1, 0)
    append!(rec, 1)

    insert!(prec, 1, 0)
    append!(prec, 1)

    mpre = prec[:]
    mrec = rec[:]

    for i in length(mpre)-1:-1:1
       mpre[i] = max.(mpre[i], mpre[i+1])
    end

    i_list = Array{Integer}([])
    for i in 2:length(mrec)
        if mrec[i] != mrec[i-1]
            append!(i_list, i)
        end
    end

    ap = 0.0
    for i in i_list
        ap += ((mrec[i] - mrec[i-1]) * mpre[i])
    end

    return ap, mrec, mpre
end

function _get_x1y1x2y2(label, img_size=(416, 416))
    x1 = Integer(round((label[2] - label[4]/2) * img_size[2]))
    y1 = Integer(round((label[3] - label[5]/2) * img_size[1]))
    x2 = Integer(round((label[2] + label[4]/2) * img_size[2]))
    y2 = Integer(round((label[3] + label[5]/2) * img_size[1]))

    return [label[1], x1, y1, x2, y2]
end
