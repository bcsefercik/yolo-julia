using Knet


function is_valid(l)
    if length(strip(l)) < 1; return false; end

    if l[1] == '#'; return false; end

    return true
end


function get_number(s)
    result = tryparse(Int, s)

    if result == nothing; result = tryparse(Float32, s); end

    return result
end


function parse_model_cfg(path::String="yolov3.cfg"; atype=Knet.atype())
    lines = [strip(l) for l in eachline(path) if is_valid(l)]

    mdefs = []

    for line in lines
        if line[1] == '['
            push!(mdefs, Dict())

            mdefs[end]["type"] = strip(line[2:end-1])

            if mdefs[end]["type"] == "convolutional"
                mdefs[end]["batch_normalize"] = false
            end
        else
            key, val = split(line, "=")
            key, val = strip(key), strip(val)
            if key == "anchors"
                anchors =  [parse(Float32, x) for x in split(val, ",")]
                mdefs[end][key] = convert(atype, transpose(reshape(anchors, 2, :)))

            elseif (key in ("from", "layers", "mask")) || (key == "size" && occursin(",", val))
                mdefs[end][key] = [parse(Int, x) + 1 for x in split(val, ",")]

            elseif key == "batch_normalize"
                mdefs[end][key] = get_number(val) == 1

            else
                mdefs[end][key] = get_number(val)
                if mdefs[end][key] == nothing; mdefs[end][key] = val; end
            end
        end
    end

    # Field support check
    supported_fields = Set([
        "type", "batch_normalize", "filters", "size", "stride", "pad", "activation", "layers", "groups",
        "from", "mask", "anchors", "classes", "num", "jitter", "ignore_thresh", "truth_thresh", "random",
        "stride_x", "stride_y", "weights_type", "weights_normalization", "scale_x_y", "beta_nms", "nms_kind",
        "iou_loss", "iou_normalizer", "cls_normalizer", "iou_thresh", "probability"
    ])

    current_fields = Set([k for x in mdefs[2:end] for k in keys(x)])

    @assert current_fields <= supported_fields "Unsupported cfg field!"

    return mdefs
end