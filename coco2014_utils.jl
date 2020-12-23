import JSON
import Images

function coco_to_yolo_bbox(coco_bbox, width, height)
    bbox::Array{Float64} = [0, 0, 0, 0]

    bbox[1] = coco_bbox[1]/width
    bbox[2] = coco_bbox[2]/height
    bbox[3] = coco_bbox[3]/width
    bbox[4] = coco_bbox[4]/height
    bbox[1] += bbox[3] / 2
    bbox[2] += bbox[4] / 2

    return bbox
end


function onehot(l, index)
    v = zeros(l)
    v[index] = 1
    return v
end


function read_image(image_path::String, img_size=(416, 416); dtype=Array{Float32})

    img = Images.load(image_path)
    img = Images.imresize(img, img_size)
    cv = Images.channelview(img)

    # Grayscale handling
    # cvp = ifelse(ndims(cv) == 3, permutedims(cv, (2,3,1)), repeat(cv, 1,1, 3)) # This is not lazy
    cvp = ndims(cv) == 3 ? permutedims(cv, (2,3,1)) : repeat(cv, 1,1, 3)
    cv2arr = convert(dtype, cvp)
    # cv2arr = cv2arr[1:416, 1:416, :]  # For baseline purposes, to be removed

    return cv2arr
end


function load_data(
    images::String, label_file::String;
    raw_labels::Bool=false, class_file::String="", dtype=Array{Float32}
    )

    x = nothing
    y = []

    # Read labels
    if raw_labels
        labels, class_ids = convert_annotations_to_labels(label_file)
    else
        labels = JSON.parsefile(
            label_file;
            dicttype=Dict,
            inttype=Integer,
            use_mmap=true
        )
        class_ids = JSON.parsefile(
            class_file;
            dicttype=Dict{String, Integer},
            inttype=Integer,
            use_mmap=true
        )
    end

    class_count = length(class_ids)

    # Read images
    for (root, _, files) in walkdir(images)
        for img_path in files

            # println("Loading: ", img_path)

            x_curr = read_image(joinpath(root, img_path), dtype=dtype)
            file_name = split(img_path, ".")[1]
            # y_curr_list = map(
            #     y -> convert(
            #         dtype,
            #         vcat(onehot(class_count, Integer(y[1])), y[1:end])
            #     ),
            #     get(labels, file_name, [])
            # )

            y_curr = get(labels, file_name, [])

            # cat(x, x_curr; dims=)
            # push!(y, y_curr)
            if x == nothing
                x = x_curr
            else
                x = cat(x, x_curr; dims=4)
            end

            push!(y, y_curr)
        end
    end

    return x, y
end


function convert_annotations_to_labels(annotation_file::String; kwargs...)
    annotations = JSON.parsefile(
        annotation_file;
        dicttype=Dict,
        inttype=Int32,
        use_mmap=true
    )

    classes = Dict{String, Integer}()  # default_category_id{String} => category_index{Int}
    images = Dict()  # image_id => (file_name, width, height)
    labels = Dict()  # file_name => [class, bbox...]

    if haskey(kwargs, :class_mappings)
        classes = JSON.parsefile(
            kwargs[:class_mappings];
            dicttype=Dict{String, Integer},
            inttype=Int32,
            use_mmap=true
        )
    else
        for c in annotations["categories"]
            if !haskey(classes, string(c["id"]))
                classes[string(c["id"])] = length(classes) + 1
            end
        end
    end

    if haskey(kwargs, :class_mappings_out)
        open(kwargs[:class_mappings_out], "w") do io
            JSON.print(io, labels)
        end
    end

    for img in annotations["images"]
        if !haskey(images, img["id"])
            images[img["id"]] = (
                split(img["file_name"], ".")[1],
                img["width"],
                img["height"]
            )
        end
    end

    for ann in annotations["annotations"]
        file_name = images[ann["image_id"]][1]
        if !haskey(labels, file_name)
            labels[file_name] = []
        end

        push!(
            labels[file_name],
            [
                classes[string(ann["category_id"])],
                round.(
                    coco_to_yolo_bbox(ann["bbox"], images[ann["image_id"]][2:end]...),
                    digits=6
                )...
            ]
        )
    end

    if haskey(kwargs, :out)
        open(kwargs[:out], "w") do io
            JSON.print(io, labels)
        end
    end

    return labels
end