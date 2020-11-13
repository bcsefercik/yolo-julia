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


function load_data(images::String, annotations::String; raw_labels::Bool=true)
    for (root, dirs, files) in walkdir(images)
        println("Directories in $root")
        for dir in dirs
            println(joinpath(root, dir)) # path to directories
        end
        # println("Files in $root")
        # for file in files
        #     println(joinpath(root, file)) # path to files
        # end
    end

end


function convert_annotations_to_labels(annotation_file::String; kwargs...)
    annotations = JSON.parsefile(
        annotation_file;
        dicttype=Dict,
        inttype=Int32,
        use_mmap=true
    )

    category_ids = Dict()

    for c in annotations["categories"]
        if !haskey(category_ids, c["id"])
            category_ids[c["id"]] = length(category_ids) + 1
        end
    end

    images = Dict()  # image_id => (file_name, width, height)

    for img in annotations["images"]
        if !haskey(images, img["id"])
            images[img["id"]] = (
                split(img["file_name"], ".")[1],
                img["width"],
                img["height"]
            )
        end
    end

    labels = Dict()

    for ann in annotations["annotations"]
        file_name = images[ann["image_id"]][1]
        if !haskey(labels, file_name)
            labels[file_name] = []
        end

        push!(
            labels[file_name],
            [
                category_ids[ann["category_id"]],
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