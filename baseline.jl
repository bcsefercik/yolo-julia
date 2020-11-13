include("coco2014.jl")

convert_annotations_to_labels("dataset/annotations/instances_val2014.json"; out="val.json")