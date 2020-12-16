using Pkg

Pkg.activate("Project.toml")


#=include("coco2014.jl")

# convert_annotations_to_labels(
#     "dataset/annotations/instances_val2014.json";
#     out="val_labels.json",
#     class_mappings="class_mappings.json"
# )

load_data(
    "/home/bcs/Desktop/MSc/repos/yolo-julia/dataset/images/val2014",
    "val_labels.json",
    class_file="class_mappings.json"
)=#



#=include("utils/nn.jl")

import .NN
=#

include("utils/parse_config.jl")

mdefs = parse_model_cfg("yolov3.cfg")
