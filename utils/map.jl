using Knet
import Knet: Data

include("utils.jl")
include("constants.jl")
include("../models.jl")
include("../coco2014.jl")

function get_mAP(model::Darknet, data::COCO14Data)
    MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)
    NUM_CLASSES = length(CLASS_MAP)

    gt_counter_per_class = Dict(Integer, Integer)



    for (img, y_gt) in data
        print(size(img))
    end

end