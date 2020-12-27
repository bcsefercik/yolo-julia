# module coco2014

import Pkg

import Base: length
import FileIO

import JSON

include("coco2014_utils.jl")


struct COCO14Data
    x
    y

    function COCO14Data(
        images::String, label_file::String;
        raw_labels::Bool=false, class_file::String="", dtype=Array{Float32}
    )
        x, y = load_data_raw(
            images, label_file;
            raw_labels=raw_labels, class_file=class_file, dtype=dtype
        )
        return new(x, y)
    end
end


function length(cd::COCO14Data)
    return length(cd.y)
end


function save_data(cd::COCO14Data, filename::String)
    FileIO.save(filename, "cd2014", cd)
    @info "Saved to: $filename."
end


function load_data(filename::String)
    od_d = FileIO.load(filename)
    @info "Loaded: $filename."
    return od_d["cd2014"]
end

# export COCO14Data, length, save_data, load_data

# end  # module end
