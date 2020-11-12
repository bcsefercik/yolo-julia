

struct COCO14Data
    x
    y
    batchsize::Int
    shuffle::Bool
    num_instances::Int

    function COCO14Data(
        x, y; batchsize::Int=100, shuffle::Bool=false, dtype::Type=Array{Float64})

        return new(
            convert(dtype, x),
            convert(dtype, y),
            batchsize,
            shuffle,
            size(x)[2]
        )
    end
end

