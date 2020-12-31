import JSON
import FileIO

import Images: N0f8, RGB


CLASS_MAP = JSON.parsefile(
    "../class_mappings.json";
    dicttype=Dict{String, Integer},
    inttype=Integer,
    use_mmap=true
)

CLASS_NAMES = JSON.parsefile(
    "../class_names.json";
    dicttype=Dict{String, String},
    inttype=Integer,
    use_mmap=true
)

CLASS_MAP_R = Dict{Integer, String}()
CLASS_NAMES_R = Dict{Integer, String}()
for (k, v) in CLASS_MAP
    CLASS_MAP_R[v] = k
    CLASS_NAMES_R[v] = CLASS_NAMES[k]
end

COLORS = FileIO.load("../colors.jld2")["COLORS"]

# To generate new colors for classes use following line
# COLORS = [RGB{N0f8}(rand(3)...) for (_, _) in CLASS_NAMES_R]

LABEL_COLOR = RGB{N0f8}(0.95,1,0.95)
PRED_COLOR = RGB{N0f8}(0.1,0.4,0.9)
