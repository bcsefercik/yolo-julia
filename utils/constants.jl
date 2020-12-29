import JSON

using Colors


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

COLORS = [RGB{N0f8}(rand(3)...) for (_, _) in CLASS_NAMES_R]
