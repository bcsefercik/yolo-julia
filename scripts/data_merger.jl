import Pkg

Pkg.activate("../Project.toml")

using ArgParse

include("../coco2014.jl")


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--out"
            help = "Model file path to save the trained model."
            arg_type = String
            required = true
        "--data"
            help = "COCO2014Data files for training."
            action = :append_arg
            nargs = '+'
            required = true

    end

    return parse_args(s)
end


function main()
    args = parse_commandline()

    data = merge_data(args["data"][1])
    @info "Merged all data."

    save_data(data, args["out"])

end


main()

