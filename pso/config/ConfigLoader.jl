module ConfigLoader

using ..ConfigModule

import YAML
using DataStructures

using CSV, DataFrames

export load_config

function load_config(config_file::String)
    data = YAML.load_file(config_file; dicttype=()->DefaultDict{String,Any}(Missing))
    df = CSV.read(data["file_name"], DataFrame)
    
    geno_len = length(digits(maximum(df[!, "features"]))) # Find max and count number of bits.
    
    lookup_table = Dict()
    
    for i in 1:length(df[!, "features"])
        key_str = string(df[!, "features"][i])
        key_str = "0"^(geno_len - length(key_str)) * key_str
        lookup_table[key_str] = df[!, "loss"][i]
    end

    lookup_table["0"^geno_len] = maximum(df[!, "features"])

    conf = Config(
        data["n_generations"],   
        data["population_size"], 
        geno_len,
        data["log_frequency"],
        data["file_name"]
    )

    return conf, lookup_table
end

end
