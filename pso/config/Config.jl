# Here the configuration of the pipeline is loaded from the config.yml file.
# This metadata is then used when loading a dataset, etc.
module ConfigModule

export Config

struct Config
    num_gen::Int
    pop_size::Int
    genotype_len::Int
    log_freq::Int
    file_name::String
end

end