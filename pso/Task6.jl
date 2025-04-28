include("./config/Config.jl")
using .ConfigModule

include("./config/ConfigLoader.jl")
using .ConfigLoader

include("./Log.jl")
using .Log

include("./PSO_6.jl")
using .PSO6



using Random

conf, lookup_table = load_config("./config/config.yaml")

best, history = @time run_sim(conf, lookup_table)

output_file_name = replace(conf.file_name, ".csv"=>"")
output_file_name = string(split(output_file_name, "/")[end])

plot_history(history, output_file_name)
