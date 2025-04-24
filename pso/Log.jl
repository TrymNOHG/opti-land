module Log

    using Plots

    export History, plot_history

    mutable struct History
        avg_swarm_loss::Vector{Float64}
        best_global_loss::Vector{Float64}
    end

    function plot_history(history::History, file_name::String)
        num_gen = length(history.avg_swarm_loss)
        x_val = range(0, num_gen, length=num_gen)
        
        i = plot(x_val, history.avg_swarm_loss, title="Average Swarm Loss", label="Avg Loss") # Add title on axis and diagram...
        xlabel!(i, "Generation")
        ylabel!(i, "Loss")
        savefig(i, "./log/"* file_name * "_average_swarm_loss.png") 

        p = plot(x_val, history.best_global_loss, title="Best Global Swarm Loss", label="Global Swarm Loss") # Add title on axis and diagram...
        xlabel!(p, "Generation")
        ylabel!(p, "Loss")
        plot!(i, x_val, history.best_global_loss, title="Overview of Swarm Loss Progression", label="Global Swarm Loss") # Add title on axis and diagram...

        savefig(p, "./log/" * file_name * "_best_swarm_loss.png") 
        savefig(i, "./log/" * file_name * "_combined_loss.png") 
    end

end