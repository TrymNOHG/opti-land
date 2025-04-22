# ========== Configurazioni globali ==========
using CSV, DataFrames, Random, Plots, Statistics, StatsPlots, DecisionTree, HDF5

POP_SIZE = 200
N_GEN = 200
MUT_PROB = 0.2
TOURNAMENT_SIZE = 20
ELITE_COUNT = 6

# ========== Utility ==========
function hamming_distance(a::Vector{Bool}, b::Vector{Bool})
    return sum(a .!= b)
end

function check_individual_in_h5(bitstr::Vector{Bool}, h5_path::String, epsilon::Float64)
    decimal = parse(Int, join(reverse(Int.(bitstr))), base=2)  # reverse = crescita da destra
    h5open(h5_path, "r") do file
        data = read(file["data"])
        acc = data[decimal, 1]
        error = 1.0 - acc
        penalty = epsilon * sum(bitstr)
        return error + penalty
    end
end


function initialize_population(pop_size, n_bits)
    return [rand(Bool, n_bits) for _ in 1:pop_size]
end

function tournament_selection(pop, fitness, k)
    selected = Vector{Vector{Bool}}()
    for _ in 1:length(pop)
        candidates = rand(1:length(pop), k)
        best_idx = argmin(fitness[candidates])
        push!(selected, copy(pop[candidates[best_idx]]))
    end
    return selected
end

function two_point_crossover(pop)
    offspring = copy(pop)
    shuffle!(offspring)
    for i in 1:2:(length(pop)-1)
        p1, p2 = offspring[i], offspring[i+1]
        pt1, pt2 = sort(rand(1:length(p1), 2))
        p1[pt1:pt2], p2[pt1:pt2] = p2[pt1:pt2], p1[pt1:pt2]
    end
    return offspring
end

function mutation!(pop, mut_prob, n_flips)
    for ind in pop
        for _ in 1:n_flips
            i = rand(1:length(ind))
            if rand() < mut_prob
                ind[i] = !ind[i]
            end
        end
    end
end

# ========== Metodo 1: Lookup Table ==========
function load_lookup_table(path::String)
    df = CSV.read(path, DataFrame)
    n_bits = length(string(df.features[1]))
    table = Dict{BitVector, Float64}()
    for row in eachrow(df)
        vec = BitVector([c == '1' for c in string(row.features)])
        table[vec] = row.loss
    end
    return table, n_bits
end

function fitness_lookup(ind::Union{Vector{Bool}, BitVector}, table::Dict{BitVector, Float64}, epsilon=0.0)
    key = BitVector(ind)
    raw = get(table, key, 2.0)
    return raw + epsilon * sum(ind)
end


function run_ga_lookup(filepath::String; epsilon=0.0, task_name::String="lookup")

    table, N_BITS = load_lookup_table(filepath)
    N_MUT = N_BITS < 10 ? 0 : max(1, round(Int, 0.1 * N_BITS))
    population = initialize_population(POP_SIZE, N_BITS)

    best = deepcopy(population[1])
    best_fitness = fitness_lookup(best, table, epsilon)
    mean_f, min_f = Float64[], Float64[]
    prev_mean = nothing
    
    for gen in 1:N_GEN
        fitness = [fitness_lookup(ind, table, epsilon) for ind in population]
        current_best_idx = argmin(fitness)
        current_best = population[current_best_idx]
        if fitness[current_best_idx] < best_fitness
            best = deepcopy(current_best)
            best_fitness = fitness[current_best_idx]
        end

        raw_mean = mean(fitness)
        smoothed = isnothing(prev_mean) ? raw_mean : 0.9 * prev_mean + 0.1 * raw_mean
        prev_mean = smoothed

        println("Gen $gen | Best RMSE: $(round(best_fitness, digits=6)) | Mean: $(round(smoothed, digits=6))")
        push!(mean_f, smoothed)
        push!(min_f, minimum(fitness))

        selected = tournament_selection(population, fitness, TOURNAMENT_SIZE)
        offspring = two_point_crossover(selected)
        mutation!(offspring, MUT_PROB, N_MUT)
        elites = sort(collect(zip(population, fitness)), by = x -> x[2])[1:ELITE_COUNT]
        population = offspring[1:(POP_SIZE - ELITE_COUNT)]
        append!(population, [copy(e[1]) for e in elites])
    end

    println("\nFinal bitstring: ", join(map(x -> x ? "1" : "0", best)))
    println("Final RMSE: ", round(best_fitness, digits=6))
    p = plot(mean_f, label="Mean RMSE", lw=2, color=:blue)
    plot!(p, min_f, label="Best RMSE", lw=2, color=:red)
    title!("GA Lookup - Fitness Evolution")
    xlabel!("Generation")
    ylabel!("RMSE")
    savefig("step6_plot_$(replace(task_name, " " => "_")).png")
    println("Plot saved to: step6_plot_$(replace(task_name, " " => "_")).png")
    display(p)

end

# ========== Metodo 2: Random Forest ==========
function load_dataset(path::String; n_features::Int)
    df = CSV.read(path, DataFrame; header=false, missingstring="?")
    if eltype(df[!, 1]) <: AbstractString
        df = df[:, 2:end]
    end
    for c in names(df)
        if eltype(df[!, c]) <: AbstractString
            df[!, c] = passmissing(parse).(Float64, df[!, c])
        end
    end
    dropmissing!(df)
    X = Matrix{Float64}(df[:, 1:n_features])
    y = convert(Vector{Int}, df[:, n_features+1])
    return X, y
end

function fitness_rf(ind::Vector{Bool}, X, y; epsilon=0.0)
    if sum(ind) == 0
        return 1.0
    end
    cols = reverse(ind) .== 1
    X_sub = X[:, cols]
    accs = Float64[]
    for _ in 1:30
        idx = shuffle(1:size(X, 1))
        train = idx[1:floor(Int, 0.7 * length(idx))]
        test = idx[(floor(Int, 0.7 * length(idx)) + 1):end]
        n_subf = max(1, floor(Int, sqrt(size(X_sub, 2))))
        model = build_forest(y[train], X_sub[train, :], n_subf, 30, 0.7, 0, 2, 2, 0.0; rng=Random.GLOBAL_RNG)
        pred = apply_forest(model, X_sub[test, :])
        push!(accs, mean(pred .== y[test]))
    end
    return (1.0 - mean(accs)) + epsilon * sum(ind)
end

function run_ga_rf(X, y, x_star::Vector{Bool}, task_name::String; epsilon=0.0, h5file::String="")

    N_BITS = size(X, 2)
    N_MUT = max(1, round(Int, 0.1 * N_BITS))
    population = initialize_population(POP_SIZE, N_BITS)
    best = deepcopy(population[1])
    best_fitness = fitness_rf(best, X, y; epsilon=epsilon)
    mean_f, min_f = Float64[], Float64[]
    prev_mean = nothing

    for gen in 1:N_GEN
        fitness = [fitness_rf(ind, X, y; epsilon=epsilon) for ind in population]
        current_best_idx = argmin(fitness)
        current_best = population[current_best_idx]
        if fitness[current_best_idx] < best_fitness
            best = deepcopy(current_best)
            best_fitness = fitness[current_best_idx]
        end

        raw_mean = mean(fitness)
        smoothed = isnothing(prev_mean) ? raw_mean : 0.9 * prev_mean + 0.1 * raw_mean
        prev_mean = smoothed

        println("Gen $gen | Best h(x): $(round(best_fitness, digits=6)) | Mean h(x): $(round(smoothed, digits=6))")
        push!(mean_f, smoothed)
        push!(min_f, minimum(fitness))

        selected = tournament_selection(population, fitness, TOURNAMENT_SIZE)
        offspring = two_point_crossover(selected)
        mutation!(offspring, MUT_PROB, N_MUT)
        elites = sort(collect(zip(population, fitness)), by = x -> x[2])[1:ELITE_COUNT]
        population = offspring[1:(POP_SIZE - ELITE_COUNT)]
        append!(population, [copy(e[1]) for e in elites])
    end

    bitstr = join(map(x -> x ? "1" : "0", best))
    println("\nStep 6 Results - $task_name")
    if h5file != ""
        hx_h5 = check_individual_in_h5(best, h5file, epsilon)
        println("Checking HDF5: h(x) = $(round(hx_h5, digits=6))")

        if abs(hx_h5 - best_fitness) < 1e-2
            println("✅ Value approved with HDF5 (within tolerance).")
        else
            println("⚠️ Value NOT approved with HDF5 (too different).")
        end
        
    end
    println("Bitstring found: $bitstr")
    println("h(x*) = ", round(best_fitness, digits=6))
    println("Number of active features: ", sum(best))
    println("Hamming distance from x*: ", hamming_distance(best, x_star))
    p = plot(mean_f, label="Mean", lw=2, color=:blue)
    plot!(p, min_f, label="Best", lw=2, color=:red)
    title!("GA Step 6 - $task_name")
    savefig("step6_plot_$task_name.png")
    println("Plot saved to: step6_plot_$task_name.png")
    
end

function is_local_optimum_with_penalty(vec::BitVector, table::Dict{BitVector, Float64}, epsilon::Float64)
    current_fitness = fitness_lookup(Vector{Bool}(vec), table, epsilon)
    for i in 1:length(vec)
        neighbor = copy(vec)
        neighbor[i] = !neighbor[i]
        if haskey(table, neighbor)
            neighbor_fitness = fitness_lookup(Vector{Bool}(neighbor), table, epsilon)
            if neighbor_fitness < current_fitness - 1e-6
                return false
            end
        end
    end
    return true
end


function visualize_local_optima_with_penalty(filepath::String; epsilon::Float64=0.0)
    table, _ = load_lookup_table(filepath)
    points = collect(keys(table))

    # Tutti i punti con penalità
    penalized_all = [fitness_lookup(Vector{Bool}(b), table, epsilon) for b in points]
    active_all = [count(==(true), b) for b in points]

    # Minimi locali rispetto alla fitness penalizzata
    local_minima = [(b, table[b], fitness_lookup(Vector{Bool}(b), table, epsilon), count(==(true), b))
                    for b in points if is_local_optimum_with_penalty(b, table, epsilon)]
    sort!(local_minima, by = x -> x[3])

    # Estrazione per scatter plot
    x_vals = [a for (b, raw, penal, a) in local_minima]
    y_vals = [penal for (b, raw, penal, a) in local_minima]

    # Plot con penalità per tutti
    scatter(active_all, penalized_all, label="All points", alpha=0.3, color=:gray, markersize=3)
    scatter!(x_vals, y_vals, label="Local minimum", color=:red, markershape=:star5, markersize=6)
    xlabel!("Feature active")
    ylabel!("Fitness (RMSE + penalty)")
    title!("Local minim map")

    # Output in console
    println("Local minimum found: $(length(local_minima))")
    for (bit, raw_val, penalized, a) in local_minima
        println("Bitstring: $(join(map(x -> x ? 1 : 0, bit))) | RMSE(no penalty): $(round(raw_val, digits=6)) | RMSE: $(round(penalized, digits=6)) | Feature active: $a")
    end

    # Salva e mostra il grafico
    savefig("local_optima_with_penalty.png")
    println("Plot saved as local_optima_with_penalty.png")
    display(current())
end



function main(; 
    mode::Symbol = :rf,
    file_path::String = "",
    x_star::Union{Vector{Bool}, Nothing} = nothing,
    n_features::Union{Int, Nothing} = nothing,
    task_name::String = "task",
    epsilon::Float64 = 0.0,
    h5file::String = ""
)
    if mode == :lookup
        println("Starting GA with lookup table")
        run_ga_lookup(file_path; epsilon=epsilon, task_name=task_name)
        visualize_local_optima_with_penalty(file_path; epsilon)
    elseif mode == :rf
        if isnothing(n_features) || isnothing(x_star)
            error("Per il mode :rf servono `n_features` e `x_star` definiti.")
        end
        println("Starting GA with Random Forest...")
        X, y = load_dataset(file_path; n_features=n_features)
        run_ga_rf(X, y, x_star, task_name; epsilon=epsilon, h5file=h5file)
    else
        error(" Modalità non riconosciuta: usa :lookup oppure :rf")
    end
end


function run_experiments()
    open("ga_results_summary.txt", "w") do io
        for run in 1:10
            println(io, "\n==========================")
            println(io, "Run $run")
            println(io, "==========================")
            redirect_stdout(io) do
                try
                    main(
                        mode = :rf,  
                        file_path = "C:/Users/giovy/Desktop/scopiazziamo/heart+disease/processed.cleveland.data",
                        n_features = 13,
                        x_star = [false, false, true, true, false, false, true, false, true, false, false, true, false],
                        epsilon = 0.0,
                        task_name = "cleverland_task_run_$run",
                        h5file = "C:/Users/giovy/Desktop/scopiazziamo/5-heart-c_rf_mat.h5"
                    )
                catch e
                    println(io, "⚠️ ERROR in run $run: $e")
                end
            end
        end
    end
end


"""main(
    mode = :lookup,
    file_path = "C:/Users/giovy/Desktop/scopiazziamo/log_reg_feature.csv",
    epsilon = 0.1,
    task_name = "lookup_test"
)"""

"""main(
    mode = :lookup,
    file_path = "C:/Users/giovy/Desktop/scopiazziamo/svm_feature.csv",
    epsilon = 0.1,
    task_name = "lookup_test"
)"""

"""main(
    mode = :lookup,
    file_path = "C:/Users/giovy/Desktop/scopiazziamo/ensemble_feature.csv",
    epsilon = 0.1,
    task_name = "lookup_test"
)"""

"""main(
    mode = :rf,
    file_path = "C:/Users/giovy/Desktop/scopiazziamo/heart+disease/processed.cleveland.data",
    n_features = 13,
    x_star = [false, false, true, true, false, false, true, false, true, false, false, true, false],
    epsilon=0.0,
    task_name = "cleverland_task",
    h5file = "C:/Users/giovy/Desktop/scopiazziamo/5-heart-c_rf_mat.h5"
)"""



"""main(
        mode = :rf,
        file_path = "C:/Users/giovy/Desktop/scopiazziamo/zoo/zoo.data",
        n_features = 16,
        x_star = [false,false,false,false,false,false,false,true,false,false,false,false,false,false,true,false],
        epsilon = 1/64,
        task_name = "zoo_task",
        h5file = "C:/Users/giovy/Desktop/scopiazziamo/8-zoo_rf_mat.h5"
    )"""


for i in 1:10
    main(
    mode = :rf,
    file_path = "C:/Users/giovy/Desktop/scopiazziamo/heart+disease/processed.cleveland.data",
    n_features = 13,
    x_star = [false, false, true, true, false, false, true, false, true, false, false, true, false],
    epsilon=0.0,
    task_name = "cleverland_task",
    h5file = "C:/Users/giovy/Desktop/scopiazziamo/5-heart-c_rf_mat.h5"
    )
end 


#run_experiments()
