using CSV, DataFrames, Random, Plots, Statistics,StatsPlots

POP_SIZE = 50
N_GEN = 200
MUT_PROB = 0.2
N_MUT = 0
TOURNAMENT_SIZE = 5
MAX_STAGNATION = 10
ELITE_COUNT = 1

function load_lookup_table(path::String)
    df = CSV.read(path, DataFrame)
    sample_feature = string(df.features[1])
    n_bits = length(sample_feature)
    table = Dict{BitVector, Float64}()
    for row in eachrow(df)
        bin = lpad(string(row.features), n_bits, '0')
        vec = BitVector([c == '1' for c in bin])
        table[vec] = row.loss
    end
    return table, n_bits
end

function get_fitness_lookup(ind, table::Dict{BitVector, Float64})
    key = BitVector(ind)
    if haskey(table, key)
        return table[key] + 0.0001 * count(x -> x == true, key)
    else
        return 2.0
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

function mean_hamming(pop)
    total = 0
    count = 0
    for i in 1:length(pop), j in i+1:length(pop)
        total += sum(pop[i] .!= pop[j])
        count += 1
    end
    return count == 0 ? 0.0 : total / count
end

function run_ga_lookup(filepath::String)
    table, N_BITS = load_lookup_table(filepath)

    if N_BITS > 10
        N_MUT = max(1, round(Int, 0.1 * N_BITS))  
    else 
        N_MUT = 0

    end

    population = initialize_population(POP_SIZE, N_BITS)

    best = deepcopy(population[1])
    best_fitness = get_fitness_lookup(best, table)
    mean_f, min_f, divs = Float64[], Float64[], Float64[]
    stagnation = 0
    mut_prob = MUT_PROB
    α = 0.9
    prev_mean_rmse = nothing

    for gen in 1:N_GEN
        fitness = [get_fitness_lookup(ind, table) for ind in population]
        current_best_idx = argmin(fitness)
        current_best = population[current_best_idx]
        current_best_fitness = fitness[current_best_idx]

        if current_best_fitness < best_fitness
            best = deepcopy(current_best)
            best_fitness = current_best_fitness
            stagnation = 0
            mut_prob = MUT_PROB
        else
            stagnation += 1
        end

        raw_mean_rmse = mean(fitness)
        smooth_mean_rmse = isnothing(prev_mean_rmse) ? raw_mean_rmse : α * prev_mean_rmse + (1 - α) * raw_mean_rmse
        prev_mean_rmse = smooth_mean_rmse

        println("Gen $gen | Best RMSE: $(round(best_fitness, digits=6)) | Mean RMSE: $(round(smooth_mean_rmse, digits=6))")

        push!(mean_f, smooth_mean_rmse)
        push!(min_f, minimum(fitness))
        push!(divs, mean_hamming(population))

        """if stagnation ≥ MAX_STAGNATION
            println("🔁 Stagnazione: reset parziale alla generazione $gen")
            mut_prob *= 1.2
            stagnation = 0
        end"""

        selected = tournament_selection(population, fitness, TOURNAMENT_SIZE)
        offspring = two_point_crossover(selected)
        mutation!(offspring, mut_prob, N_MUT)
        sorted_pop = sort(collect(zip(population, fitness)), by = x -> x[2])
        elites = [copy(x[1]) for x in sorted_pop[1:ELITE_COUNT]]
        population = offspring[1:(POP_SIZE - ELITE_COUNT)]
        append!(population, elites)
    end

    bitstr = join(map(x -> x ? "1" : "0", best))
    println("\n Miglior individuo trovato:")
    println("Bitstring: $bitstr")
    println("Final RMSE (lookup): ", round(best_fitness, digits=6))
    println("Num feature attive: ", sum(best))

    p = plot(mean_f, label="Fitness Media Smooth (RMSE)", lw=2, color=:blue)
    plot!(p, min_f, label="Miglior Fitness (RMSE)", lw=2, color=:red)
    xlabel!("Generazione")
    ylabel!("RMSE")
    title!("Evoluzione Fitness da Lookup Table (Smoothed)")
    display(p)
end

function is_local_optimum(vec::BitVector, table::Dict{BitVector, Float64})
    val = table[vec]
    for i in 1:length(vec)
        neighbor = copy(vec)
        neighbor[i] = !neighbor[i]
        if haskey(table, neighbor)
            if table[neighbor] < val - 1e-6
                return false
            end
        end
    end
    return true
end

function visualize_local_optima(filepath::String)
    table = load_lookup_table(filepath)[1]
    points = collect(keys(table))
    rmse_all = [table[b] for b in points]
    active_all = [count(==(true), b) for b in points]
    local_minima = [(b, table[b], get_fitness_lookup(b, table), count(==(true), b)) for b in points if is_local_optimum(b, table)]
    sort!(local_minima, by = x -> x[2])
    x_vals = [a for (b, raw, penal, a) in local_minima]
    y_vals = [raw for (b, raw, penal, a) in local_minima]
    scatter(active_all, rmse_all, label="Tutti i punti", alpha=0.3, color=:gray, markersize=3)
    scatter!(x_vals, y_vals, label="Minimi Locali", color=:red, markershape=:star5, markersize=6)
    xlabel!("Numero di feature attive")
    ylabel!("RMSE puro")
    title!("Mappa dei Minimi Locali vs Tutti i Punti")
    println("\n Minimi locali trovati: $(length(local_minima))")
    for (bit, raw_val, penalized, a) in local_minima
        println("Bitstring: $(join(map(x -> x ? 1 : 0, bit))) | RMSE puro: $(round(raw_val, digits=6)) | RMSE penalizzato: $(round(penalized, digits=6)) | Feature attive: $a")
    end
    display(current())
end

function visualize_heatmap_distribution(filepath::String)
    table, _ = load_lookup_table(filepath)
    points = collect(keys(table))
    sorted_points = sort(points, by = x -> count(x))
    n_rows = 32
    n_cols = cld(length(sorted_points), n_rows)
    grid_vals = fill(NaN, n_rows, n_cols)
    for (i, vec) in enumerate(sorted_points)
        row = mod1(i, n_rows)
        col = cld(i, n_rows)
        grid_vals[row, col] = get_fitness_lookup(vec, table)
    end
    heatmap(grid_vals, xlabel="Colonna", ylabel="Riga", title="Heatmap Fitness Penalizzata (lookup table)", colorbar_title="RMSE")
    display(current())
end
function benchmark_ga(filepath::String; n_runs=10)
    table, N_BITS = load_lookup_table(filepath)
    best_rmse_vals = Float64[]
    active_features = Int[]
    best_bitstrings = String[]

    for run in 1:n_runs
        println("\n🔁 Run $run/$n_runs")
        population = initialize_population(POP_SIZE, N_BITS)
        best = deepcopy(population[1])
        best_fitness = get_fitness_lookup(best, table)
        stagnation = 0
        mut_prob = MUT_PROB

        for gen in 1:N_GEN
            fitness = [get_fitness_lookup(ind, table) for ind in population]
            current_best_idx = argmin(fitness)
            current_best = population[current_best_idx]
            current_best_fitness = fitness[current_best_idx]

            if current_best_fitness < best_fitness
                best = deepcopy(current_best)
                best_fitness = current_best_fitness
                stagnation = 0
                mut_prob = MUT_PROB
            else
                stagnation += 1
            end

            selected = tournament_selection(population, fitness, TOURNAMENT_SIZE)
            offspring = two_point_crossover(selected)
            mutation!(offspring, mut_prob, N_MUT)
            sorted_pop = sort(collect(zip(population, fitness)), by = x -> x[2])
            elites = [copy(x[1]) for x in sorted_pop[1:ELITE_COUNT]]
            population = offspring[1:(POP_SIZE - ELITE_COUNT)]
            append!(population, elites)
        end

        final_rmse = round(get_fitness_lookup(best, table), digits=6)
        bitstr = join(map(x -> x ? "1" : "0", best))
        println("🎯 Run $run best: $final_rmse | Features: $(sum(best)) | $bitstr")
        push!(best_rmse_vals, final_rmse)
        push!(active_features, sum(best))
        push!(best_bitstrings, bitstr)
    end

    
    println("\n📊 Risultati riassuntivi su $n_runs run:")
    println("Media RMSE: ", round(mean(best_rmse_vals), digits=6))
    println("Deviazione standard: ", round(std(best_rmse_vals), digits=6))
    println("Media feature attive: ", round(mean(active_features), digits=2))
    println("Best assoluto: ", minimum(best_rmse_vals))

    # Tabella finale
    println("\n📋 Tabella dei migliori individui per run:")
    println(rpad("Run", 6), rpad("RMSE", 12), rpad("Feature", 10), "Bitstring")
    for i in 1:n_runs
        println(rpad(string(i), 6), rpad(string(best_rmse_vals[i]), 12),
                rpad(string(active_features[i]), 10), best_bitstrings[i])
    end

   
    boxplot(best_rmse_vals, label="RMSE", ylabel="Valore", title="Distribuzione RMSE su $n_runs run")
    display(current())
    df = DataFrame(Run = 1:n_runs,
               RMSE = best_rmse_vals,
               Features = active_features,
               Bitstring = best_bitstrings)

    CSV.write("benchmark_results.csv", df)
    println("📁 Tabella salvata in benchmark_results.csv")
end



run_ga_lookup("C:/Users/giovy/Desktop/scopiazziamo/log_reg_feature.csv")
visualize_local_optima("C:/Users/giovy/Desktop/scopiazziamo/log_reg_feature.csv")
#benchmark_ga("C:/Users/giovy/Desktop/scopiazziamo/log_reg_feature.csv", n_runs=10)

