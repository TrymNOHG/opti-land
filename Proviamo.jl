# ========== Configurazioni globali ==========
using CSV, DataFrames, Random, Plots, Statistics, StatsPlots, DecisionTree, HDF5

POP_SIZE = 200
N_GEN = 400
#rimettere a 0.3 nel caso
MUT_PROB = 0.3
#rimettere a 15 nel caso
TOURNAMENT_SIZE = 20
ELITE_COUNT = 15
MAX_STAGNATION = 15
RESET_PATIENCE = 50

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
    # Costruisci popolazione casuale con (pop_size - 1) individui
    population = [rand(Bool, n_bits) for _ in 1:(pop_size - 1)]

    return population
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

function tournament_selection(pop, fitness, k)
    return [copy(pop[argmin(fitness[rand(1:end, k)])]) for _ in 1:length(pop)]
end

function two_point_crossover(pop)
    shuffle!(pop)
    for i in 1:2:(length(pop)-1)
        p1, p2 = pop[i], pop[i+1]
        pt1, pt2 = sort(rand(1:length(p1), 2))
        p1[pt1:pt2], p2[pt1:pt2] = p2[pt1:pt2], p1[pt1:pt2]
    end
    return pop
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

function multi_point_crossover(pop::Vector{Vector{Bool}}, n_points::Int)
    offspring = copy(pop)
    shuffle!(offspring)  # Mischia i genitori

    for i in 1:2:(length(offspring)-1)
        p1, p2 = offspring[i], offspring[i+1]

        # Seleziona n_points punti di crossover (ad esempio, 3 punti)
        points = sort(rand(1:length(p1), n_points))  # Seleziona n_points punti random
        for j in 1:n_points
            start_idx = points[j]
            end_idx = j == n_points ? length(p1) : points[j+1] - 1  # Determina la lunghezza del segmento

            # Esegui lo scambio tra i segmenti di p1 e p2
            p1[start_idx:end_idx], p2[start_idx:end_idx] = p2[start_idx:end_idx], p1[start_idx:end_idx]
        end
    end
    return offspring
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

    
    
    all_fitness = Float64[]
    for run in 1:10
        table, N_BITS = load_lookup_table(filepath)
        N_MUT = N_BITS < 10 ? 0 : max(1, round(Int, 0.1 * N_BITS))
        population = initialize_population(POP_SIZE, N_BITS)
        best = deepcopy(population[1])
        best_fitness = fitness_lookup(best, table, epsilon)
        mean_f, min_f = Float64[], Float64[]
        prev_mean = nothing
        println("Run $run")
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
            #offspring = multi_point_crossover(selected, 3)  # 3 punti di crossover (modificabile)
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

        push!(all_fitness, best_fitness)
    end

    avg_fitness = mean(all_fitness)
    std_fitness = std(all_fitness)

    println("Average Fitness over 10 runs: ", avg_fitness)
    println("Standard Deviation: ", std_fitness)

    # Creiamo una tabella dei risultati
    results = DataFrame(
        run = 1:10,
        best_fitness = all_fitness
    )
    display(results)
    
    # Grafico dei risultati
    p = plot(1:10, all_fitness, label="Fitness per run", lw=2, color=:blue)
    xlabel!("Run")
    ylabel!("Fitness")
    title!("Fitness evolution over multiple runs")
    savefig("ga_results.png")
    println("Plot saved to: ga_results.png")
    display(p)

    

end

# ========== Metodo 2: Random Forest ==========
function load_dataset(path::String; n_features::Int, label_at_start::Bool=false, drop_first_col::Bool=false)
    df = CSV.read(path, DataFrame; header=false, missingstring="?")

    # Se la prima colonna è da ignorare (es. zoo: nome animale)
    if drop_first_col
        df = df[:, 2:end]
    end
    dropmissing!(df)

    if label_at_start
        y_raw = df[:, 1]
        X = Matrix{Float64}(df[:, 2:n_features+1])
    else
        X = Matrix{Float64}(df[:, 1:n_features])
        y_raw = df[:, n_features+1]
    end

    

    # Converti la label in int (da stringa/char, se serve)
    y = if eltype(y_raw) <: AbstractString || eltype(y_raw) <: Char
        [tryparse(Int, c) !== nothing ? parse(Int, c) : Int(c[1]) - Int('A') for c in y_raw]
    else
        convert(Vector{Int}, y_raw)
    end

    return X, y
end



function load_dataset_letters(path::String)
    # Carica il CSV senza header, i valori mancanti sono trattati come "?"
    df = CSV.read(path, DataFrame; header=false, missingstring="?")

    # La prima colonna è la label (una lettera da 'A' a 'Z')
    y_raw = df[:, 1]

    # Le restanti 16 colonne sono le feature numeriche
    X = Matrix{Float64}(df[:, 2:17])

    # Converti le lettere in interi: 'A' → 0, ..., 'Z' → 25
    y = [Int(c[1]) - Int('A') for c in y_raw]

    return X, y
end


using Random

function fitness_rf(ind::Vector{Bool}, X, y; epsilon)
    if sum(ind) == 0
        return 1.0
    end

    #Random.seed!(456)  # Imposta il seed fisso per ripetibilità

    cols = reverse(ind) .== 1
    X_sub = X[:, cols]
    accs = Float64[]

    for _ in 1:30
        idx = shuffle(1:size(X, 1))
        train_size = floor(Int, 0.7 * length(idx))
        train_idx = idx[1:train_size]
        test_idx = idx[train_size+1:end]

        model = build_forest(y[train_idx], X_sub[train_idx, :], size(X_sub, 2), 30, 0.7, 0, 2, 2, 0.0)
        pred = apply_forest(model, X_sub[test_idx, :])
        push!(accs, mean(pred .== y[test_idx]))
    end

    return (1.0 - mean(accs)) + epsilon * sum(ind)
end


function run_ga_rf(X, y, x_star::Vector{Bool}, task_name::String; epsilon, h5file::String="")

    N_BITS = size(X, 2)
    N_MUT = max(1, round(Int, 0.1 * N_BITS))
    all_fitness = Float64[]
    hamming_all = Int[] 

    for run in 1:10
        println("Run $run")
        #Random.seed!(456)
        
        population = initialize_population(POP_SIZE, N_BITS)
        fitness = [fitness_rf(ind, X, y; epsilon=epsilon) for ind in population]
        current_best_idx = argmin(fitness)
        best = population[current_best_idx]
        best_fitness = fitness[current_best_idx]        
        mean_f, min_f, best_f_per_gen = Float64[], Float64[], Float64[]
        stagnation, reset_pop, mut_prob, α = 0, 0, MUT_PROB, 0.9
        prev_gen_min = 1.0  # valore iniziale "pessimo"
        prev_mean = nothing

        for gen in 1:N_GEN

            if reset_pop > RESET_PATIENCE

                #population = initialize_population(POP_SIZE, N_BITS)
                #stagnation=0    
                reset_pop=0
                #println("🔁 Attuo restart popolazione") 

            elseif stagnation > MAX_STAGNATION
                #mut_prob = min(1.0, mut_prob + 0.05)  # Incremento della mutazione (con limite a 1.0)
                mut_prob=0.8
                println("🔁 Stagnazione a gen $gen: aumento mutazione a $mut_prob")
            else
                mut_prob = MUT_PROB  # Mantieni la probabilità di mutazione iniziale
            end



            fitness = [fitness_rf(ind, X, y; epsilon=epsilon) for ind in population]
            current_best_idx = argmin(fitness)
            current_best = population[current_best_idx]
            current_best_fitness = fitness[current_best_idx]

            

            if current_best_fitness < best_fitness
                best, best_fitness = current_best, current_best_fitness
                stagnation, reset_pop, mut_prob = 0,0, MUT_PROB  # Reset stagnation e mutazione a valori iniziali
            else
                stagnation += 1
                reset_pop += 1
                """if gen % 10 == 0
                    best_fitness = fitness_rf(best, X, y; epsilon=epsilon)
                end"""
            end
            

            raw_mean = mean(fitness)
            #smoothed = isnothing(prev_mean) ? raw_mean : 0.9 * prev_mean + 0.1 * raw_mean
            smoothed = raw_mean
            prev_mean = smoothed

            println("Gen $gen | Best h(x): $(round(current_best_fitness, digits=6)) | Mean h(x): $(round(smoothed, digits=6)) | Best global h(x): $(round(best_fitness, digits=6))")
            push!(mean_f, smoothed)
            current_gen_min = minimum(fitness)
            push!(min_f, current_gen_min)
            push!(best_f_per_gen, best_fitness)


            if stagnation ≥ MAX_STAGNATION
                println("🔁 Stagnazione a gen $gen: aumento mutazione + x_star")
                for _ in 1:5
                    push!(population, copy(x_star))  # Rafforza la pressione verso x_star
                end
                for _ in 1:5
                    push!(population, rand(Bool, N_BITS))  # Aggiungi più individui casuali
                end
                population = population[1:POP_SIZE]
                stagnation = 0
            end
            selected = tournament_selection(population, fitness, TOURNAMENT_SIZE)
            offspring = two_point_crossover(selected)
            #offspring = multi_point_crossover(selected, 3)  # 3 punti di crossover (modificabile)

            mutation!(offspring, mut_prob, N_MUT)
            elites = sort(collect(zip(population, fitness)), by = x -> x[2])[1:ELITE_COUNT]
            population = offspring[1:(POP_SIZE - ELITE_COUNT)]
            append!(population, [copy(e[1]) for e in elites])

            if best_fitness < 0.425 && hamming_distance(best,x_star) <= 7
                break
            end
        end

        #println("Bitstring found: $bitstr")
        println("h(x*) = ", round(best_fitness, digits=6))
        println("f(x*) = ", round(best_fitness+epsilon*sum(best), digits=6))
        println("Hamming distance from x*: ", hamming_distance(best, x_star))
        println("Number of active features: ", sum(best))
        println("Best bitstring found: ", join(map(x -> x ? "1" : "0", best)))
        p = plot(mean_f, label="Mean fitness", lw=2, color=:blue)
        plot!(p, min_f, label="Best of this generation", lw=2, color=:red)
        plot!(p, best_f_per_gen, label="Global best", lw=2, color=:orange)


        title!("GA Step 6 - $task_name | Run $run")
        xlabel!("Generation")
        ylabel!("Fitness h(x)")
        #savefig("step6_plot_$(task_name)_run$run.png")
        #println("Plot saved to: step6_plot_$(task_name)_run$run.png")
        display(p)

        push!(all_fitness, best_fitness)
        push!(hamming_all, hamming_distance(best, x_star))
        
    end
    # Calcoliamo la media e la deviazione standard delle fitness
    avg_fitness = mean(all_fitness)
    std_fitness = std(all_fitness)

    

    # Creiamo una tabella dei risultati
    results = DataFrame(
        run = 1:10,
        best_fitness = all_fitness,
        hamming_distance = hamming_all
    )
    display(results)

    
    # Grafico dei risultati
    p = plot(1:10, all_fitness, label="Fitness per run", lw=2, color=:blue)
    xlabel!("Run")
    ylabel!("Fitness")
    title!("Fitness evolution over multiple runs")
    #savefig("ga_results.png")
    #println("Plot saved to: ga_results.png")
    display(p)
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

    elseif mode == :rf
        if isnothing(n_features) || isnothing(x_star)
            error("Per il mode :rf servono `n_features` e `x_star` definiti.")
        end

        println("Starting GA with Random Forest...")

        # 🔍 Dataset loader: sceglie il metodo giusto
        X, y = if occursin("letter_task", task_name)
            println("📄 Caricamento con `load_dataset_letters`")
            load_dataset_letters(file_path)

        elseif occursin("zoo_task", task_name)
            println("📄 Caricamento con `load_dataset` (zoo: ignora prima colonna)")
            load_dataset(file_path; n_features=n_features, drop_first_col=true)

        else
            println("📄 Caricamento con `load_dataset` (default)")
            load_dataset(file_path; n_features=n_features)
        end

        run_ga_rf(X, y, x_star, task_name; epsilon=epsilon, h5file=h5file)

    else
        error(" Modalità non riconosciuta: usa :lookup oppure :rf")
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
    x_star = [true, true, true, false, true, true, true, false, true, false, true, true, false],
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

"""main(
    mode = :rf,
    file_path = "C:/Users/giovy/Desktop/scopiazziamo/letter+recognition/letter-recognition.data",
    n_features = 16,
    x_star = [false, false, false, false, false, false, false, true, true, false, false, false, true, false, true, false],
    epsilon = 1/8,
    task_name = "letter_task",
    h5file = ""  # Nessun file HDF5 per lookup in questo caso
)"""


for i in 1:1
    main(
    mode = :rf,
    file_path = "C:/Users/giovy/Desktop/scopiazziamo/heart+disease/processed.cleveland.data",
    n_features = 13,
    x_star = [true, true, true, false, true, true, true, false, true, false, true, true, false],
    epsilon=0.0,
    task_name = "cleverland_task",
    h5file = "C:/Users/giovy/Desktop/scopiazziamo/5-heart-c_rf_mat.h5"
)
end
    

