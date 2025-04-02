using CSV
using DataFrames
using DecisionTree
using Random
using Plots
using Statistics  # Import the Statistics module for mean

# Funzione per caricare il dataset Iris (dal file locale)
function load_data()
    filepath = "C:/Users/giovy/Desktop/Project3/iris/iris.data" # Assicurati che il file iris.data sia nella stessa cartella dello script
    colnames = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = CSV.File(filepath, header = false)
    data = DataFrame(df)
    rename!(data, colnames)
    return data
end

# Carica il dataset
iris_df = load_data()

# Controlla le dimensioni e i dati letti
println("Dataset loaded with dimensions: ", size(iris_df))
println("First few rows of data: ")
println(first(iris_df, 5))

# Suddividi il dataset in X (caratteristiche) e y (target)
X = Matrix(iris_df[!, 1:4])  # Le prime 4 colonne sono le caratteristiche (convertito in matrice)
y = String.(iris_df[!, :class])  # La colonna target è 'class' (convertito in vettore di stringhe)

# Verifica che X e y siano corretti
println("X dimensions: ", size(X))  # Dovrebbe essere (150, 4)
println("y length: ", length(y))  # Dovrebbe essere 150

function train_and_evaluate(X, y, selected_features)
    println("Selected Features: ", selected_features)  # Stampa le caratteristiche selezionate
    X_selected = X[:, selected_features]
    
    # Creiamo e alleniamo un albero decisionale
    model = DecisionTreeClassifier(max_depth=5)
    fit!(model, X_selected, y)
    
    # Calcoliamo l'accuratezza sul set di addestramento
    predictions = predict(model, X_selected)
    accuracy = mean(predictions .== y)  # Utilizza mean per calcolare la percentuale di previsione corretta
    
    return accuracy
end

# Funzione per creare la tabella di ricerca con tutte le combinazioni di caratteristiche
function create_lookup_table(X, y)
    num_features = size(X, 2)
    lookup_table = []
    
    # Generiamo tutte le possibili combinazioni di caratteristiche
    for mask in 1:(2^num_features - 1)  # Cambiato da 0 a 1 per evitare subset vuoti
        selected_features = findall(x -> (mask & (1 << (x - 1))) != 0, 1:num_features)
        
        # Solo se c'è almeno una caratteristica selezionata
        if length(selected_features) > 0
            accuracy = train_and_evaluate(X, y, selected_features)
            push!(lookup_table, (selected_features, accuracy))
        end
    end
    
    return lookup_table
end

# Funzione di fitness per l'individuo
function fitness_function(individual, lookup_table)
    selected_features = findall(x -> x == true, individual)  # Estrai gli indici delle caratteristiche selezionate
    for (features, accuracy) in lookup_table
        if Set(features) == Set(selected_features)
            return accuracy
        end
    end
    return 0.0  # Se non trovata, restituisce una fitness di 0 (mai ottimale)
end

# Crossover (1-point crossover)
function crossover(parent1, parent2)
    point = rand(1:length(parent1))
    offspring1 = vcat(parent1[1:point], parent2[point+1:end])
    offspring2 = vcat(parent2[1:point], parent1[point+1:end])
    return offspring1, offspring2
end

function mutate(individual, mutation_rate)
    for i in 1:length(individual)
        if rand() < mutation_rate
            individual[i] = !individual[i]
            println("Mutated individual at position $i")  # Stampa quando si verifica una mutazione
        end
    end
    return individual
end


function select_parents(population, lookup_table)
    fitness_values = [fitness_function(ind, lookup_table) for ind in population]
    total_fitness = sum(fitness_values)
    selection_probs = fitness_values ./ total_fitness
    
    # Selezione tramite probabilità (ruota della roulette)
    parent1_idx = aliasing_selection(1:length(population), selection_probs)
    parent2_idx = aliasing_selection(1:length(population), selection_probs)

    println("Selected parents: ", population[parent1_idx], " and ", population[parent2_idx])  # Stampa i genitori selezionati
    
    return population[parent1_idx], population[parent2_idx]
end


# Funzione di selezione con aliasing
function aliasing_selection(population_indices, selection_probs)
    cumulative_probs = cumsum(selection_probs)
    r = rand()  # Genera un numero casuale tra 0 e 1
    idx = findfirst(cumulative_probs .>= r)  # Trova l'indice corrispondente
    return population_indices[idx]
end

# Funzione principale SGA
function sga(num_generations, population_size, num_features, mutation_rate, lookup_table)
    population = [rand(Bool, num_features) for _ in 1:population_size]
    best_solution = nothing
    best_fitness = -Inf

    fitness_history = []

    for generation in 1:num_generations
        # Calcola la fitness di ogni individuo
        fitness_values = [fitness_function(ind, lookup_table) for ind in population]
        
        # Trova la miglior soluzione
        max_fitness_idx = argmax(fitness_values)
        if fitness_values[max_fitness_idx] > best_fitness
            best_fitness = fitness_values[max_fitness_idx]
            best_solution = population[max_fitness_idx]
        end

        # Memorizza la storia della fitness per la visualizzazione
        push!(fitness_history, best_fitness)

        # Stampa la fitness della generazione corrente
        println("Generation $generation - Best Fitness: $best_fitness")

        # Selezione: Seleziona i genitori per la generazione successiva
        new_population = []
        for _ in 1:(population_size ÷ 2)
            parent1, parent2 = select_parents(population, lookup_table)
            
            # Crossover per creare i figli
            offspring1, offspring2 = crossover(parent1, parent2)
            
            # Mutazione
            push!(new_population, mutate(offspring1, mutation_rate))
            push!(new_population, mutate(offspring2, mutation_rate))
        end

        # Aggiorna la popolazione
        population = new_population
    end

    return best_solution, best_fitness, fitness_history
end

# Crea la tabella di ricerca
lookup_table = create_lookup_table(X, y)

# Parametri dell'algoritmo SGA
num_generations = 50
population_size = 20
num_features = 4  # Il dataset Iris ha 4 caratteristiche
mutation_rate = 0.1

# Esegui l'algoritmo SGA
best_solution, best_fitness, fitness_history = sga(num_generations, population_size, num_features, mutation_rate, lookup_table)

# Stampa la miglior soluzione trovata
println("Best solution (selected features): ", best_solution)
println("Best fitness (accuracy): ", best_fitness)

# Visualizza la storia della fitness
plot(1:num_generations, fitness_history, xlabel="Generation", ylabel="Best Fitness", label="SGA", title="Fitness History")
