import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# UMDA function
def umda(feature_loss_dict,
         num_bits=13,
         pop_size=100,
         select_size=20,
         generations=50,
         min_loss_possible=None,
         seed=None):
    rng = np.random.default_rng(seed)
    p = np.full(num_bits, 0.5)

    history = []
    best_gene = None
    best_loss = np.inf

    cache = {}
    unique_accesses = 0

    for gen in range(1, generations + 1):
        population = rng.random((pop_size, num_bits)) < p
        genes = [''.join(pop.astype(int).astype(str)) for pop in population]
        losses = []

        for g in genes:
            if g in cache:
                loss = cache[g]
            else:
                loss = feature_loss_dict.get(g, np.inf)
                cache[g] = loss
                unique_accesses += 1
            losses.append(loss)
        losses = np.array(losses)

        gen_best_idx = np.argmin(losses)
        gen_best_gene = genes[gen_best_idx]
        gen_best_loss = losses[gen_best_idx]

        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_gene = gen_best_gene

        history.append((losses.mean(), gen_best_loss))

        if min_loss_possible is not None and best_loss <= min_loss_possible:
            break

        sel_idx = np.argsort(losses)[:select_size]
        selected = population[sel_idx]
        p = selected.mean(axis=0)
        eps = 1.0 / pop_size
        p = np.clip(p, eps, 1 - eps)

    return best_gene, best_loss, history, unique_accesses


if __name__ == "__main__":
    # Load lookup table
    df = pd.read_csv("lookup_tables/task1_feature.csv")
    df['features'] = df['features'].astype(str).str.zfill(11)
    feature_loss_dict = dict(zip(df['features'], df['loss']))
    min_loss_possible = min(feature_loss_dict.values())

    avg_unique_accesses_by_popsize = {}

    for pop_size in tqdm(range(2, 101), desc="Evaluating pop_sizes"):
        run_unique_accesses = []

        for run in range(50):
            _, _, _, unique_accesses = umda(
                feature_loss_dict=feature_loss_dict,
                num_bits=11,
                pop_size=pop_size,
                select_size=max(1, pop_size // 6),  # Adjust select_size proportionally
                generations=100,
                min_loss_possible=min_loss_possible,
                seed=run  # Different seed for each run
            )
            run_unique_accesses.append(unique_accesses)

        avg_unique = np.mean(run_unique_accesses)
        avg_unique_accesses_by_popsize[pop_size] = avg_unique

    # Find best pop_size with minimal average unique accesses
    best_pop_size = min(avg_unique_accesses_by_popsize, key=avg_unique_accesses_by_popsize.get)
    best_avg_accesses = avg_unique_accesses_by_popsize[best_pop_size]

    print(f"\n==> Best pop_size: {best_pop_size} with avg unique accesses: {best_avg_accesses:.2f}")

    # Optional: plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(list(avg_unique_accesses_by_popsize.keys()),
             list(avg_unique_accesses_by_popsize.values()),
             marker='o')
    plt.axvline(best_pop_size, color='r', linestyle='--', label=f'Best pop_size = {best_pop_size}')
    plt.xlabel('Population Size')
    plt.ylabel('Average Unique Lookups')
    plt.title('Average Unique Feature Dictionary Accesses vs Population Size')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
