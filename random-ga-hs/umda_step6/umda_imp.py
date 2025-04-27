import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_ftiness import fitness_of_mask_t1, fitness_of_mask_t3, fitness_of_mask_t1

# UMDA for binary feature selection minimizing loss via a lookup dict
def umda(
         num_bits=11,
         pop_size=100,
         select_size=20,
         generations=50,
         min_loss_possible=None,
         seed=None):
    """
    Run a Univariate Marginal Distribution Algorithm (UMDA), tracking unique lookups.
    Terminates early if best_loss reaches min_loss_possible.

    Args:
      feature_loss_dict: dict mapping bit-string gene -> fitness (loss) to minimize
      num_bits: length of each gene (11 features)
      pop_size: number of individuals per generation
      select_size: how many best to select each generation
      generations: max number of iterations
      min_loss_possible: known lowest possible loss (early stopping target)
      seed: random seed for reproducibility

    Returns:
      best_gene: bit-string of the best individual found
      best_loss: its associated loss
      history: list of (mean_loss, best_loss) per generation
      unique_accesses: total unique lookups into feature_loss_dict
    """
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
                # loss = feature_loss_dict.get(g, np.inf)
                loss = fitness_of_mask_t1(g)
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

        print(f"Gen {gen:3d}: mean_loss={history[-1][0]:.4f}, best_loss={history[-1][1]:.4f}, unique_accesses={unique_accesses}")

        if min_loss_possible is not None and best_loss <= min_loss_possible:
            print(f"Early stopping: best_loss reached min_loss_possible ({min_loss_possible})")
            break

        sel_idx = np.argsort(losses)[:select_size]
        selected = population[sel_idx]
        p = selected.mean(axis=0)
        eps = 1.0 / pop_size
        p = np.clip(p, eps, 1 - eps)

    return best_gene, best_loss, history, unique_accesses


if __name__ == "__main__":
    # df = pd.read_csv("lookup_tables/task2_feature.csv")
    # df['features'] = df['features'].astype(str).str.zfill(11)
    # feature_loss_dict = dict(zip(df['features'], df['loss']))
    min_loss_possible = 0
    pop_size_inp = 20
    best_gene, best_loss, hist, unique_lookups = umda(
        num_bits=13,
        pop_size=pop_size_inp,
        select_size=max(1, pop_size_inp // 6),
        generations=10,
        min_loss_possible=min_loss_possible,
        seed=123
    )

    print(f"\n==> Best gene: {best_gene} with loss {best_loss:.4f}")
    print(f"Total unique feature_loss_dict accesses: {unique_lookups}")
    print(f"Min possible loss: {min_loss_possible}")
    mean_losses, best_losses = zip(*hist)
    plt.figure(figsize=(10, 6))
    plt.plot(mean_losses, label='Mean Loss', linestyle='--', marker='o')
    plt.plot(best_losses, label='Best Loss', linestyle='-', marker='x')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.title('UMDA: Mean and Best Loss Over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
