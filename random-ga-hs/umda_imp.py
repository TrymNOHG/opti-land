import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# UMDA for binary feature selection minimizing loss via a lookup dict

def umda(feature_loss_dict,
         num_bits=11,
         pop_size=100,
         select_size=20,
         generations=50,
         seed=None):
    """
    Run a Univariate Marginal Distribution Algorithm (UMDA).

    Args:
      feature_loss_dict: dict mapping bit-string gene -> fitness (loss) to minimize
      num_bits: length of each gene (11 features)
      pop_size: number of individuals per generation
      select_size: how many best to select each generation
      generations: number of iterations
      seed: random seed for reproducibility

    Returns:
      best_gene: bit-string of the best individual found
      best_loss: its associated loss
      history: list of (mean_loss, best_loss) per generation
    """
    # rng = np.random.default_rng(seed)
    rng = np.random.default_rng()

    # Initialize probability vector to 0.5
    p = np.full(num_bits, 0.5)

    history = []
    best_gene = None
    best_loss = np.inf

    for gen in range(1, generations + 1):
        # Sampling: generate pop_size individuals
        # Each bit is 1 with probability p[i]
        population = rng.random((pop_size, num_bits)) < p
        # Convert to string keys and evaluate
        genes = [''.join(pop.astype(int).astype(str)) for pop in population]
        losses = np.array([feature_loss_dict.get(g, np.inf) for g in genes])

        # Identify best in this generation
        gen_best_idx = np.argmin(losses)
        gen_best_gene = genes[gen_best_idx]
        gen_best_loss = losses[gen_best_idx]

        # Update global best
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_gene = gen_best_gene

        # Record history
        history.append((losses.mean(), gen_best_loss))

        # Selection: choose select_size lowest-loss individuals
        sel_idx = np.argsort(losses)[:select_size]
        selected = population[sel_idx]

        # Update probabilities: average of selected bits
        p = selected.mean(axis=0)

        # (Optional) enforce margins to avoid p==0 or 1
        eps = 1.0 / (pop_size)
        p = np.clip(p, eps, 1 - eps)

        print(f"Gen {gen:3d}: mean_loss={history[-1][0]:.4f}, best_loss={history[-1][1]:.4f}, p[:5]={p[:5]}...")

    return best_gene, best_loss, history


if __name__ == "__main__":
    # Example usage — assume feature_loss_dict already defined elsewhere
    # feature_loss_dict = { '00000000001': 0.5101, ... }
    df = pd.read_csv("lookup_tables/svm_feature.csv")
    df['features'] = df['features'].astype(str).apply(lambda x: x.zfill(11))
    feature_loss_dict = dict(zip(df['features'], df['loss']))

    best_gene, best_loss, hist = umda(
        feature_loss_dict=feature_loss_dict,
        num_bits=11,
        pop_size=30,
        select_size=5,
        generations=10,
        seed=123
    )
    print(f"\n==> Best gene: {best_gene} with loss {best_loss:.4f}")

    # Plotting best_loss and mean_loss over generations
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