import pandas as pd
import matplotlib.pyplot as plt
import os

files = ["svm_feature", "ensemble_feature", "log_reg_feature"]

for f in files:
    for color in ["blue", "red"]:
        if color == "red":
            filename = f"fitness_{f}.csv"
        else:
            filename = f"fitness_{f}_dominated.csv"

        df = pd.read_csv(filename)
        df["fitness_length"] = df["fitness_length"].astype(int)

        legend = "Pareto optimal" if color == "red" else "Dominated solution"
        plt.scatter(
            df["fitness_length"],
            df["fitness_ml_metric"],
            alpha=0.7,
            c=color,
            label=legend,
        )
        if color == "blue":
            plt.xticks(sorted(df["fitness_length"].unique()))
        plt.xlabel("Number of features")
        plt.ylabel("Loss")
        plt.title(f)
        plt.grid(True)
        os.makedirs("plots", exist_ok=True)
    plt.legend()
    plt.savefig(f"plots/{f}.png")
    plt.close()
