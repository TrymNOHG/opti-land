import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fitness_log.csv")

plt.scatter(df["fitness_length"], df["fitness_ml_metric"], alpha=0.7)
plt.xlabel("Fitness Length")
plt.ylabel("Fitness ML Metric")
plt.title("Fitness Length vs ML Metric")
plt.grid(True)
plt.show()
