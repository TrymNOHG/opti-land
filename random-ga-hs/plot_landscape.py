import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv("ensemble_feature.csv")
df = pd.read_csv("lookup_tables\log_reg_feature.csv")
loss_series = df['loss']
loss_series = loss_series.sample(frac=1, random_state=42).reset_index(drop=True)
loss_series = np.log(loss_series + 1e-6)  # Add small epsilon to avoid log(0)

plt.plot(loss_series[:100])
plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('Loss Series')
plt.show()