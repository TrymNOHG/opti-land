import pandas as pd

# Load your data
df = pd.read_csv("non_eps_task2_feature.csv")

# Define epsilon
eps = 1 / 64

# Make sure 'features' is treated as string
df['features'] = df['features'].astype(str)

# Count number of '1's in each 'features' string
num_ones = df['features'].apply(lambda x: x.count('1'))

# Modify the 'loss' column
df['loss'] = df['loss'] + eps * num_ones

df.to_csv("task2_feature.csv", index=False)
