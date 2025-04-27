import h5py
import pandas as pd

task1_mat_loc = "./8-zoo_rf_mat.h5"
task2_mat_loc = "./5-heart-c_rf_mat.h5"

with h5py.File(task2_mat_loc, 'r') as f:
    data = f['data'][:]

data = data.T

df = pd.DataFrame(data, columns=["accuracy", "time"])

new_df = pd.DataFrame({
    'features': [(bin(idx + 1)[2:]).zfill(16) for idx in df.index],
    'loss': 1 - df['accuracy']
})

new_df.to_csv('./benchmark_2.csv', index=False)

print("Saved as task2_feature.csv successfully!")
