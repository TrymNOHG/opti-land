import h5py
import pandas as pd

tas1_mat_loc = "8-zoo_rf_mat.h5"

# Load the file
with h5py.File(tas1_mat_loc, 'r') as f:
    data = f['data'][:]

# Transpose to correct shape
data = data.T

# Put into DataFrame
df = pd.DataFrame(data, columns=["accuracy", "time"])

# Create the new DataFrame
new_df = pd.DataFrame({
    'features': [(bin(idx + 1)[2:]).zfill(16) for idx in df.index],
    'loss': 1 - df['accuracy']
})

# Save to CSV
new_df.to_csv('task2_feature.csv', index=False)

print("Saved as task2_feature.csv successfully!")
