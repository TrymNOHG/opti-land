import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import itertools
from tqdm import tqdm
import time

# Load dataset
print("Loading dataset...")
data = pd.read_csv('train_processed.csv')

# Identify target and features
target = 'Survived'
all_features = [col for col in data.columns if col != target]

# Print feature information
print(f"Target variable: {target}")
print(f"Features available ({len(all_features)}): {all_features}")

X = data[all_features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Function to train model on a subset of features and return loss
def evaluate_feature_subset(feature_indices, features_list, X_train, X_test, y_train, y_test):
    # Select only the specified features
    selected_features = [features_list[i] for i in feature_indices]
    
    X_train_subset = X_train[selected_features]
    X_test_subset = X_test[selected_features]
    
    # Create and train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_subset, y_train)
    
    # Get probabilities for log loss calculation
    y_pred_proba = model.predict_proba(X_test_subset)
    
    # Calculate log loss
    loss = log_loss(y_test, y_pred_proba)
    
    # Create binary string representation of selected features
    feature_string = ""
    for i in range(len(features_list)):
        if i in feature_indices:
            feature_string += "1"
        else:
            feature_string += "0"
    
    return feature_string, loss

# Initialize results storage
results = []

# Total number of possible feature combinations (excluding empty set)
total_combinations = 2**len(all_features) - 1
print(f"Total feature subsets to evaluate: {total_combinations}")

# Use tqdm for progress tracking
start_time = time.time()
try:
    # Iterate through all possible feature combinations (from 1 feature to all features)
    for r in range(1, len(all_features) + 1):
        print(f"\nEvaluating combinations with {r} features...")
        # Generate all combinations of r features
        feature_combinations = list(itertools.combinations(range(len(all_features)), r))
        
        for feature_indices in tqdm(feature_combinations, desc=f"{r} features", ncols=100):
            feature_string, loss = evaluate_feature_subset(
                feature_indices, all_features, X_train, X_test, y_train, y_test
            )
            results.append({"features": feature_string, "loss": loss})
        
        # Show progress summary after each feature count level
        elapsed = time.time() - start_time
        combinations_done = sum(len(list(itertools.combinations(range(len(all_features)), i))) for i in range(1, r+1))
        progress_pct = combinations_done / total_combinations * 100
        print(f"Progress: {combinations_done}/{total_combinations} ({progress_pct:.2f}%) - Elapsed: {elapsed:.2f}s")

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Saving results collected so far...")

# Create and save results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("loss")
output_file = "feature_subset_evaluation.csv"
results_df.to_csv(output_file, index=False)

print(f"\nEvaluation complete! Results saved to {output_file}")
print(f"Total feature subsets evaluated: {len(results)}")
print(f"Best feature subset: {results_df.iloc[0]['features']} with loss: {results_df.iloc[0]['loss']:.6f}")
print(f"Total time: {time.time() - start_time:.2f} seconds")

# Show feature importance based on frequency in top results
top_n = min(20, len(results))
top_results = results_df.head(top_n)
print(f"\nFeature importance based on top {top_n} results:")

feature_counts = np.zeros(len(all_features))
for _, row in top_results.iterrows():
    feature_binary = row['features']
    for i, bit in enumerate(feature_binary):
        if bit == '1':
            feature_counts[i] += 1

feature_importance = [(all_features[i], count / top_n) for i, count in enumerate(feature_counts)]
feature_importance.sort(key=lambda x: x[1], reverse=True)

for feature, importance in feature_importance:
    print(f"{feature}: {importance:.2f}")