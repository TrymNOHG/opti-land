import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
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

# Categorical and numerical features (for preprocessing in the ensemble case)
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Has_Cabin', 'Title', 'IsAlone', 'Age', 'Parch', 'Fare']
numerical_features = [col for col in all_features if col not in categorical_features]

# Function to train SVM model on a subset of features and return loss
def evaluate_feature_subset_svm(feature_indices, features_list, X_train, X_test, y_train, y_test):
    # Select only the specified features
    selected_features = [features_list[i] for i in feature_indices]
    
    X_train_subset = X_train[selected_features]
    X_test_subset = X_test[selected_features]
    
    # Create and train model
    model = SVC(probability=True, random_state=42)
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

# Function to train ensemble model on a subset of features and return loss
def evaluate_feature_subset_ensemble(feature_indices, features_list, X_train, X_test, y_train, y_test):
    # Select only the specified features
    selected_features = [features_list[i] for i in feature_indices]
    selected_cat_features = [f for f in categorical_features if f in selected_features]
    selected_num_features = [f for f in numerical_features if f in selected_features]
    
    X_train_subset = X_train[selected_features]
    X_test_subset = X_test[selected_features]
    
    # Define preprocessor based on selected features
    preprocessor = None
    if selected_num_features and selected_cat_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), selected_num_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), selected_cat_features)
            ])
    elif selected_num_features:
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), selected_num_features)])
    elif selected_cat_features:
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), selected_cat_features)])
    
    # Define base models for the ensemble
    knn = KNeighborsClassifier()
    cat = CatBoostClassifier(verbose=0, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(probability=True, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    nn3 = MLPClassifier(
        hidden_layer_sizes=(64, 64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.0003,
        batch_size=32,
        learning_rate_init=0.0005,
        max_iter=200,
        early_stopping=True,
        random_state=42
    )
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('knn', knn),
            ('cat', cat),
            ('rf', rf),
            ('svm', svm),
            ('lr', lr),
            ('nn3', nn3)
        ],
        voting='soft'
    )
    
    # Create pipeline with preprocessing if necessary
    if preprocessor:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', ensemble)
        ])
    else:
        pipeline = Pipeline(steps=[
            ('classifier', ensemble)
        ])
    
    # Train model
    pipeline.fit(X_train_subset, y_train)
    
    # Get probabilities for log loss calculation
    y_pred_proba = pipeline.predict_proba(X_test_subset)
    
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

# Define maximum number of features to evaluate (to manage computational complexity)
max_features_to_evaluate = min(10, len(all_features))  # Limiting to at most 10 features for the ensemble

# Initialize results storage
svm_results = []
ensemble_results = []

# Run SVM evaluation
print("\n=== Running SVM Feature Selection ===")
svm_start_time = time.time()

try:
    # Iterate through feature combinations (from 1 feature to all features)
    for r in range(1, len(all_features) + 1):
        print(f"\nEvaluating SVM with {r} features...")
        # Generate all combinations of r features
        feature_combinations = list(itertools.combinations(range(len(all_features)), r))
        
        for feature_indices in tqdm(feature_combinations, desc=f"SVM - {r} features", ncols=100):
            feature_string, loss = evaluate_feature_subset_svm(
                feature_indices, all_features, X_train, X_test, y_train, y_test
            )
            svm_results.append({"features": feature_string, "loss": loss})
        
        # Print progress
        svm_elapsed = time.time() - svm_start_time
        print(f"SVM Progress: Completed {r}/{len(all_features)} feature counts - Elapsed: {svm_elapsed:.2f}s")
except KeyboardInterrupt:
    print("\nSVM evaluation interrupted by user. Saving results collected so far...")

# Create and save SVM results DataFrame
svm_results_df = pd.DataFrame(svm_results)
svm_results_df = svm_results_df.sort_values("loss")
svm_output_file = "svm_feature_subset_evaluation.csv"
svm_results_df.to_csv(svm_output_file, index=False)

print(f"\nSVM Evaluation complete! Results saved to {svm_output_file}")
print(f"Total SVM feature subsets evaluated: {len(svm_results)}")
if len(svm_results) > 0:
    print(f"Best SVM feature subset: {svm_results_df.iloc[0]['features']} with loss: {svm_results_df.iloc[0]['loss']:.6f}")
print(f"Total SVM time: {time.time() - svm_start_time:.2f} seconds")

# Run Ensemble evaluation with limited feature combinations due to computational cost
print("\n=== Running Ensemble Feature Selection ===")
ensemble_start_time = time.time()

try:
    # Iterate through limited feature combinations (from 1 feature to max_features_to_evaluate)
    for r in range(1, max_features_to_evaluate + 1):
        print(f"\nEvaluating Ensemble with {r} features...")
        # Generate all combinations of r features
        feature_combinations = list(itertools.combinations(range(len(all_features)), r))
        
        for feature_indices in tqdm(feature_combinations, desc=f"Ensemble - {r} features", ncols=100):
            feature_string, loss = evaluate_feature_subset_ensemble(
                feature_indices, all_features, X_train, X_test, y_train, y_test
            )
            ensemble_results.append({"features": feature_string, "loss": loss})
        
        # Print progress
        ensemble_elapsed = time.time() - ensemble_start_time
        print(f"Ensemble Progress: Completed {r}/{max_features_to_evaluate} feature counts - Elapsed: {ensemble_elapsed:.2f}s")
except KeyboardInterrupt:
    print("\nEnsemble evaluation interrupted by user. Saving results collected so far...")

# Create and save Ensemble results DataFrame
ensemble_results_df = pd.DataFrame(ensemble_results)
ensemble_results_df = ensemble_results_df.sort_values("loss")
ensemble_output_file = "ensemble_feature_subset_evaluation.csv"
ensemble_results_df.to_csv(ensemble_output_file, index=False)

print(f"\nEnsemble Evaluation complete! Results saved to {ensemble_output_file}")
print(f"Total Ensemble feature subsets evaluated: {len(ensemble_results)}")
if len(ensemble_results) > 0:
    print(f"Best Ensemble feature subset: {ensemble_results_df.iloc[0]['features']} with loss: {ensemble_results_df.iloc[0]['loss']:.6f}")
print(f"Total Ensemble time: {time.time() - ensemble_start_time:.2f} seconds")

# Show feature importance based on frequency in top results for both models
if len(svm_results) > 0:
    top_n_svm = min(20, len(svm_results))
    top_svm_results = svm_results_df.head(top_n_svm)
    print(f"\nFeature importance based on top {top_n_svm} SVM results:")

    svm_feature_counts = np.zeros(len(all_features))
    for _, row in top_svm_results.iterrows():
        feature_binary = row['features']
        for i, bit in enumerate(feature_binary):
            if bit == '1':
                svm_feature_counts[i] += 1

    svm_feature_importance = [(all_features[i], count / top_n_svm) for i, count in enumerate(svm_feature_counts)]
    svm_feature_importance.sort(key=lambda x: x[1], reverse=True)

    for feature, importance in svm_feature_importance:
        print(f"{feature}: {importance:.2f}")

if len(ensemble_results) > 0:
    top_n_ensemble = min(20, len(ensemble_results))
    top_ensemble_results = ensemble_results_df.head(top_n_ensemble)
    print(f"\nFeature importance based on top {top_n_ensemble} Ensemble results:")

    ensemble_feature_counts = np.zeros(len(all_features))
    for _, row in top_ensemble_results.iterrows():
        feature_binary = row['features']
        for i, bit in enumerate(feature_binary):
            if bit == '1':
                ensemble_feature_counts[i] += 1

    ensemble_feature_importance = [(all_features[i], count / top_n_ensemble) for i, count in enumerate(ensemble_feature_counts)]
    ensemble_feature_importance.sort(key=lambda x: x[1], reverse=True)

    for feature, importance in ensemble_feature_importance:
        print(f"{feature}: {importance:.2f}")