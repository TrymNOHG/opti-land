import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm
import time

# Load dataset
print("Loading dataset...")
data = pd.read_csv('train_processed.csv')

# Identify target and features for the main dataset
target = 'Survived'
features = [col for col in data.columns if col != target]

X = data[features]
y = data[target]

# Split data (use stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing used in your model_titanic code
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Has_Cabin', 'Title', 'IsAlone', 'Age', 'Parch', 'Fare']
numerical_features = [col for col in features if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# Define the six base models (in fixed order)
base_models = [
    ('KNN', KNeighborsClassifier()),
    ('CatBoost', CatBoostClassifier(verbose=0, random_state=42)),
    ('Random Forest',  VotingClassifier(estimators=[])),  # not used here—use RandomForestClassifier instead!
    # We'll use RandomForestClassifier instead of an inner voting classifier.
    ('Random Forest',  # placeholder name and model
     # Use RandomForestClassifier as defined in model_titanic.py:
     # Since VotingClassifier takes tuple (name, estimator), we create below:
     # Note: Changing duplicate key name is not allowed so I use same string for consistency.
     None),
    ('SVM', SVC(probability=True, random_state=42)),
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Neural Network 3', MLPClassifier(
        hidden_layer_sizes=(64, 64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.0003,
        batch_size=32,
        learning_rate_init=0.0005,
        max_iter=200,
        early_stopping=True,
        random_state=42
    ))
]

# Instead of having duplicate names, we define our six base models in order:
model_names = ['KNN', 'CatBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'Neural Network 3']
model_instances = [
    KNeighborsClassifier(),
    CatBoostClassifier(verbose=0, random_state=42),
    # Random Forest
    # Using RandomForestClassifier with 100 trees as in your original script
    # (this substitutes for your random forest model)
    __import__('sklearn.ensemble').ensemble.RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(probability=True, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42),
    MLPClassifier(
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
]

# Total number of models for ensemble
n_models = len(model_instances)
print(f"Evaluating ensembles for {n_models} base models.")

# Create a list of all non-empty binary strings of length 6
all_combinations = [combo for combo in product([0, 1], repeat=n_models) if any(combo)]
total_combinations = len(all_combinations)
print(f"Total ensemble subsets to evaluate: {total_combinations}")

results = []
start_time = time.time()

# Iterate over all possible model combinations
for combo in tqdm(all_combinations, desc="Evaluating model ensembles", ncols=100):
    binary_str = ''.join(str(bit) for bit in combo)
    
    # Select the base models corresponding to ones with a 1 in the binary string
    selected_estimators = []
    for include, name, estimator in zip(combo, model_names, model_instances):
        if include:
            selected_estimators.append((name, estimator))
    
    # If no models are selected, skip (shouldn't happen due to filter)
    if not selected_estimators:
        continue
    
    # Create the Voting ensemble with selected models and soft voting
    ensemble_model = VotingClassifier(estimators=selected_estimators, voting='soft')
    
    # Build pipeline with preprocessor and our ensemble_model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ensemble_model)
    ])
    
    try:
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)
        loss = log_loss(y_test, y_pred_proba)
    except Exception as e:
        loss = np.nan   # in case an error arises during training
    results.append({"features": binary_str, "loss": loss})

# Convert results to DataFrame and sort by loss
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("loss")

# Save results CSV file
output_file = "ensemble_model_subset_evaluation.csv"
results_df.to_csv(output_file, index=False)

elapsed = time.time() - start_time
print(f"\nEvaluation complete! Total ensembles evaluated: {len(results)}")
print(f"Results saved to {output_file}")
print(f"Total time: {elapsed:.2f} seconds")

# Optionally: Print best ensemble details
if not results_df.empty:
    best_row = results_df.iloc[0]
    print(f"Best ensemble binary string: {best_row['features']} with loss: {best_row['loss']:.6f}")
