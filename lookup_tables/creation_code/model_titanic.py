import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

# Load dataset
data = pd.read_csv('train_processed.csv')
print("Dataset preview:")
print(data.head())

# Identify target and features
target = 'Survived'
features = [col for col in data.columns if col != target]

X = data[features]
y = data[target]

# Categorical and numerical features
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Has_Cabin', 'Title', 'IsAlone', 'Age', 'Parch', 'Fare']
numerical_features = [col for col in features if col not in categorical_features]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Define base models
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

# Standard ensemble using voting
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

# Define stacking ensembles with different final estimators
stacking_lr = StackingClassifier(
    estimators=[
        ('knn', knn),
        ('cat', cat),
        ('rf', rf),
        ('svm', svm),
        ('lr', lr),
        ('nn3', nn3)
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=None
)

stacking_rf = StackingClassifier(
    estimators=[
        ('knn', knn),
        ('cat', cat),
        ('rf', rf),
        ('svm', svm),
        ('lr', lr),
        ('nn3', nn3)
    ],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv=None
)

stacking_cat = StackingClassifier(
    estimators=[
        ('knn', knn),
        ('cat', cat),
        ('rf', rf),
        ('svm', svm),
        ('lr', lr),
        ('nn3', nn3)
    ],
    final_estimator=CatBoostClassifier(verbose=0, random_state=42),
    cv=None
)

# All models for evaluation
models = {
    'KNN': knn,
    'CatBoost': cat,
    'Random Forest': rf,
    'SVM': svm,
    'Logistic Regression': lr,
    'Neural Network 3': nn3,
    'Voting Ensemble': ensemble,
    'Stacking Ensemble (LR)': stacking_lr,
    'Stacking Ensemble (RF)': stacking_rf,
    'Stacking Ensemble (Cat)': stacking_cat,
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train and evaluate
accuracy_scores = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[model_name] = accuracy
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Final accuracy summary
print("\n=== Final Accuracy Scores ===")
for name, acc in sorted(accuracy_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.4f}")