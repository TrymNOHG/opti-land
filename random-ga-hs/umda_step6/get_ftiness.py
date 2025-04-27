import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# global constants
N_FORESTS    = 5
RF_PARAMS    = {
    "n_estimators":       30,
    "max_depth":          None,
    "min_impurity_decrease": 0.0,
    "min_samples_split":  2,
    "max_features":       None,
    "criterion":          "gini",
}
RF_SEED       = 456
SPLIT_SEED    = 123
EPS           = 1/8.0

def fitness_of_mask(mask_str: str) -> float:
    """
    mask_str: 16-char string of '0'/'1'.  
    df: DataFrame whose last column is the target letter, and first 16 columns are integer features.
    
    Returns: average accuracy over 30 RFs + EPS * num_selected_features.
    """
    df = pd.read_csv("letter-recognition.data", header=None)
    first_col = df.iloc[:, 0]
    rest      = df.iloc[:, 1:]
    df        = pd.concat([rest, first_col], axis=1)
    # --- parse mask and select columns ---
    if len(mask_str) != df.shape[1] - 1:
        raise ValueError(f"mask length must be {df.shape[1]-1}")
    mask = np.array([int(c) for c in mask_str], dtype=bool)
    num_features = mask.sum()
    if num_features == 0:
        return 0.0  # no features → no predictive power
    
    # feature matrix and target vector
    X = df.iloc[:, :-1].values     # all 16 features
    y = df.iloc[:,  -1].values     # target letter
    X_sel = X[:, mask]             # keep only masked-in columns

    # split once
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y,
        train_size=0.70,
        random_state=SPLIT_SEED,
        stratify=y
    )

    # --- train N_FORESTS forests and record accuracy ---
    accuracies = []
    for i in range(N_FORESTS):
        # use the same seed for reproducibility
        clf = RandomForestClassifier(
            **RF_PARAMS,
            random_state=RF_SEED
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    avg_acc = np.mean(accuracies)

    # fitness = average accuracy + eps * num_features
    fitness = (1-avg_acc) + EPS * num_features
    return fitness


# mask = "0000000110001010"
# print("fitness(", mask, ") =", fitness_of_mask(mask))
