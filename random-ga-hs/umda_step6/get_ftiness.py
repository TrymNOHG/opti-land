import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

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


def fitness_of_mask_t3(mask_str: str) -> float:
    """
    mask_str: 16-char string of '0'/'1'.  
    df: DataFrame whose last column is the target letter, and first 16 columns are integer features.
    
    Returns: average accuracy over 30 RFs + EPS * num_selected_features.
    """
    EPS = 1/8.0
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


def fitness_of_mask_t2(mask_str: str) -> float:
    """
    mask_str: 16-char string of '0'/'1'.  
    df: DataFrame whose last column is the target letter, and first 16 columns are integer features.
    
    Returns: average accuracy over 30 RFs + EPS * num_selected_features.
    """
    EPS = 1/64.0
    df = pd.read_csv("zoo.data", header=None)
    df = df.drop(columns=0)
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


def fitness_of_mask_t1(mask_str: str, eps: float = 0.0) -> float:
    """
    mask_str: string of '0'/'1' of length = (# features)
    Returns: (1 - avg_accuracy) + eps * num_selected_features
    """
    # --- load, parse “?” as NaN, and impute ---
    df = pd.read_csv("processed.cleveland.data", header=None,
                     na_values='?')
    # separate feature columns vs target
    X = df.iloc[:, :-1]
    y = df.iloc[:,  -1].astype(int)

    # impute all missing with the most-frequent value in each column
    imputer = SimpleImputer(strategy="most_frequent")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )

    # --- check mask length and apply ---
    if len(mask_str) != X_imputed.shape[1]:
        raise ValueError(f"mask length must be {X_imputed.shape[1]}")
    mask = np.array([c=='1' for c in mask_str], dtype=bool)
    num_features = mask.sum()
    if num_features == 0:
        return 0.0

    X_sel = X_imputed.values[:, mask]

    # --- one stratified split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y,
        train_size=0.70,
        random_state=SPLIT_SEED,
        stratify=y
    )

    # --- train N_FORESTS random-forests and record accuracy ---
    accuracies = []
    for _ in range(N_FORESTS):
        clf = RandomForestClassifier(
            **RF_PARAMS,
            random_state=RF_SEED
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    avg_acc = np.mean(accuracies)

    # lower fitness is better: (1 − accuracy) plus small penalty on #features
    return (1 - avg_acc) + eps * num_features

# mask_str = "1110111010110"
# print(fitness_of_mask_t1(mask_str))

# df = pd.read_csv("processed.cleveland.data", header=None)
# print(df)