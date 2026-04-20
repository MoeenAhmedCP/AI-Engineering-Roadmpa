"""
2.3 ML Algorithms — Exercises
5 exercises + solutions at the bottom.

Run with: python exercises.py
"""

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

try:
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# EXERCISE 1: k-NN from Scratch
# ---------------------------------------------------------------------------
# Implement k-NN classifier using numpy.
# Predict class by majority vote among k nearest neighbors (Euclidean distance).

def knn_predict(X_train, y_train, X_test, k=3):
    """
    k-NN classification from scratch.

    Args:
        X_train: np.array shape (n_train, n_features)
        y_train: np.array shape (n_train,) — integer class labels
        X_test:  np.array shape (n_test, n_features)
        k:       number of nearest neighbors

    Returns:
        predictions: np.array shape (n_test,) — predicted class labels
    """
    predictions = []
    for test_point in X_test:
        # Compute Euclidean distance from test_point to all training points
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))

        # Get indices of k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_indices]

        # Majority vote
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_class = unique[np.argmax(counts)]
        predictions.append(predicted_class)

    return np.array(predictions)


# ---------------------------------------------------------------------------
# EXERCISE 2: Elbow Method (ASCII Bar Chart)
# ---------------------------------------------------------------------------
# For k=1..8, fit KMeans, collect inertia, print as ASCII bar chart.

def elbow_method(X, k_range=range(1, 9)):
    """
    Find the optimal k using the elbow method.
    Prints an ASCII bar chart of inertia vs k.
    """
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    max_inertia = max(inertias)
    bar_width = 40

    print(f"  {'k':>4} | {'Inertia':>12} | Bar")
    print(f"  {'-'*4} | {'-'*12} | {'-'*bar_width}")
    for k, inertia in zip(k_range, inertias):
        filled = int((inertia / max_inertia) * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"  {k:>4} | {inertia:>12.1f} | {bar}")

    return inertias


# ---------------------------------------------------------------------------
# EXERCISE 3: best_model
# ---------------------------------------------------------------------------
# Returns name and model with highest accuracy on validation set.

def best_model(models_dict, X_val, y_val):
    """
    Find the best model by validation accuracy.

    Args:
        models_dict: dict mapping model_name (str) -> fitted model
        X_val:       validation features
        y_val:       validation labels

    Returns:
        (best_name, best_model_object)
    """
    best_name = None
    best_acc = -1
    best_model_obj = None

    for name, model in models_dict.items():
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"    {name:<25} accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model_obj = model

    print(f"\n    Winner: {best_name} (accuracy={best_acc:.4f})")
    return best_name, best_model_obj


# ---------------------------------------------------------------------------
# EXERCISE 4: One-Hot Encode from Scratch
# ---------------------------------------------------------------------------
# No pandas or sklearn. Returns dict mapping value to binary vector.

def one_hot_encode(categories):
    """
    One-hot encode a list of categorical values.

    Args:
        categories: list of values (e.g., ['red', 'blue', 'red', 'green'])

    Returns:
        dict: {value: binary_vector} for each unique value
        Also prints the encoding for each input value.

    Example:
        one_hot_encode(['cat', 'dog', 'bird'])
        -> {'cat': [1,0,0], 'dog': [0,1,0], 'bird': [0,0,1]}
    """
    unique_values = sorted(set(categories), key=str)  # sort for determinism
    n = len(unique_values)
    value_to_index = {val: i for i, val in enumerate(unique_values)}

    encoding = {}
    for val in unique_values:
        vector = [0] * n
        vector[value_to_index[val]] = 1
        encoding[val] = vector

    return encoding


# ---------------------------------------------------------------------------
# EXERCISE 5: Feature Importance Report
# ---------------------------------------------------------------------------
# Works for RandomForest and GradientBoosting, prints top-5 features.

def feature_importance_report(model, feature_names=None, top_n=5):
    """
    Print top-N most important features for RandomForest or GradientBoosting.

    Args:
        model:         a fitted sklearn ensemble model
        feature_names: list of feature name strings (optional)
        top_n:         number of top features to print (default 5)
    """
    importances = model.feature_importances_
    n_features = len(importances)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Sort descending
    sorted_indices = np.argsort(importances)[::-1]
    top_indices = sorted_indices[:top_n]

    model_type = type(model).__name__
    print(f"  Feature Importance Report — {model_type}")
    print(f"  {'Rank':<6} {'Feature':<20} {'Importance':>12} {'Bar'}")
    print(f"  {'-'*6} {'-'*20} {'-'*12} {'-'*25}")

    max_imp = importances[top_indices[0]]
    for rank, idx in enumerate(top_indices, 1):
        bar_len = int((importances[idx] / max_imp) * 25)
        bar = "#" * bar_len
        print(f"  {rank:<6} {feature_names[idx]:<20} {importances[idx]:>12.4f} {bar}")


# ---------------------------------------------------------------------------
# SOLUTIONS — Run all exercises with demonstrations
# ---------------------------------------------------------------------------

def solutions():
    if not NUMPY_OK or not SKLEARN_OK:
        print("Install numpy and scikit-learn to run solutions:")
        print("  pip install numpy scikit-learn")
        return

    np.random.seed(42)

    # --- Generate shared data ---
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=6,
        n_classes=2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    # --- Exercise 1: k-NN from Scratch ---
    print_section("Exercise 1: k-NN from Scratch (k=5)")
    knn_preds = knn_predict(X_train, y_train, X_val, k=5)
    knn_acc = accuracy_score(y_val, knn_preds)
    print(f"  k-NN (k=5) accuracy on val set: {knn_acc:.4f}")
    print(f"  Predictions (first 10): {knn_preds[:10]}")
    print(f"  True labels (first 10): {y_val[:10]}")

    # Compare with sklearn's k-NN
    try:
        from sklearn.neighbors import KNeighborsClassifier
        skl_knn = KNeighborsClassifier(n_neighbors=5)
        skl_knn.fit(X_train, y_train)
        skl_acc = accuracy_score(y_val, skl_knn.predict(X_val))
        print(f"  sklearn k-NN accuracy (sanity check): {skl_acc:.4f}")
    except ImportError:
        pass

    # --- Exercise 2: Elbow Method ---
    print_section("Exercise 2: Elbow Method (k=1..8)")
    from sklearn.datasets import make_blobs
    X_cluster, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=42)
    inertias = elbow_method(X_cluster)
    print()
    print("  Look for the 'elbow' — the k where the curve bends sharply.")
    print("  After the elbow, adding more clusters gives diminishing returns.")

    # --- Exercise 3: best_model ---
    print_section("Exercise 3: best_model()")
    models_to_compare = {
        "RandomForest(n=50)": RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train),
        "RandomForest(n=100)": RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42).fit(X_train, y_train),
    }
    winner_name, winner_model = best_model(models_to_compare, X_val, y_val)

    # --- Exercise 4: One-Hot Encode ---
    print_section("Exercise 4: one_hot_encode()")
    colors = ["red", "blue", "green", "red", "blue"]
    encoding = one_hot_encode(colors)
    unique_vals = sorted(set(colors), key=str)
    print(f"  Vocabulary (sorted): {unique_vals}")
    print(f"  Encodings:")
    for val in unique_vals:
        print(f"    '{val}': {encoding[val]}")
    print()
    # Show encoding of full list
    print(f"  Encoding each input:")
    for val in colors:
        print(f"    '{val}' -> {encoding[val]}")

    # --- Exercise 5: Feature Importance Report ---
    print_section("Exercise 5: feature_importance_report()")
    feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print("  [RandomForest]")
    feature_importance_report(rf, feature_names=feature_names, top_n=5)

    print()
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    print("  [GradientBoosting]")
    feature_importance_report(gb, feature_names=feature_names, top_n=5)


if __name__ == "__main__":
    solutions()
