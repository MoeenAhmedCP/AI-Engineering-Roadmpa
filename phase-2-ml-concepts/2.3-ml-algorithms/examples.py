"""
2.3 ML Algorithms — Examples
Uses scikit-learn with try/except ImportError.

Covers:
- RandomForest vs GradientBoosting classification comparison
- KMeans clustering with cluster centers
- GridSearchCV hyperparameter tuning
- Model comparison table with timing
"""

import time

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False
    print("numpy not installed. Run: pip install numpy")

try:
    from sklearn.datasets import make_classification, make_blobs
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("scikit-learn not installed. Run: pip install scikit-learn")


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def ascii_bar(value, max_value, width=30):
    """Return an ASCII bar chart segment."""
    filled = int((value / max_value) * width)
    return "#" * filled + "-" * (width - filled)


# ---------------------------------------------------------------------------
# PART 1: RandomForest vs GradientBoosting
# ---------------------------------------------------------------------------

def compare_classifiers():
    np.random.seed(42)

    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "model": model,
            "accuracy": acc,
            "train_time": train_time,
            "predict_time": predict_time,
        }

    # --- Comparison Table ---
    print(f"  {'Model':<20} {'Accuracy':>10} {'Train(s)':>10} {'Predict(s)':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12}")
    for name, res in results.items():
        print(f"  {name:<20} {res['accuracy']:>10.4f} {res['train_time']:>10.4f} {res['predict_time']:>12.6f}")

    # --- Feature Importance (Random Forest) ---
    rf = results["RandomForest"]["model"]
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:5]

    print("\n  Top-5 Feature Importances (RandomForest):")
    print(f"  {'Feature':<12} {'Importance':>12} {'Bar'}")
    print(f"  {'-'*12} {'-'*12} {'-'*30}")
    max_imp = importances[top_indices[0]]
    for idx in top_indices:
        bar = ascii_bar(importances[idx], max_imp)
        print(f"  feature_{idx:<4} {importances[idx]:>12.4f} {bar}")

    return results


# ---------------------------------------------------------------------------
# PART 2: KMeans Clustering
# ---------------------------------------------------------------------------

def kmeans_demo():
    np.random.seed(42)

    # Generate 2D data with 3 natural clusters
    X, true_labels = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

    # Fit KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X)

    print("  Cluster centers (x, y):")
    for i, center in enumerate(km.cluster_centers_):
        count = (km.labels_ == i).sum()
        print(f"    Cluster {i}: ({center[0]:>7.3f}, {center[1]:>7.3f}) — {count} points")

    print(f"\n  Inertia (sum of squared distances to centroids): {km.inertia_:.4f}")

    # Check alignment with true labels (assignment may differ)
    # Just show distribution per cluster
    print("\n  True label distribution per cluster:")
    for i in range(3):
        mask = km.labels_ == i
        true_in_cluster = true_labels[mask]
        counts = {label: (true_in_cluster == label).sum() for label in range(3)}
        print(f"    KMeans cluster {i}: {counts}")


# ---------------------------------------------------------------------------
# PART 3: GridSearchCV
# ---------------------------------------------------------------------------

def gridsearch_demo():
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=15, n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, None],
    }

    rf = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)

    print(f"  Best parameters: {gs.best_params_}")
    print(f"  Best CV accuracy: {gs.best_score_:.4f}")

    # Test set score
    best_model = gs.best_estimator_
    test_acc = accuracy_score(y_test, best_model.predict(X_test))
    print(f"  Test accuracy:    {test_acc:.4f}")

    # Show all combinations
    print("\n  All parameter combinations (sorted by mean CV accuracy):")
    cv_results = gs.cv_results_
    rows = list(zip(cv_results["params"], cv_results["mean_test_score"]))
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"  {'Params':<40} {'CV Accuracy':>12}")
    print(f"  {'-'*40} {'-'*12}")
    for params, score in rows:
        param_str = f"n_est={params['n_estimators']}, depth={params['max_depth']}"
        print(f"  {param_str:<40} {score:>12.4f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not NUMPY_OK or not SKLEARN_OK:
        print("\nPlease install required packages:")
        print("  pip install numpy scikit-learn")
    else:
        print_section("PART 1: RandomForest vs GradientBoosting")
        results = compare_classifiers()

        print_section("PART 2: KMeans Clustering (k=3, 300 points)")
        kmeans_demo()

        print_section("PART 3: GridSearchCV on RandomForest")
        gridsearch_demo()

        print_section("SUMMARY")
        print("  1. RandomForest: parallel trees, robust baseline, easy to tune")
        print("  2. GradientBoosting: sequential trees, often higher accuracy, slower")
        print("  3. KMeans: assign points to nearest centroid, repeat until stable")
        print("  4. GridSearchCV: exhaustively search hyperparameter combinations")
        print("  5. Always use a held-out test set — GridSearchCV reports CV scores")
