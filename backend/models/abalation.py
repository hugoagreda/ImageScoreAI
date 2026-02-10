import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


DATASET_PATH = "backend/data/datasets/visual_quality_dataset.csv"

ALL_FEATURES = [
    "lighting",
    "sharpness",
    "composition",
    "color",
    "clutter"
]


def evaluate_features(feature_subset):
    """
    Trains and evaluates a regression model using a subset of features.
    """

    df = pd.read_csv(DATASET_PATH)

    X = df[feature_subset]
    y = df["target_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, r2


def run_ablation():
    """
    Runs ablation study by removing one feature at a time.
    """

    results = []

    # Baseline (all features)
    mae, r2 = evaluate_features(ALL_FEATURES)
    results.append(("ALL", mae, r2))

    # Remove one feature at a time
    for feature in ALL_FEATURES:
        subset = [f for f in ALL_FEATURES if f != feature]
        mae, r2 = evaluate_features(subset)
        results.append((f"NO_{feature.upper()}", mae, r2))

    # Print results
    print("Ablation Study Results")
    print("----------------------")
    for name, mae, r2 in results:
        print(f"{name:15s} | MAE: {mae:.2f} | R²: {r2:.2f}")


if __name__ == "__main__":
    run_ablation()
