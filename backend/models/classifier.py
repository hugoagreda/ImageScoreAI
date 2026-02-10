"""
Image Quality Classifier

This module trains a simple classification model that predicts
the perceived quality category of a real estate image:
- Bad
- Medium
- Good

The classifier uses technical visual metrics as input features.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# 1. Load dataset
# -----------------------------

# Load the dataset generated from visual metrics
df = pd.read_csv("backend/data/datasets/visual_quality_dataset.csv")

if "label" not in df.columns:
    inferred_label = None

    if "image_name" in df.columns:
        name_labels = (
            df["image_name"]
            .astype(str)
            .str.split("_")
            .str[0]
            .str.lower()
        )
        if name_labels.isin(["good", "medium", "bad"]).all():
            inferred_label = name_labels

    if inferred_label is None and "target_score" in df.columns:
        def score_to_label(score: float) -> str:
            if score >= 70:
                return "good"
            if score >= 40:
                return "medium"
            return "bad"

        inferred_label = df["target_score"].apply(score_to_label)

    if inferred_label is None:
        raise ValueError(
            "Dataset is missing 'label' and cannot be inferred from 'image_name' or 'target_score'."
        )

    df["label"] = inferred_label


# -----------------------------
# 2. Feature selection
# -----------------------------

# Input features: technical visual metrics
FEATURES = [
    "lighting",
    "sharpness",
    "composition",
    "color",
    "clutter"
]

X = df[FEATURES]

# Target: categorical quality label
# Expected values: "bad", "medium", "good"
y = df["label"]


# -----------------------------
# 3. Train / test split
# -----------------------------

# Split data to evaluate generalization
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # preserve class distribution
)


# -----------------------------
# 4. Model definition
# -----------------------------

# Logistic Regression is chosen because:
# - it is interpretable
# - it works well with small datasets
# - coefficients are easy to analyze
model = LogisticRegression(
    max_iter=1000
)


# -----------------------------
# 5. Training
# -----------------------------

model.fit(X_train, y_train)


# -----------------------------
# 6. Evaluation
# -----------------------------

y_pred = model.predict(X_test)

print("Classification Report")
print("----------------------")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print("----------------")
cm = confusion_matrix(y_test, y_pred, labels=["bad", "medium", "good"])
print(cm)
print("\nConfusion Matrix Labels")
print("-----------------------")
print("Order of classes: ['bad', 'medium', 'good']")
print("Rows = true label (y_test), Columns = predicted label (y_pred)")

# -----------------------------
# 7. Model interpretation
# -----------------------------

# Show feature importance per class
coefficients = pd.DataFrame(
    model.coef_,
    columns=FEATURES,
    index=model.classes_
)

print("\nModel Coefficients")
print("------------------")
print(coefficients)
