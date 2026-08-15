"""
utils.py
--------
Shared helper functions for the Hb/CBC tabular branch of the
Multi-Modal Anemia Detection project.

Keeping preprocessing and evaluation logic in one place means both
the XGBoost script and the Logistic Regression script are evaluated
identically -> a fair, reproducible comparison (important for the
paper's ablation section later).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42  # fixed seed everywhere -> reproducible splits across models


def load_dataset(csv_path: str):
    """Load the CSV and do basic sanity checks."""
    df = pd.read_csv(csv_path)

    expected_cols = {
        "Sexo",
        "Hemoglobina",
        "Normal",
        "Leve",
        "Moderado"
    }

    missing = expected_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # convert gender
    df["Gender"] = df["Sexo"].map({
        "M": 1,
        "F": 0
    })

    # rename Hb
    df["Hemoglobin"] = df["Hemoglobina"]

    # binary target
    df["Result"] = (
        (df["Leve"] == 1) |
        (df["Moderado"] == 1)
    ).astype(int)

    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))

    return df


def split_features_target(df: pd.DataFrame):
    """Split into X (features) and y (target: 0 = no anemia, 1 = anemia)."""
    X = df[["Gender", "Hemoglobin"]].copy()
    y = df["Result"].copy()
    return X, y


def make_train_test_split(X, y, test_size=0.2):
    """Stratified split so class ratio is preserved in train/test."""
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def evaluate_model(model, X_test, y_test, model_name: str, results_dir: str):
    """Compute standard metrics, print them, and save to JSON for later
    comparison against the deep-learning fusion branches."""
    y_pred = model.predict(X_test)

    # Some models (e.g. plain SVM without probability=True) may not have predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n=== {model_name} results ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:>10}: {v}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Anemia", "Anemia"]))

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{model_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")

    return metrics
