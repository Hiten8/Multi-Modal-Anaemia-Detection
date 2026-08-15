"""
train_random_forest.py
----------------------

Baseline Machine Learning model for the Nail Image Branch.

Pipeline
--------
Dataset
    ↓
Feature Extraction
    ↓
Train/Test Split
    ↓
Random Forest
    ↓
Evaluation
    ↓
Save Model
"""

import os

from sklearn.ensemble import RandomForestClassifier

from dataset import load_dataset
from feature_extraction import feature_names
from utils import (
    make_train_test_split,
    evaluate_model,
    save_model,
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "nail_dataset"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "results"
)

CLASS_NAMES = [
    "Normal",
    "Leve",
    "Moderada"
]

def main():

    print()
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    X, y, filenames = load_dataset(
        DATASET_PATH
    )

    print()
    print(f"Dataset Shape : {X.shape}")
    print()
    print("=" * 60)
    print("Train/Test Split")
    print("=" * 60)

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=0.20
    )

    print()
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")
    print()
    print("=" * 60)
    print("Training Random Forest")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print()
    print("Training Complete")
    print()
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)
    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=CLASS_NAMES,
        results_dir=RESULTS_DIR,
        model_name="random_forest"
    )

    print()
    print("=" * 60)
    print("Top Feature Importances")
    print("=" * 60)

    importance = model.feature_importances_

    names = feature_names()

    ranked = sorted(
        zip(names, importance),
        key=lambda x: x[1],
        reverse=True
    )

    for name, score in ranked[:20]:
        print(
            f"{name:25s} {score:.4f}"
        )

    print()
    print("=" * 60)
    print("Saving Model")
    print("=" * 60)

    save_model(
        model,
        MODEL_DIR,
        "random_forest_model.joblib"
    )

    print()
    print("Done.")
    print()

if __name__ == "__main__":
    main()