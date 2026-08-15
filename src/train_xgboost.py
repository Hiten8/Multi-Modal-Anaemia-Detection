"""
train_xgboost.py
-----------------
Baseline tabular branch (Hb + CBC indices) for the Multi-Modal Anemia
Detection project, using XGBoost.

This is the tabular counterpart that will later feed into the dynamic
attention-based fusion gate alongside the visual (fingertip / palm /
nail) CNN branches. For now, it's trained and evaluated standalone.

Usage:
    python src/train_xgboost.py
"""

import os
import joblib
from xgboost import XGBClassifier

from utils import load_dataset, split_features_target, make_train_test_split, evaluate_model

# ---- Paths -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "metadata.csv")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")


def main():
    # 1. Load data
    df = load_dataset(DATA_PATH)
    X, y = split_features_target(df)

    # 2. Train/test split (stratified, same seed as other models for a fair comparison)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.2)

    # 3. Define XGBoost classifier
    # Kept intentionally simple for the first baseline pass. Hyperparameter
    # tuning (grid/random search, early stopping on a validation fold) is a
    # natural next step once this baseline number is in hand.
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )

    # 4. Train
    print("Training XGBoost classifier...")
    model.fit(X_train, y_train)

    # 5. Evaluate
    evaluate_model(model, X_test, y_test, model_name="xgboost", results_dir=RESULTS_DIR)

    # 6. Feature importance (useful for the paper's discussion section)
    importances = model.feature_importances_
    print("\nFeature importances:")
    for feat, imp in sorted(zip(X.columns, importances), key=lambda t: -t[1]):
        print(f"  {feat:>12}: {imp:.4f}")

    # 7. Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
