"""
train_logistic_regression.py
-----------------------------
Simple, interpretable baseline for the Hb/CBC tabular branch:
Logistic Regression.

Why Logistic Regression as the "other simple algorithm":
- It's linear and interpretable -> coefficients directly show which
  blood indices push the prediction toward "anemia", which is useful
  when you write the paper's baseline-comparison section.
- It's a natural sanity-check baseline: if XGBoost doesn't clearly
  outperform it, that itself is a reportable (and defensible) finding.
- It trains in milliseconds, so it costs nothing to include as a
  reference point in the fusion-strategy comparison table.

Usage:
    python src/train_logistic_regression.py
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

    # 2. Train/test split (same seed/ratio as the XGBoost script -> fair comparison)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.2)

    # 3. Logistic Regression needs scaled features (unlike tree models),
    # so we wrap scaling + the classifier in a single Pipeline. This keeps
    # the scaler "attached" to the model when saved, so you can't
    # accidentally apply it to unscaled data at inference time.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    # 4. Train
    print("Training Logistic Regression classifier...")
    model.fit(X_train, y_train)

    # 5. Evaluate
    evaluate_model(model, X_test, y_test, model_name="logistic_regression", results_dir=RESULTS_DIR)

    # 6. Coefficients (interpretability -> useful for the paper's discussion)
    coefs = model.named_steps["clf"].coef_[0]
    print("\nStandardized coefficients (positive = pushes toward 'Anemia'):")
    for feat, coef in sorted(zip(X.columns, coefs), key=lambda t: -abs(t[1])):
        print(f"  {feat:>12}: {coef:+.4f}")

    # 7. Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "logistic_regression_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
