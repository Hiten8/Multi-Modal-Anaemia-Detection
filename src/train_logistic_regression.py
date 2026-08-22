"""
Clinical/Tabular branch using Logistic Regression.

Input:
    Sexo
    Edad
    Frecuencia Cardiaca
    Hemoglobina
    Oxígeno
    Peso
    Talla

Output:
    Anemia probability
    Predicted class
    Reliability score
"""

import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    load_dataset,
    prepare_features,
    make_train_test_split,
    evaluate_model,
    NUMERIC_CLINICAL_COLUMNS,
    calculate_reliability,
)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "metadata.csv",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "models",
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
)

def main():

    # 1. Load dataset
    df = load_dataset(DATA_PATH)

    # 2. Calculate reliability BEFORE imputation
    df["Reliability"] = calculate_reliability(df)

    print("\nReliability statistics:")
    print(df["Reliability"].describe())

    # 3. Prepare features and target
    X, y = prepare_features(df)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=0.20,
    )

    # 5. Logistic Regression pipeline
    numeric_features = X.columns.tolist()

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]),
                numeric_features,
            )
        ]
    )

    model = Pipeline([
        (
            "preprocessing",
            preprocessing,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    # 6. Train
    print("\nTraining Logistic Regression...")

    model.fit(
        X_train,
        y_train,
    )

    # 7. Evaluate
    metrics, y_prob = evaluate_model(
        model,
        X_test,
        y_test,
        model_name="logistic_regression_clinical",
        results_dir=RESULTS_DIR,
    )

    # 8. Produce branch outputs for test patients
    test_reliability = df.loc[
        X_test.index,
        "Reliability",
    ]

    branch_output = X_test.copy()

    branch_output["Actual"] = y_test
    branch_output["Anemia_Probability"] = y_prob
    branch_output["Predicted_Class"] = (
        y_prob >= 0.5
    ).astype(int)

    branch_output["Reliability"] = (
        test_reliability
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "logistic_regression_clinical_predictions.csv",
    )

    branch_output.to_csv(
        output_path,
        index=True,
    )

    print(
        f"\nSaved branch predictions to: "
        f"{output_path}"
    )

    # 9. Save model
    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    model_path = os.path.join(
        MODEL_DIR,
        "logistic_regression_clinical.joblib",
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Saved model to: {model_path}"
    )

    # 10. Display sample branch outputs
    print("\nSample Hb/Clinical branch outputs:")

    print(
        branch_output[
            [
                "Actual",
                "Anemia_Probability",
                "Predicted_Class",
                "Reliability",
            ]
        ].head(10)
    )

if __name__ == "__main__":
    main()