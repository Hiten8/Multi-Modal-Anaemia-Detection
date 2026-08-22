"""
Dataset:
    Asare dataset

Total patients:
    909

Split:
    80% development/training = 727
    20% final test           = 182

Features:
    Gender
    Age
    Hemoglobin

Target:
    0 = Normal
    1 = Leve
    2 = Moderado

The model learns the relationship between:
    Hb + Age + Sex

and the clinically assigned:
    Normal / Leve / Moderado

After predicting the severity class, binary anemia status is
derived from the predicted severity:
    Normal   -> No Anemia
    Leve     -> Anemia
    Moderado -> Anemia
"""

import os
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "metadata.csv",
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "models",
)

RANDOM_STATE = 42

FEATURES = [
    "Gender",
    "Edad",
    "Hemoglobina",
]

CLASS_NAMES = [
    "Normal",
    "Leve",
    "Moderado",
]


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "ID",
        "Sexo",
        "Edad",
        "Hemoglobina",
        "Normal",
        "Leve",
        "Moderado",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["Gender"] = (
        df["Sexo"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({
            "M": 1,
            "F": 0,
        })
    )

    df["Edad"] = pd.to_numeric(
        df["Edad"],
        errors="coerce",
    )

    df["Hemoglobina"] = pd.to_numeric(
        df["Hemoglobina"],
        errors="coerce",
    )

    df["Severity"] = np.select(
        [
            df["Normal"] == 1,
            df["Leve"] == 1,
            df["Moderado"] == 1,
        ],
        [
            0,
            1,
            2,
        ],
        default=-1,
    )

    if (df["Severity"] == -1).any():
        raise ValueError(
            "Some patients do not have a valid "
            "Normal/Leve/Moderado label."
        )

    class_sum = (
        df["Normal"]
        + df["Leve"]
        + df["Moderado"]
    )

    if (class_sum != 1).any():
        raise ValueError(
            "Some patients do not have exactly "
            "one severity label."
        )

    return df


def calculate_reliability(df):
    """
    Reliability is based on availability of the three
    clinical inputs used by this branch:
        Gender
        Age
        Hemoglobin

    1.0 = all three available
    0.667 = two available
    0.333 = one available
    0.0 = none available
    """

    availability = pd.DataFrame(
        index=df.index
    )

    availability["Gender"] = (
        df["Gender"].notna()
        .astype(float)
    )

    availability["Edad"] = (
        df["Edad"].notna()
        .astype(float)
    )

    availability["Hemoglobina"] = (
        df["Hemoglobina"].notna()
        .astype(float)
    )

    return availability.mean(axis=1)


def create_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def main():

    print("=" * 75)
    print(
        "FINAL CLINICAL / HEMOGLOBIN "
        "XGBOOST BRANCH"
    )
    print("=" * 75)

    # 1. Load dataset
    df = load_dataset()
    print(
        f"\nLoaded patients: {len(df)}"
    )

    # 2. Class distribution
    print(
        "\nClass distribution:"
    )

    distribution = (
        df["Severity"]
        .value_counts()
        .sort_index()
    )

    for class_id, count in distribution.items():
        percentage = (
            count / len(df) * 100
        )

        print(
            f"{CLASS_NAMES[class_id]:10}: "
            f"{count:3d} "
            f"({percentage:.2f}%)"
        )

    # 3. Calculate data reliability
    df["Data_Reliability"] = (
        calculate_reliability(df)
    )

    print(
        "\nClinical data reliability:"
    )

    print(
        df["Data_Reliability"]
        .describe()
    )

    # 4. Features and target
    X = df[
        FEATURES
    ].copy()

    y = df[
        "Severity"
    ].copy()

    # 5. Create train test 80/20 split
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(

        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"\nTraining patients: "
        f"{len(X_train)}"
    )

    print(
        f"Final test patients: "
        f"{len(X_test)}"
    )

    # 6. Create and train model
    print(
        "\nTraining XGBoost "
    )

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Model training complete."
    )

    # 7. Predict the test patients
    print(
        "\nPredicting the "
        "test patients..."
    )

    y_pred = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    # 8. Class probabilities
    # Column order:
    # 0 = Normal
    # 1 = Leve
    # 2 = Moderado

    p_normal = probabilities[:, 0]
    p_leve = probabilities[:, 1]
    p_moderado = probabilities[:, 2]

    # 9. Binary anemia prediction
    # Predicted severity determines binary status:
    # Normal   -> No Anemia
    # Leve     -> Anemia
    # Moderado -> Anemia

    predicted_anemia = (
        y_pred != 0
    ).astype(int)

    actual_anemia = (
        y_test.to_numpy() != 0
    ).astype(int)

    # 10. MULTICLASS TEST EVALUATION
    multiclass_accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    multiclass_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    multiclass_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    multiclass_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    multiclass_weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    multiclass_cm = confusion_matrix(
        y_test,
        y_pred,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST RESULTS - 3 CLASS"
    )

    print(
        "=" * 75
    )

    print(
        "Task: Normal / Leve / Moderado"
    )

    print(
        f"\naccuracy        : "
        f"{multiclass_accuracy:.6f}"
    )

    print(
        f"macro_precision : "
        f"{multiclass_precision:.6f}"
    )

    print(
        f"macro_recall    : "
        f"{multiclass_recall:.6f}"
    )

    print(
        f"macro_f1        : "
        f"{multiclass_f1:.6f}"
    )

    print(
        f"weighted_f1     : "
        f"{multiclass_weighted_f1:.6f}"
    )

    print(
        "\n3-class confusion matrix:"
    )

    print(
        "Rows = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print(
        "Order = Normal, Leve, Moderado"
    )

    print(
        multiclass_cm
    )

    print(
        "\n3-class classification report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # 11. BINARY TEST EVALUATION
    binary_accuracy = accuracy_score(
        actual_anemia,
        predicted_anemia,
    )

    binary_precision = precision_score(
        actual_anemia,
        predicted_anemia,
        zero_division=0,
    )

    binary_recall = recall_score(
        actual_anemia,
        predicted_anemia,
        zero_division=0,
    )

    binary_f1 = f1_score(
        actual_anemia,
        predicted_anemia,
        zero_division=0,
    )

    binary_cm = confusion_matrix(
        actual_anemia,
        predicted_anemia,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST RESULTS - BINARY ANEMIA"
    )

    print(
        "=" * 75
    )

    print(
        "Mapping:"
    )

    print(
        "Normal   -> No Anemia"
    )

    print(
        "Leve     -> Anemia"
    )

    print(
        "Moderado -> Anemia"
    )

    print(
        f"\naccuracy  : "
        f"{binary_accuracy:.6f}"
    )

    print(
        f"precision : "
        f"{binary_precision:.6f}"
    )

    print(
        f"recall    : "
        f"{binary_recall:.6f}"
    )

    print(
        f"f1_score  : "
        f"{binary_f1:.6f}"
    )

    print(
        "\nBinary confusion matrix:"
    )

    print(
        "[[TN, FP], [FN, TP]]"
    )

    print(
        binary_cm
    )

    print(
        "\nBinary classification report:"
    )

    print(
        classification_report(
            actual_anemia,
            predicted_anemia,
            target_names=[
                "No Anemia",
                "Anemia",
            ],
            zero_division=0,
        )
    )

    # 12. Create patient output
    test_output = pd.DataFrame({
        "Patient_ID":
            df.loc[
                X_test.index,
                "ID",
            ].values,

        "P_Normal":
            p_normal,

        "P_Leve":
            p_leve,

        "P_Moderado":
            p_moderado,

        "Predicted_Severity":
            y_pred,

        "Predicted_Anemia":
            predicted_anemia,

        "Data_Reliability":
            df.loc[
                X_test.index,
                "Data_Reliability",
            ].values,

        "Actual_Severity":
            y_test.to_numpy(),

        "Actual_Anemia":
            actual_anemia,
    })

    # Convert numeric class labels to readable labels
    test_output["Predicted_Severity_Label"] = (
        test_output[
            "Predicted_Severity"
        ].map({
            0: "Normal",
            1: "Leve",
            2: "Moderado",
        })
    )

    test_output["Actual_Severity_Label"] = (
        test_output[
            "Actual_Severity"
        ].map({
            0: "Normal",
            1: "Leve",
            2: "Moderado",
        })
    )

    test_output["Predicted_Anemia_Label"] = (
        test_output[
            "Predicted_Anemia"
        ].map({
            0: "No Anemia",
            1: "Anemia",
        })
    )

    test_output["Actual_Anemia_Label"] = (
        test_output[
            "Actual_Anemia"
        ].map({
            0: "No Anemia",
            1: "Anemia",
        })
    )

    # 13. Save patient predictions
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    predictions_path = os.path.join(
        RESULTS_DIR,
        "final_clinical_xgboost_test_predictions.csv",
    )

    test_output.to_csv(
        predictions_path,
        index=False,
    )

    print(
        "\nSaved patient predictions:"
    )

    print(
        predictions_path
    )

    # 14. Display sample predictions
    print(
        "\nSample test outputs:"
    )

    display_columns = [
        "Patient_ID",
        "P_Normal",
        "P_Leve",
        "P_Moderado",
        "Predicted_Severity_Label",
        "Predicted_Anemia_Label",
        "Data_Reliability",
        "Actual_Severity_Label",
        "Actual_Anemia_Label",
    ]

    print(
        test_output[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    # 15. Feature importance
    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "XGBOOST FEATURE IMPORTANCE"
    )

    print(
        "=" * 75
    )

    feature_importance = sorted(
        zip(
            FEATURES,
            classifier.feature_importances_,
        ),
        key=lambda x: -x[1],
    )

    for feature, importance in feature_importance:
        print(
            f"{feature:20}: "
            f"{importance:.6f}"
        )

    # 16. Save model
    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    model_path = os.path.join(
        MODEL_DIR,
        "final_clinical_xgboost_multiclass.joblib",
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        "\nSaved final model:"
    )

    print(
        model_path
    )

    # 17. Save metrics
    metrics = {
        "model":
            "final_clinical_xgboost_multiclass",

        "total_patients":
            len(df),

        "training_patients":
            len(X_train),

        "final_test_patients":
            len(X_test),

        "features":
            FEATURES,

        "classes":
            {
                "0": "Normal",
                "1": "Leve",
                "2": "Moderado",
            },

        "multiclass_test": {
            "accuracy":
                float(multiclass_accuracy),

            "macro_precision":
                float(multiclass_precision),

            "macro_recall":
                float(multiclass_recall),

            "macro_f1":
                float(multiclass_f1),

            "weighted_f1":
                float(multiclass_weighted_f1),

            "confusion_matrix":
                multiclass_cm.tolist(),
        },

        "binary_anemia_test": {
            "accuracy":
                float(binary_accuracy),

            "precision":
                float(binary_precision),

            "recall":
                float(binary_recall),

            "f1_score":
                float(binary_f1),

            "confusion_matrix":
                binary_cm.tolist(),
        },
    }

    metrics_path = os.path.join(
        RESULTS_DIR,
        "final_clinical_xgboost_test_metrics.json",
    )

    with open(
        metrics_path,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        "\nSaved metrics:"
    )

    print(
        metrics_path
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CLINICAL BRANCH COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Development patients : {len(X_train)}"
    )

    print(
        f"Test patients  : {len(X_test)}"
    )

    print(
        "\nThe test patients "
        "were evaluated using a model trained "
        "only on the development patients."
    )


if __name__ == "__main__":
    main()