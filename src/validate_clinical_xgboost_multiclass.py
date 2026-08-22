"""
5-fold stratified cross-validation for the
Clinical/Hemoglobin 3-class XGBoost branch.

Inputs:
    Gender
    Age
    Hemoglobin

Target:
    0 = Normal
    1 = Leve
    2 = Moderado

The model is NOT given any clinical thresholds.

It learns the relationship between:
    Hb + Age + Sex
and the clinically assigned:
    Normal / Leve / Moderado

Two evaluation tasks are reported:
1. Multiclass severity:
       Normal / Leve / Moderado

2. Binary anemia status:
       Normal   -> No Anemia
       Leve     -> Anemia
       Moderado -> Anemia
"""

import os
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)

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

    # Three-class target
    # Normal   -> 0
    # Leve     -> 1
    # Moderado -> 2

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

    # Validate severity labels
    if (df["Severity"] == -1).any():

        raise ValueError(
            "Some patients do not have a valid "
            "Normal/Leve/Moderado label."
        )

    # Each patient must have exactly one class
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

    print("=" * 70)
    print(
        "5-FOLD STRATIFIED CV"
    )
    print(
        "CLINICAL / HEMOGLOBIN "
        "3-CLASS XGBOOST"
    )
    print("=" * 70)

    # 1. Load dataset
    df = load_dataset()

    X = df[
        FEATURES
    ].copy()

    y = df[
        "Severity"
    ].copy()

    print(
        f"\nTotal patients: {len(df)}"
    )

    # 2. Class distribution
    print("\nClass distribution:")

    distribution = (
        y.value_counts()
        .sort_index()
    )

    for class_id, count in distribution.items():
        percentage = (
            count / len(y) * 100
        )

        print(
            f"{CLASS_NAMES[class_id]:10}: "
            f"{count:3d} "
            f"({percentage:.2f}%)"
        )

    # 3. Hold out final test set
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
        f"\nTraining patients: {len(X_train)}"
    )
    print(
        f"Final test patients: {len(X_test)}"
    )

    # 4. 5-fold stratified CV
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 5. Cross-validation
    # We use cross_val_predict so every training patient gets
    # exactly one out-of-fold prediction.
    print(
        "\nStarting 5-fold CV..."
    )

    model = create_model()

    y_pred_cv = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict",
        n_jobs=-1,
    )

    # 6. Multiclass Evaluation
    multiclass_accuracy = accuracy_score(
        y_train,
        y_pred_cv,
    )

    multiclass_precision = precision_score(
        y_train,
        y_pred_cv,
        average="macro",
        zero_division=0,
    )

    multiclass_recall = recall_score(
        y_train,
        y_pred_cv,
        average="macro",
        zero_division=0,
    )

    multiclass_f1 = f1_score(
        y_train,
        y_pred_cv,
        average="macro",
        zero_division=0,
    )

    multiclass_weighted_f1 = f1_score(
        y_train,
        y_pred_cv,
        average="weighted",
        zero_division=0,
    )

    multiclass_cm = confusion_matrix(
        y_train,
        y_pred_cv,
    )

    print("\n"+ "=" * 70)
    print("MULTICLASS CV RESULTS")
    print("=" * 70)

    print(
        f"accuracy        : "
        f"{multiclass_accuracy:.4f}"
    )

    print(
        f"macro_precision : "
        f"{multiclass_precision:.4f}"
    )

    print(
        f"macro_recall    : "
        f"{multiclass_recall:.4f}"
    )

    print(
        f"macro_f1        : "
        f"{multiclass_f1:.4f}"
    )

    print(
        f"weighted_f1     : "
        f"{multiclass_weighted_f1:.4f}"
    )

    print("\n 3-class confusion matrix:")
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

    print("\n 3-class classification report:")
    print(
        classification_report(
            y_train,
            y_pred_cv,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # 7. Binary Anemia Task
    # We simply map the predicted severity:
    # Normal   -> 0 = No Anemia
    # Leve     -> 1 = Anemia
    # Moderado -> 1 = Anemia

    binary_actual = (
        y_train != 0
    ).astype(int)

    binary_predicted = (
        y_pred_cv != 0
    ).astype(int)

    binary_accuracy = accuracy_score(
        binary_actual,
        binary_predicted,
    )

    binary_precision = precision_score(
        binary_actual,
        binary_predicted,
        zero_division=0,
    )

    binary_recall = recall_score(
        binary_actual,
        binary_predicted,
        zero_division=0,
    )

    binary_f1 = f1_score(
        binary_actual,
        binary_predicted,
        zero_division=0,
    )

    binary_cm = confusion_matrix(
        binary_actual,
        binary_predicted,
    )

    print("\n"+ "=" * 70)
    print(
        "BINARY ANEMIA CV RESULTS"
    )
    print("=" * 70)

    print("Mapping:")
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
        f"{binary_accuracy:.4f}"
    )

    print(
        f"precision : "
        f"{binary_precision:.4f}"
    )

    print(
        f"recall    : "
        f"{binary_recall:.4f}"
    )

    print(
        f"f1_score  : "
        f"{binary_f1:.4f}"
    )

    print("\nBinary confusion matrix:")
    print(
        "[[TN, FP], [FN, TP]]"
    )
    print(
        binary_cm
    )

    print("\nBinary classification report:")
    print(
        classification_report(
            binary_actual,
            binary_predicted,
            target_names=[
                "No Anemia",
                "Anemia",
            ],
            zero_division=0,
        )
    )

    # 8. Save CV results
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    results = {
        "model":
            "xgboost_multiclass",

        "features":
            FEATURES,

        "classes":
            {
                "0": "Normal",
                "1": "Leve",
                "2": "Moderado",
            },

        "training_patients":
            len(X_train),

        "held_out_test_patients":
            len(X_test),

        "multiclass": {
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

        "binary_anemia": {
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

    output_path = os.path.join(
        RESULTS_DIR,
        "clinical_xgboost_multiclass_5fold_cv.json",
    )

    with open(
        output_path,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"\nSaved CV results to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()