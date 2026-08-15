"""
utils.py
--------

Shared utility functions for the Nail Image Branch.

Responsibilities
----------------
1. Train/Test split
2. Model evaluation
3. Confusion matrix plotting
4. Feature importance plotting
5. Save metrics
6. Save / Load models
"""

import json
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

RANDOM_STATE = 42

def make_train_test_split(
        X,
        y,
        test_size=0.2):
    """
    Stratified train/test split.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

def plot_confusion_matrix(
        cm,
        classes,
        save_path):
    """
    Save confusion matrix figure.
    """

    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(classes))

    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(
        model,
        feature_names,
        save_path,
        top_n=20):
    """
    Plot top feature importances.
    """

    importance = model.feature_importances_

    indices = np.argsort(
        importance
    )[::-1][:top_n]

    plt.figure(figsize=(10,6))
    plt.bar(
        range(top_n),
        importance[indices]
    )
    plt.xticks(
        range(top_n),
        [feature_names[i] for i in indices],
        rotation=90
    )
    plt.ylabel("Importance")
    plt.title(
        f"Top {top_n} Features"
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(
        metrics,
        results_dir,
        filename):
    """
    Save metrics JSON.
    """

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    path = os.path.join(
        results_dir,
        filename
    )

    with open(path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )
    print(f"Saved metrics to {path}")

def evaluate_model(
        model,
        X_test,
        y_test,
        class_names,
        results_dir,
        model_name):
    """
    Evaluate trained classifier.
    """

    y_pred = model.predict(X_test)

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        y_test,
        y_pred
    )

    metrics["precision"] = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )

    metrics["recall"] = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    metrics["f1_score"] = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(
            X_test
        )

        metrics["roc_auc"] = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr"
        )

    else:

        metrics["roc_auc"] = None

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    metrics["confusion_matrix"] = cm.tolist()

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    metrics["classification_report"] = report

    print()
    print("="*50)
    print(model_name)
    print("="*50)
    print()
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    print()

    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names
    ))

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(
            results_dir,
            f"{model_name}_confusion_matrix.png"
        )
    )

    save_metrics(
        metrics,
        results_dir,
        f"{model_name}_metrics.json"
    )

    if hasattr(model, "feature_importances_"):
        from feature_extraction import feature_names    

        plot_feature_importance(
            model,
            feature_names(),
            os.path.join(
                results_dir,
                f"{model_name}_feature_importance.png"
            )
        )

    return metrics

def save_model(
        model,
        model_dir,
        filename):
    """
    Save trained model.
    """

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    path = os.path.join(
        model_dir,
        filename
    )

    joblib.dump(
        model,
        path
    )

    print(f"Saved model to {path}")

def load_model(model_path):
    """
    Load saved model.
    """

    return joblib.load(model_path)