import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from joblib import dump

from src.features.preprocess import build_preprocessor

def build_models(cfg: Dict[str, Any], numeric_cols):
    pre = build_preprocessor(
        numeric_cols=numeric_cols,
        imputer_strategy=cfg["preprocess"]["imputer_strategy"],
        scaler=cfg["preprocess"]["scaler"],
    )
    models = {}

    if "logistic_regression" in cfg["modeling"]["models"]:
        params = cfg["modeling"]["models"]["logistic_regression"]
        clf = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight=params.get("class_weight", None),
            n_jobs=None
        )
        models["logreg"] = Pipeline([("pre", pre), ("clf", clf)])

    if "random_forest" in cfg["modeling"]["models"]:
        params = cfg["modeling"]["models"]["random_forest"]
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            class_weight=params.get("class_weight", None),
            n_jobs=params.get("n_jobs", -1),
            random_state=cfg["project"]["seed"]
        )
        models["rf"] = Pipeline([("pre", pre), ("clf", clf)])

    return models

def evaluate_cls(y_true, proba, y_pred) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    # One-vs-rest ROC AUC (weighted)
    try:
        out["roc_auc_ovr_weighted"] = roc_auc_score(
            y_true, proba, multi_class="ovr", average="weighted"
        )
    except Exception:
        out["roc_auc_ovr_weighted"] = float("nan")
    return out

def save_confmat(cm: np.ndarray, out_path: str):
    np.savetxt(out_path, cm.astype(int), fmt="%d", delimiter=",")

def train_and_eval(models: Dict[str, Pipeline],
                   X_train, y_train, X_test, y_test,
                   out_dir: str, cfg: Dict[str, Any]):

    os.makedirs(out_dir, exist_ok=True)
    report = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Try probabilities if available
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)
        else:
            # Fallback: fake probs from predictions
            n_classes = len(np.unique(y_test))
            proba = np.zeros((len(y_pred), n_classes))
            proba[np.arange(len(y_pred)), y_pred] = 1.0

        metrics = evaluate_cls(y_test, proba, y_pred)
        report[name] = metrics

        # Save confusion matrix
        if cfg["evaluation"].get("confusion_matrix", True):
            cm = confusion_matrix(y_test, y_pred)
            save_confmat(cm, os.path.join(out_dir, f"{name}_confusion_matrix.csv"))

        # Persist model
        dump(pipe, os.path.join(out_dir, f"{name}.joblib"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report