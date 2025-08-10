from typing import Dict, List, Any
import os
import json
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def _make_scaler(name: str):
    """Return a sklearn scaler from config name."""
    name = (name or "standard").lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaler: {name}")


def build_models(cfg: dict, numeric_cols: List[str]) -> Dict[str, Pipeline]:
    """
    Build model pipelines with preprocessing for numeric and categorical columns.
    """
    pre_cfg = cfg.get("preprocess", {})

    # Numeric imputer with safe handling for all-NaN columns
    num_strategy = pre_cfg.get("numeric_imputer_strategy", "mean")
    num_fill = pre_cfg.get("numeric_imputer_fill_value", 0)
    num_imputer_kwargs = {"strategy": num_strategy, "keep_empty_features": True}
    if num_strategy == "constant":
        num_imputer_kwargs["fill_value"] = num_fill

    cat_strategy = pre_cfg.get("categorical_imputer_strategy", "most_frequent")
    use_ohe = bool(pre_cfg.get("encode_categorical", True))
    scaler = _make_scaler(pre_cfg.get("scaler", "standard"))
    cat_cols = list(pre_cfg.get("categorical_cols", []))

    numeric_cols = [c for c in numeric_cols if c not in cat_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(**num_imputer_kwargs)),
        ("scaler", scaler),
    ])

    if use_ohe and cat_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=cat_strategy, keep_empty_features=True)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    else:
        categorical_pipe = "drop"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )

    models_cfg: dict = cfg["modeling"]["models"]
    models: Dict[str, Pipeline] = {}

    if "logistic_regression" in models_cfg:
        params = models_cfg["logistic_regression"]
        clf = LogisticRegression(max_iter=params.get("max_iter", 1000), C=params.get("C", 1.0))
        models["logistic_regression"] = Pipeline([("pre", preprocessor), ("clf", clf)])

    if "random_forest" in models_cfg:
        params = models_cfg["random_forest"]
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=params.get("random_state", 42),
        )
        models["random_forest"] = Pipeline([("pre", preprocessor), ("clf", clf)])

    if "svm_rbf" in models_cfg:
        params = models_cfg["svm_rbf"]
        clf = SVC(kernel=params.get("kernel", "rbf"), probability=params.get("probability", True))
        models["svm_rbf"] = Pipeline([("pre", preprocessor), ("clf", clf)])

    if "mlp" in models_cfg:
        params = models_cfg["mlp"]
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (64, 32))),
            max_iter=params.get("max_iter", 500),
            random_state=cfg["project"].get("seed", 42),
        )
        models["mlp"] = Pipeline([("pre", preprocessor), ("clf", clf)])

    if "xgboost" in models_cfg and _HAS_XGB:
        params = models_cfg["xgboost"]
        clf = XGBClassifier(
            use_label_encoder=params.get("use_label_encoder", False),
            eval_metric=params.get("eval_metric", "logloss"),
            max_depth=params.get("max_depth", 6),
            random_state=cfg["project"].get("seed", 42),
        )
        models["xgboost"] = Pipeline([("pre", preprocessor), ("clf", clf)])

    return models


def _predict_proba_or_scores(model, X):
    """
    Return class probabilities if available; otherwise convert decision_function to probabilities.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exps = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exps / exps.sum(axis=1, keepdims=True)
    preds = model.predict(X)
    classes = getattr(model, "classes_", np.unique(preds))
    proba = np.zeros((len(preds), len(classes)))
    for i, c in enumerate(classes):
        proba[:, i] = (preds == c).astype(float)
    return proba


def train_and_eval(models: Dict[str, Any],
                   X_train, y_train, X_test, y_test,
                   out_dir: str,
                   cfg: dict) -> Dict[str, Any]:
    """
    Train each model pipeline and compute metrics.
    """
    os.makedirs(out_dir, exist_ok=True)

    report: Dict[str, Any] = {}
    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes) if len(classes) > 2 else None

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = _predict_proba_or_scores(pipe, X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        if y_test_bin is not None and y_proba.shape[1] == len(classes):
            model_classes = getattr(pipe.named_steps["clf"], "classes_", classes)
            order = [list(model_classes).index(c) for c in classes]
            y_proba_aligned = y_proba[:, order]
            auc_micro = roc_auc_score(y_test_bin, y_proba_aligned, average="micro", multi_class="ovr")
            ap_micro = average_precision_score(y_test_bin, y_proba_aligned, average="micro")
        else:
            try:
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    auc_micro = roc_auc_score(y_test, y_proba[:, 1])
                    ap_micro = average_precision_score(y_test, y_proba[:, 1])
                else:
                    auc_micro = float("nan")
                    ap_micro = float("nan")
            except Exception:
                auc_micro = float("nan")
                ap_micro = float("nan")

        report[name] = {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
            "roc_auc_micro": auc_micro,
            "average_precision_micro": ap_micro,
        }

    with open(os.path.join(out_dir, "metrics_by_model.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report