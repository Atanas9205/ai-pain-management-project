import os
import yaml
import json
import logging
from pathlib import Path

from src.data.loaders import load_raw_or_generate, ensure_dir
from src.features.preprocess import split_xy
from src.models.models import build_models, train_and_eval
from src.viz.plots import plot_metrics


def load_cfg(path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(cfg_path: str = "config/config.yaml") -> None:
    # 0) Load config and set logging
    cfg = load_cfg(cfg_path)
    logging.basicConfig(level=getattr(logging, cfg["logging"]["verbosity"]))
    out_dir = cfg["project"]["output_dir"]
    ensure_dir(out_dir)

    # 1) Load (or synthesize) data
    df = load_raw_or_generate(
        raw_dir=cfg["data"]["raw_dir"],
        synth_rows=cfg["data"]["synth_rows"],
    )

    # --- Fix for NaN in target ---
    target = cfg["data"]["target"]
    if df[target].isna().any():
        logging.warning("Target column '%s' contains NaN. Removing affected rows...", target)
        df = df.dropna(subset=[target])
    # -----------------------------

    # Check dataset structure
    logging.info("Columns in dataset: %s", df.columns.tolist())

    # 2) Split
    features = cfg["data"]["features"]
    X_train, X_test, y_train, y_test = split_xy(
        df,
        features,
        target,
        test_size=cfg["split"]["test_size"],
        stratify=cfg["split"]["stratify"],
        seed=cfg["project"]["seed"],
    )

    # 3) Build models from config
    models = build_models(cfg, numeric_cols=features)

    # 4) Train & evaluate
    report = train_and_eval(models, X_train, y_train, X_test, y_test, out_dir, cfg)

    # 5) Persist metrics and plots
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    try:
        plot_dir = os.path.join(out_dir, "plots")
        ensure_dir(plot_dir)
        plot_metrics(metrics_path, plot_dir)
    except Exception as e:
        logging.warning("Plotting skipped: %s", e)

    print("Done. Artifacts saved under:", out_dir)


if __name__ == "__main__":
    main()