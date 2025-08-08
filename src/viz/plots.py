import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(metrics_json: str, out_dir: str):
    with open(metrics_json, "r") as f:
        metrics = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    names = list(metrics.keys())
    for key in ["accuracy", "f1_macro", "roc_auc_ovr_weighted"]:
        vals = [metrics[m].get(key, np.nan) for m in names]
        plt.figure()
        plt.bar(names, vals)
        plt.title(key)
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{key}.png"), dpi=160)
        plt.close()