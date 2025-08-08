import os
import logging
import numpy as np
import pandas as pd

RNG = np.random.default_rng

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def generate_synthetic(raw_dir: str, n_rows: int) -> str:
    """Generate synthetic multimodal pain dataset and save to CSV."""
    ensure_dir(raw_dir)
    rng = RNG(42)
    heart_rate = rng.normal(80, 12, size=n_rows).clip(45, 180)
    skin_cond  = rng.normal(3.0, 0.9, size=n_rows).clip(0.2, 8.0)     # microsiemens (EDA)
    cortisol   = rng.normal(12, 4, size=n_rows).clip(2, 35)           # ug/dL (simulated)

    # Hidden rule to form pain level with noise
    z = 0.03*(heart_rate-70) + 0.6*(skin_cond-2.5) + 0.08*(cortisol-10)
    probs = np.stack([
        1/(1+np.exp( z)),               # low
        1/(1+np.exp(-z)) * 0.6,         # moderate
        1/(1+np.exp(-z)) * 0.4          # high
    ], axis=1)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pain_level = np.array([rng.choice([0,1,2], p=p) for p in probs])

    df = pd.DataFrame({
        "heart_rate": heart_rate,
        "skin_conductance": skin_cond,
        "cortisol_level": cortisol,
        "pain_level": pain_level
    })
    out_path = os.path.join(raw_dir, "physio_hormonal_pain.csv")
    df.to_csv(out_path, index=False)
    return out_path

def load_raw_or_generate(raw_dir: str, synth_rows: int) -> pd.DataFrame:
    ensure_dir(raw_dir)
    csvs = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not csvs:
        logging.info("No raw CSVs found. Generating synthetic dataset...")
        csvs = [generate_synthetic(raw_dir, synth_rows)]
    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    return df