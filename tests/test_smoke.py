import os
from src.data.loaders import load_raw_or_generate

def test_can_generate_or_load(tmp_path):
    raw_dir = tmp_path / "raw"
    df = load_raw_or_generate(str(raw_dir), synth_rows=100)
    assert len(df) == 100
    assert {"heart_rate", "skin_conductance", "cortisol_level", "pain_level"} <= set(df.columns)