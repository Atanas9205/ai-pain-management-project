# tests/test_smoke.py
from pathlib import Path
import pandas as pd

from src.data.loaders import load_raw_or_generate


# ---- Helpers ----
NUMERIC_GROUPS = [
    {"cortisol_ug_dL", "cortisol_level", "cortisol"},         # cortisol aliases
    {"eda_uS", "eda_us", "eda", "skin_conductance"},          # EDA aliases
    {"skin_temp_C", "skin_temp", "temperature"},              # temperature aliases
    {"heart_rate_bpm", "heart_rate", "hr", "ecg_hr"},         # heart-rate aliases
]

def _cols(df) -> set:
    return set(map(str, df.columns))

def _has_any(df_cols: set, group: set) -> bool:
    # true if at least one alias from the group is present
    return len(df_cols.intersection(group)) >= 1


# ---- Test 1: schema / shape (relaxed) ----
def test_schema_and_min_rows(tmp_path: Path):
    """
    Smoke test for data loading / generation:

    - DataFrame is returned with > 0 rows
    - 'pain_level' exists (target)
    - At least 3 out of the 4 numeric feature groups are present
      (cortisol / EDA / temperature / heart-rate).
    - 'timestamp' and 'subject_id' are treated as optional (no assert).
    """
    raw_dir = tmp_path / "raw"
    df = load_raw_or_generate(str(raw_dir), synth_rows=120)

    # Shape ok
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0, "Empty DataFrame returned by load_raw_or_generate()."

    cols = _cols(df)

    # Target must exist
    assert "pain_level" in cols, "Missing required target column: 'pain_level'."

    # Count how many numeric groups we cover
    covered = sum(1 for g in NUMERIC_GROUPS if _has_any(cols, g))
    assert (
        covered >= 3
    ), f"Expected at least 3 numeric feature groups, got {covered}. Columns: {sorted(cols)}"

    # Optional identity columns (no failure if absent)
    # Kept as a soft note for human readers in notebook output
    if not {"timestamp", "time"}.intersection(cols):
        print("[note] No timestamp-like column detected (this is OK for the smoke test).")
    if not {"subject_id", "id", "subject"}.intersection(cols):
        print("[note] No subject-id-like column detected (this is OK for the smoke test).")


# ---- Test 2: target semantics + numeric dtypes ----
def test_target_values_and_numeric_types(tmp_path: Path):
    """
    - 'pain_level' is present and uses either string labels {low, moderate, high}
      or integer codes {0, 1, 2}.
    - The present numeric features among the 4 groups are numeric dtype.
    """
    raw_dir = tmp_path / "raw"
    df = load_raw_or_generate(str(raw_dir), synth_rows=80)

    cols = _cols(df)

    # Target values
    assert "pain_level" in cols, "Missing 'pain_level'."
    unique_vals = set(pd.Series(df["pain_level"]).dropna().unique().tolist())
    str_ok = unique_vals.issubset({"low", "moderate", "high"})
    int_ok = unique_vals.issubset({0, 1, 2})
    assert str_ok or int_ok, f"Unexpected values in pain_level: {unique_vals}"

    # Check numeric dtype only for the features that actually exist
    possible_numeric = {
        "cortisol_ug_dL", "cortisol_level", "cortisol",
        "eda_uS", "eda_us", "eda", "skin_conductance",
        "skin_temp_C", "skin_temp", "temperature",
        "heart_rate_bpm", "heart_rate", "hr", "ecg_hr",
    }
    present_numeric = sorted(cols.intersection(possible_numeric))
    assert present_numeric, "No numeric feature columns found."

    for col in present_numeric:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric."