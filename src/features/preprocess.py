from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def build_preprocessor(numeric_cols, imputer_strategy="median", scaler="standard"):
    imputer = SimpleImputer(strategy=imputer_strategy)
    scaler_step = StandardScaler() if scaler == "standard" else MinMaxScaler()
    numeric_pipe = Pipeline(steps=[
        ("imputer", imputer),
        ("scaler", scaler_step),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric_cols),
    ])

def split_xy(
    df: pd.DataFrame, features, target, test_size=0.2, stratify=True, seed=42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X = df[features].copy()
    y = df[target].values
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=seed
    )
    return X_train, X_test, y_train, y_test