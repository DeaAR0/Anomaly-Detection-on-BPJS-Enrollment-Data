"""
Feature engineering for BPJS enrollment data.
Shared between train.py and score.py to guarantee consistency.
"""

import pandas as pd
import numpy as np
from .config import COLUMN_MAP, REFERENCE_DATE, CATEGORICAL_COLS, NUMERICAL_FEATURES


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename BPJS coded columns to readable names."""
    return df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features from raw enrollment data.
    Returns df with new columns added (does not drop originals).
    """
    df = df.copy()

    # --- Age ---
    df["tanggal_lahir"] = pd.to_datetime(df["tanggal_lahir"], errors="coerce")
    ref = pd.Timestamp(REFERENCE_DATE)
    df["umur"] = (ref - df["tanggal_lahir"]).dt.days // 365

    # --- Active status ---
    status_col = "status_peserta" if "status_peserta" in df.columns else "status"
    df["is_active"] = (df[status_col] == "AKTIF").astype(int)

    # --- Family-level features ---
    grp = df.groupby("id_keluarga")
    df["jml_keluarga"] = grp["id_peserta"].transform("count")
    df["jml_keluarga_aktif"] = grp["is_active"].transform("sum")
    df["rasio_aktif"] = df["jml_keluarga_aktif"] / df["jml_keluarga"]

    df["is_kepala"] = df["peran"].isin(["PESERTA", "SUAMI", "ISTRI"]).astype(int)
    df["kepala_exists"] = grp["is_kepala"].transform("max")
    df["jml_peran_unik"] = grp["peran"].transform("nunique")

    # --- Cross-consistency flags ---
    df["anak_tapi_dewasa"] = ((df["peran"] == "ANAK") & (df["umur"] > 25)).astype(int)
    df["kepala_tapi_muda"] = (
        (df["peran"].isin(["PESERTA", "SUAMI", "ISTRI"])) & (df["umur"] < 17)
    ).astype(int)
    df["kawin_tapi_muda"] = (
        (df["status_kawin"] == "KAWIN") & (df["umur"] < 16)
    ).astype(int)

    # --- Kapitasi ---
    if "kapitasi" in df.columns:
        df["kapitasi"] = pd.to_numeric(df["kapitasi"], errors="coerce").fillna(0.0)
    else:
        df["kapitasi"] = 0.0

    return df


def encode_and_select(df: pd.DataFrame, fit_columns=None):
    """
    One-hot encode categoricals and select model features.

    Args:
        df: DataFrame after engineer_features()
        fit_columns: list of columns from training. If provided (scoring mode),
                     the output is aligned to these columns. If None (training mode),
                     columns are discovered from the data.

    Returns:
        X: np.ndarray of shape (n_samples, n_features)
        feature_names: list of column names (save this during training)
    """
    cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=cats, drop_first=True)

    # Dummy columns: only from categorical encoding (no ID leakage)
    valid_prefixes = tuple(f"{c}_" for c in CATEGORICAL_COLS)
    dummy_cols = [c for c in df_encoded.columns if c.startswith(valid_prefixes)]

    # Numerical + dummy
    all_features = [f for f in NUMERICAL_FEATURES if f in df_encoded.columns] + dummy_cols

    if fit_columns is not None:
        # Scoring mode: align to training columns
        for col in fit_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        all_features = fit_columns

    X = df_encoded[all_features].fillna(0).astype(np.float32)

    return X.values, list(all_features)
