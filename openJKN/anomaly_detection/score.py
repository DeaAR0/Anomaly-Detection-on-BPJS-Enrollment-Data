"""
Scoring pipeline: loads saved models and scores new enrollment data.
This is what the backend team will call.
"""

import os
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from scipy.stats import rankdata

from . import config
from .features import rename_columns, engineer_features, encode_and_select
from .rules import apply_rules


def load_artifacts() -> dict:
    """Load all saved model artifacts from disk."""
    artifacts = {
        "autoencoder": load_model(
            os.path.join(config.MODEL_DIR, "autoencoder.keras")
        ),
        "iso_forest": joblib.load(
            os.path.join(config.MODEL_DIR, "isolation_forest.joblib")
        ),
        "scaler": joblib.load(os.path.join(config.MODEL_DIR, "scaler.joblib")),
        "feature_names": joblib.load(
            os.path.join(config.MODEL_DIR, "feature_names.joblib")
        ),
        "threshold": joblib.load(os.path.join(config.MODEL_DIR, "threshold.joblib")),
    }
    print(f"Loaded models from {config.MODEL_DIR}/")
    return artifacts


def score_dataframe(df_raw: pd.DataFrame, artifacts: dict = None) -> pd.DataFrame:
    """
    Score a DataFrame of enrollment records.

    Args:
        df_raw: raw enrollment data (BPJS coded or already renamed)
        artifacts: pre-loaded artifacts dict, or None to load from disk

    Returns:
        DataFrame with original columns + anomaly scores and flags
    """
    if artifacts is None:
        artifacts = load_artifacts()

    ae = artifacts["autoencoder"]
    iso = artifacts["iso_forest"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    threshold = artifacts["threshold"]

    # --- Prepare ---
    df = rename_columns(df_raw)
    df = engineer_features(df)
    X, _ = encode_and_select(df, fit_columns=feature_names)
    X_scaled = scaler.transform(X)

    # --- AE scores ---
    X_pred = ae.predict(X_scaled, batch_size=config.AE_BATCH_SIZE, verbose=0)
    ae_scores = np.mean((X_scaled - X_pred) ** 2, axis=1)
    ae_feature_errors = (X_scaled - X_pred) ** 2

    # --- IF scores ---
    if_raw = iso.decision_function(X_scaled)
    if_scores = -if_raw

    # --- Hybrid rank score ---
    ae_ranks = rankdata(ae_scores) / len(ae_scores)
    if_ranks = rankdata(if_scores) / len(if_scores)
    hybrid_scores = (ae_ranks + if_ranks) / 2

    # --- Attach scores to df ---
    df["ae_score"] = ae_scores
    df["if_score"] = if_scores
    df["ae_rank"] = ae_ranks
    df["if_rank"] = if_ranks
    df["hybrid_score"] = hybrid_scores

    # --- Business rules ---
    df = apply_rules(df, df)

    # --- Final decision ---
    df["final_anomaly"] = (
        (df["hybrid_score"] > threshold) | (df["rule_flag"] == 1)
    ).astype(int)

    df["anomaly_source"] = "normal"
    df.loc[
        (df["hybrid_score"] > threshold) & (df["rule_flag"] == 0), "anomaly_source"
    ] = "model_only"
    df.loc[
        (df["hybrid_score"] <= threshold) & (df["rule_flag"] == 1), "anomaly_source"
    ] = "rule_only"
    df.loc[
        (df["hybrid_score"] > threshold) & (df["rule_flag"] == 1), "anomaly_source"
    ] = "model+rule"

    # --- Top contributing features (for anomalies) ---
    df["top_ae_features"] = ""
    anomaly_indices = df.index[df["final_anomaly"] == 1]
    for idx in anomaly_indices:
        pos = df.index.get_loc(idx)
        errors = ae_feature_errors[pos]
        top_idx = np.argsort(errors)[-3:][::-1]
        df.loc[idx, "top_ae_features"] = ", ".join(
            f"{feature_names[i]}({errors[i]:.4f})" for i in top_idx
        )

    return df


def score_single(record: dict, artifacts: dict = None) -> dict:
    """
    Score a single enrollment record.
    Convenience wrapper for the backend team's API.

    Args:
        record: dict with enrollment fields (BPJS coded or renamed)
        artifacts: pre-loaded artifacts, or None to load from disk

    Returns:
        dict with the record + anomaly scores
    """
    df = pd.DataFrame([record])
    result = score_dataframe(df, artifacts)
    return result.iloc[0].to_dict()
