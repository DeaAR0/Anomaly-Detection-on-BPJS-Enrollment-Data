"""
Evaluation and visualization utilities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config


def plot_training_history(history, save_dir=None):
    """Plot AE training/validation loss."""
    save_dir = save_dir or config.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training History")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "ae_training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_dashboard(df_scored: pd.DataFrame, threshold: float, save_dir=None):
    """Generate 6-panel anomaly dashboard."""
    save_dir = save_dir or config.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) AE Score
    ax = axes[0, 0]
    ax.hist(df_scored["ae_score"], bins=100, color="steelblue", alpha=0.7, log=True)
    ax.set_title("AE Reconstruction Error")
    ax.set_xlabel("AE Score (MSE)")
    ax.set_ylabel("Count (log)")

    # (b) IF Score
    ax = axes[0, 1]
    ax.hist(df_scored["if_score"], bins=100, color="darkorange", alpha=0.7, log=True)
    ax.set_title("Isolation Forest Score")
    ax.set_xlabel("IF Score")
    ax.set_ylabel("Count (log)")

    # (c) Hybrid Score
    ax = axes[0, 2]
    ax.hist(df_scored["hybrid_score"], bins=100, color="seagreen", alpha=0.7)
    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold")
    ax.set_title("Hybrid Score Distribution")
    ax.set_xlabel("Hybrid Score")
    ax.set_ylabel("Count")
    ax.legend()

    # (d) AE vs IF scatter
    ax = axes[1, 0]
    sample = df_scored.sample(min(10000, len(df_scored)), random_state=42)
    colors = sample["final_anomaly"].map({0: "steelblue", 1: "red"})
    ax.scatter(sample["ae_rank"], sample["if_rank"], c=colors, alpha=0.3, s=5)
    ax.set_xlabel("AE Rank")
    ax.set_ylabel("IF Rank")
    ax.set_title("AE vs IF Rank (red=anomaly)")

    # (e) Anomaly sources
    ax = axes[1, 1]
    src = df_scored[df_scored["final_anomaly"] == 1]["anomaly_source"].value_counts()
    src.plot(kind="bar", ax=ax, color=["#e74c3c", "#f39c12", "#3498db"])
    ax.set_title("Anomaly Sources")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    # (f) Top reasons
    ax = axes[1, 2]
    reasons = (
        df_scored[df_scored["reason"] != ""]["reason"]
        .str.split("; ")
        .explode()
        .str.strip()
    )
    reasons = reasons[reasons != ""]
    reason_counts = reasons.value_counts().head(10)
    reason_counts.plot(kind="barh", ax=ax, color="teal")
    ax.set_title("Top Anomaly Reasons")
    ax.set_xlabel("Count")

    plt.tight_layout()
    path = os.path.join(save_dir, "anomaly_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_age_distribution(df_scored: pd.DataFrame, save_dir=None):
    """Compare age distributions of normal vs anomaly."""
    save_dir = save_dir or config.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    df_scored[df_scored["final_anomaly"] == 0]["umur"].hist(
        bins=50, alpha=0.6, label="Normal", ax=ax, color="steelblue", density=True
    )
    df_scored[df_scored["final_anomaly"] == 1]["umur"].hist(
        bins=50, alpha=0.6, label="Anomaly", ax=ax, color="red", density=True
    )
    ax.set_title("Age Distribution: Normal vs Anomaly")
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "age_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def print_summary(df_scored: pd.DataFrame, threshold: float):
    """Print anomaly detection summary stats."""
    total = len(df_scored)
    anomalies = df_scored["final_anomaly"].sum()

    print("\n" + "=" * 60)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total records:    {total:,}")
    print(f"Anomalies:        {anomalies:,} ({anomalies/total*100:.2f}%)")
    print(f"Threshold:        {threshold:.6f}")
    print()
    print("By source:")
    print(
        df_scored[df_scored["final_anomaly"] == 1]["anomaly_source"]
        .value_counts()
        .to_string()
    )
    print()
    print("Top 15 anomalies:")
    display_cols = [
        c
        for c in [
            "id_peserta",
            "id_keluarga",
            "peran",
            "umur",
            "jml_keluarga",
            "hybrid_score",
            "anomaly_source",
            "reason",
        ]
        if c in df_scored.columns
    ]
    print(
        df_scored[df_scored["final_anomaly"] == 1]
        .sort_values("hybrid_score", ascending=False)
        .head(15)[display_cols]
        .to_string(index=False)
    )
