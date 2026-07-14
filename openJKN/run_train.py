"""
Entry point: train the anomaly detection model.

Usage:
    cd openJKN
    python run_train.py
"""

import os
import sys
import pandas as pd

# Allow running from openJKN/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anomaly_detection.config import DATA_PATH, OUTPUT_DIR
from anomaly_detection.features import rename_columns, engineer_features, encode_and_select
from anomaly_detection.rules import apply_rules
from anomaly_detection.train import train_pipeline
from anomaly_detection.evaluate import (
    plot_training_history,
    plot_dashboard,
    plot_age_distribution,
    print_summary,
)


def main():
    # --- Load ---
    print("Loading data...")
    df_raw = pd.read_stata(DATA_PATH)
    print(f"Loaded {len(df_raw):,} records\n")

    # --- Features ---
    print("Engineering features...")
    df = rename_columns(df_raw)
    df = engineer_features(df)
    X, feature_names = encode_and_select(df)
    print(f"Feature matrix: {X.shape}\n")

    # --- Train ---
    result = train_pipeline(X, feature_names)

    # --- Score full dataset (for evaluation) ---
    print("\nScoring full dataset for evaluation...")
    df["ae_score"] = result["ae_scores"]
    df["if_score"] = result["if_scores"]
    df["ae_rank"] = result["hybrid_scores"]  # will recompute below
    df["if_rank"] = 0.0
    df["hybrid_score"] = result["hybrid_scores"]

    from scipy.stats import rankdata
    df["ae_rank"] = rankdata(result["ae_scores"]) / len(result["ae_scores"])
    df["if_rank"] = rankdata(result["if_scores"]) / len(result["if_scores"])

    # --- Rules ---
    df = apply_rules(df, df)

    # --- Final flag ---
    threshold = result["threshold"]
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

    # --- Top AE features for anomalies ---
    ae_feature_errors = result["ae_feature_errors"]
    df["top_ae_features"] = ""
    for idx in df.index[df["final_anomaly"] == 1]:
        pos = df.index.get_loc(idx)
        errors = ae_feature_errors[pos]
        top_idx = errors.argsort()[-3:][::-1]
        df.loc[idx, "top_ae_features"] = ", ".join(
            f"{feature_names[i]}({errors[i]:.4f})" for i in top_idx
        )

    # --- Save results ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    anomalies = df[df["final_anomaly"] == 1].sort_values("hybrid_score", ascending=False)
    anomalies.to_csv(os.path.join(OUTPUT_DIR, "anomalies_enrollment.csv"), index=False)

    df.to_csv(os.path.join(OUTPUT_DIR, "all_scores_enrollment.csv"), index=False)

    # --- Evaluate ---
    plot_training_history(result["history"])
    plot_dashboard(df, threshold)
    plot_age_distribution(df)
    print_summary(df, threshold)

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
