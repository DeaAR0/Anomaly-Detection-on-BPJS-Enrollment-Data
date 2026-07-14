"""
Anomaly Detection for Enrollment Data
- Autoencoder learns normal enrollment patterns; high reconstruction error = anomaly
- Isolation Forest independently detects outliers in feature space
- Rank-based fusion combines both scores without arbitrary weighting
- Business rules add domain-specific explainability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from scipy.stats import percentileofscore

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
np.random.seed(42)

# --- Config ---
DATA_PATH = "DATA/Data Sampel 2015-2023/Data Sampel Reguler Edisi 2024/data/2015202301_kepesertaan.dta"
OUTPUT_DIR = "outputs_enrollment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ANOMALY_PERCENTILE = 99  # top 1%

AE_EPOCHS = 50
AE_BATCH_SIZE = 256
AE_LEARNING_RATE = 1e-3
AE_PATIENCE = 10
AE_ENCODING_DIM = 8

IF_CONTAMINATION = 0.01
IF_N_ESTIMATORS = 200

TEST_SIZE = 0.2


# --- Load data ---
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_stata(DATA_PATH)

COLUMN_MAP = {
    "PSTV01": "id_peserta",
    "PSTV02": "id_keluarga",
    "PSTV03": "tanggal_lahir",
    "PSTV04": "peran",
    "PSTV05": "jenis_kelamin",
    "PSTV06": "status_kawin",
    "PSTV07": "kelas_rawat",
    "PSTV08": "jenis_peserta",
    "PSTV09": "provinsi",
    "PSTV09_NEW": "provinsi_new",
    "PSTV10": "kabupaten",
    "PSTV11": "jenis_pemberi_kerja",
    "PSTV12": "jenis_faskes",
    "PSTV13": "provinsi_faskes",
    "PSTV14": "kabupaten_faskes",
    "PSTV15": "kapitasi",
    "PSTV16": "tahun_data",
    "PSTV17": "status_peserta",
    "PSTV18": "status",
}

df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

print(f"Loaded {len(df):,} records")
print(f"Columns: {list(df.columns)}")
print()


# --- Feature engineering ---
print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Age
df["tanggal_lahir"] = pd.to_datetime(df["tanggal_lahir"], errors="coerce")
reference_date = pd.Timestamp("2023-01-01")  # match data period, not today
df["umur"] = (reference_date - df["tanggal_lahir"]).dt.days // 365

# Active status
df["is_active"] = (df["status_peserta"] == "AKTIF").astype(int)

# Family-level features
grp = df.groupby("id_keluarga")

df["jml_keluarga"] = grp["id_peserta"].transform("count")
df["jml_keluarga_aktif"] = grp["is_active"].transform("sum")
df["rasio_aktif"] = df["jml_keluarga_aktif"] / df["jml_keluarga"]

# Does this family have a head (PESERTA or SUAMI/ISTRI)?
df["is_kepala"] = df["peran"].isin(["PESERTA", "SUAMI", "ISTRI"]).astype(int)
df["kepala_exists"] = grp["is_kepala"].transform("max")

# Number of unique roles in the family
df["jml_peran_unik"] = grp["peran"].transform("nunique")

# Age-role consistency
df["umur_kategori"] = pd.cut(
    df["umur"],
    bins=[-1, 0, 5, 17, 25, 55, 70, 200],
    labels=["bayi", "balita", "anak", "muda", "dewasa", "lansia", "sangat_tua"],
)

# Flag: child role but adult age, or vice versa
df["anak_tapi_dewasa"] = (
    (df["peran"] == "ANAK") & (df["umur"] > 25)
).astype(int)

df["kepala_tapi_muda"] = (
    (df["peran"].isin(["PESERTA", "SUAMI", "ISTRI"])) & (df["umur"] < 17)
).astype(int)

# Kapitasi as numeric
if "kapitasi" in df.columns:
    df["kapitasi"] = pd.to_numeric(df["kapitasi"], errors="coerce")
else:
    df["kapitasi"] = 0.0

# Marriage-age consistency
df["kawin_tapi_muda"] = (
    (df["status_kawin"] == "KAWIN") & (df["umur"] < 16)
).astype(int)

print(f"Features engineered. Shape: {df.shape}")
print()


# --- Encoding & feature selection ---
print("=" * 60)
print("ENCODING & FEATURE SELECTION")
print("=" * 60)

categorical_cols = [
    "peran",
    "jenis_kelamin",
    "status_kawin",
    "kelas_rawat",
    "jenis_peserta",
]

# Verify columns exist before encoding
categorical_cols = [c for c in categorical_cols if c in df.columns]

df_model = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Numerical features (hand-picked, no ID leakage)
numerical_features = [
    "umur",
    "is_active",
    "jml_keluarga",
    "jml_keluarga_aktif",
    "rasio_aktif",
    "kepala_exists",
    "jml_peran_unik",
    "anak_tapi_dewasa",
    "kepala_tapi_muda",
    "kawin_tapi_muda",
    "kapitasi",
]

# Dummy features (explicitly filtered by prefix to avoid ID leakage)
VALID_PREFIXES = tuple(f"{c}_" for c in categorical_cols)
dummy_features = [c for c in df_model.columns if c.startswith(VALID_PREFIXES)]

# Combine
all_features = numerical_features + dummy_features
# Keep only numeric columns that actually exist
all_features = [f for f in all_features if f in df_model.columns]

X = df_model[all_features].fillna(0).astype(np.float32)

print(f"Total features: {len(all_features)}")
print(f"  Numerical: {len(numerical_features)}")
print(f"  Dummy:     {len(dummy_features)}")
print(f"  X shape:   {X.shape}")
print()


# --- Scaling ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# --- Train / validation split ---
X_train, X_val = train_test_split(
    X_scaled, test_size=TEST_SIZE, random_state=42
)

print(f"Train size: {X_train.shape[0]:,}")
print(f"Val size:   {X_val.shape[0]:,}")
print()


# --- Autoencoder ---
print("=" * 60)
print("TRAINING AUTOENCODER")
print("=" * 60)

input_dim = X_scaled.shape[1]

# Encoder
input_layer = Input(shape=(input_dim,), name="input")
x = Dense(64, activation="relu", name="enc_1")(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(32, activation="relu", name="enc_2")(x)
x = BatchNormalization()(x)
x = Dense(16, activation="relu", name="enc_3")(x)
bottleneck = Dense(AE_ENCODING_DIM, activation="relu", name="bottleneck")(x)

# Decoder (mirror)
x = Dense(16, activation="relu", name="dec_1")(bottleneck)
x = Dense(32, activation="relu", name="dec_2")(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu", name="dec_3")(x)
x = BatchNormalization()(x)
output_layer = Dense(input_dim, activation="sigmoid", name="output")(x)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(
    optimizer=Adam(learning_rate=AE_LEARNING_RATE),
    loss="mse",
)
autoencoder.summary()

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=AE_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
]

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=AE_EPOCHS,
    batch_size=AE_BATCH_SIZE,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=callbacks,
    verbose=1,
)

# Plot training history
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(history.history["loss"], label="Train Loss")
ax.plot(history.history["val_loss"], label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Autoencoder Training History")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ae_training_history.png", dpi=150)
plt.close()
print(f"Training history saved to {OUTPUT_DIR}/ae_training_history.png")
print()


# --- AE anomaly scores ---
print("=" * 60)
print("COMPUTING AE SCORES")
print("=" * 60)

X_pred = autoencoder.predict(X_scaled, batch_size=AE_BATCH_SIZE)

# Per-sample mean squared reconstruction error
ae_scores = np.mean((X_scaled - X_pred) ** 2, axis=1)

# Per-feature reconstruction error (for explainability later)
ae_feature_errors = (X_scaled - X_pred) ** 2

df_model["ae_score"] = ae_scores

print(f"AE score stats:")
print(f"  Mean:   {ae_scores.mean():.6f}")
print(f"  Median: {np.median(ae_scores):.6f}")
print(f"  P95:    {np.percentile(ae_scores, 95):.6f}")
print(f"  P99:    {np.percentile(ae_scores, 99):.6f}")
print(f"  Max:    {ae_scores.max():.6f}")
print()


# --- Isolation Forest ---
print("=" * 60)
print("TRAINING ISOLATION FOREST")
print("=" * 60)

iso_forest = IsolationForest(
    n_estimators=IF_N_ESTIMATORS,
    contamination=IF_CONTAMINATION,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

iso_forest.fit(X_scaled)

# Raw anomaly score from IF (more negative = more anomalous)
if_raw_scores = iso_forest.decision_function(X_scaled)

# Invert so higher = more anomalous (consistent with AE)
df_model["if_score"] = -if_raw_scores

# Binary flag
if_labels = iso_forest.predict(X_scaled)
df_model["if_flag"] = (if_labels == -1).astype(int)

print(f"IF flagged: {df_model['if_flag'].sum():,} records ({df_model['if_flag'].mean()*100:.2f}%)")
print()


# --- Rank-based hybrid score ---
print("=" * 60)
print("COMPUTING HYBRID SCORE (RANK-BASED)")
print("=" * 60)

# Convert raw scores to percentile ranks (0-1)
# This avoids arbitrary weighting — both models contribute equally on a normalized scale
ae_ranks = df_model["ae_score"].rank(pct=True)
if_ranks = df_model["if_score"].rank(pct=True)

# Average rank (equal weight, but now on the same scale)
df_model["hybrid_score"] = (ae_ranks + if_ranks) / 2

# Also store individual normalized scores for analysis
df_model["ae_rank"] = ae_ranks
df_model["if_rank"] = if_ranks

print(f"Hybrid score stats:")
print(f"  Mean:   {df_model['hybrid_score'].mean():.4f}")
print(f"  P95:    {df_model['hybrid_score'].quantile(0.95):.4f}")
print(f"  P99:    {df_model['hybrid_score'].quantile(0.99):.4f}")
print()


# --- Business rules ---
print("=" * 60)
print("APPLYING BUSINESS RULES")
print("=" * 60)

df_model["rule_flag"] = 0
df_model["reason"] = ""


def flag_hard(mask, reason_text):
    """Hard rule: forces anomaly flag regardless of model score."""
    df_model.loc[mask, "rule_flag"] = 1
    df_model.loc[mask, "reason"] += reason_text + "; "


def flag_soft(mask, reason_text):
    """Soft rule: adds explanation but doesn't force flag."""
    df_model.loc[mask, "reason"] += reason_text + "; "


# --- Hard rules (override model) ---

# Family size impossibly large (data error)
flag_hard(
    df_model["jml_keluarga"] > 50,
    "KELUARGA_>50_ANGGOTA"
)

# Active member with impossible age
flag_hard(
    (df_model["is_active"] == 1) & (df_model["umur"] > 110),
    "AKTIF_UMUR_>110"
)

# Negative age (birth date in the future)
flag_hard(
    df_model["umur"] < 0,
    "UMUR_NEGATIF"
)

# Head of family is a child (<12 years old)
flag_hard(
    (df["peran"].isin(["PESERTA", "SUAMI", "ISTRI"])) & (df_model["umur"] < 12),
    "KEPALA_KELUARGA_ANAK"
)

# --- Soft rules (explanation only) ---

# Family with no head
flag_soft(
    (df_model["kepala_exists"] == 0) & (df_model["jml_keluarga"] > 1),
    "TANPA_KEPALA_KELUARGA"
)

# Very low active ratio in large family
flag_soft(
    (df_model["rasio_aktif"] < 0.2) & (df_model["jml_keluarga"] > 5),
    "RASIO_AKTIF_RENDAH"
)

# Large family
flag_soft(
    df_model["jml_keluarga"] > 10,
    "KELUARGA_BESAR"
)

# Child role but registered as PBPU (self-paying)
if "jenis_peserta_PBPU" in df_model.columns:
    flag_soft(
        (df["peran"] == "ANAK") & (df_model.get("jenis_peserta_PBPU", 0) == 1),
        "ANAK_TAPI_PBPU"
    )

# Married but very young
flag_soft(
    df_model["kawin_tapi_muda"] == 1,
    "KAWIN_UMUR_<16"
)

# Child role but age > 25
flag_soft(
    df_model["anak_tapi_dewasa"] == 1,
    "ANAK_TAPI_UMUR_>25"
)

# Active but very old (90-110 range, softer than hard rule)
flag_soft(
    (df_model["is_active"] == 1) & (df_model["umur"].between(90, 110)),
    "AKTIF_LANSIA_>90"
)

# Abnormally high kapitasi
if "kapitasi" in df_model.columns and df_model["kapitasi"].max() > 0:
    q99 = df_model["kapitasi"].quantile(0.99)
    flag_soft(
        df_model["kapitasi"] > q99,
        "KAPITASI_SANGAT_TINGGI"
    )

hard_count = df_model["rule_flag"].sum()
soft_count = (df_model["reason"] != "").sum()
print(f"Hard rule flags: {hard_count:,}")
print(f"Records with reasons: {soft_count:,}")
print()


# --- Final anomaly decision ---
print("=" * 60)
print("FINAL ANOMALY CLASSIFICATION")
print("=" * 60)

threshold = df_model["hybrid_score"].quantile(ANOMALY_PERCENTILE / 100)

df_model["final_anomaly"] = (
    (df_model["hybrid_score"] > threshold) | (df_model["rule_flag"] == 1)
).astype(int)

# Anomaly source
df_model["anomaly_source"] = "normal"
df_model.loc[
    (df_model["hybrid_score"] > threshold) & (df_model["rule_flag"] == 0),
    "anomaly_source",
] = "model_only"
df_model.loc[
    (df_model["hybrid_score"] <= threshold) & (df_model["rule_flag"] == 1),
    "anomaly_source",
] = "rule_only"
df_model.loc[
    (df_model["hybrid_score"] > threshold) & (df_model["rule_flag"] == 1),
    "anomaly_source",
] = "model+rule"

total_anomalies = df_model["final_anomaly"].sum()
print(f"Threshold (P{ANOMALY_PERCENTILE}): {threshold:.4f}")
print(f"Total anomalies: {total_anomalies:,} / {len(df_model):,} ({total_anomalies/len(df_model)*100:.2f}%)")
print()
print("Anomaly sources:")
print(df_model["anomaly_source"].value_counts().to_string())
print()


# --- Top feature contributors (explainability) ---
def get_top_features(idx, top_n=3):
    """Get the top N features contributing to reconstruction error for a sample."""
    errors = ae_feature_errors[idx]
    top_indices = np.argsort(errors)[-top_n:][::-1]
    return ", ".join([f"{all_features[i]}({errors[i]:.4f})" for i in top_indices])


# Add top contributing features for anomalies
anomaly_mask = df_model["final_anomaly"] == 1
anomaly_indices = df_model.index[anomaly_mask]

df_model["top_ae_features"] = ""
for idx in anomaly_indices:
    pos = df_model.index.get_loc(idx)
    df_model.loc[idx, "top_ae_features"] = get_top_features(pos)


# --- Save results ---
print("=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Full anomaly list
id_cols = ["id_peserta", "id_keluarga"]
id_cols = [c for c in id_cols if c in df.columns]

output_cols = id_cols + [
    "umur",
    "jml_keluarga",
    "rasio_aktif",
    "ae_score",
    "ae_rank",
    "if_score",
    "if_rank",
    "hybrid_score",
    "final_anomaly",
    "anomaly_source",
    "reason",
    "top_ae_features",
]

# Build output from original df (IDs) + df_model (scores)
result = pd.DataFrame(index=df_model.index)
for col in id_cols:
    if col in df.columns:
        result[col] = df[col]
for col in output_cols:
    if col not in result.columns and col in df_model.columns:
        result[col] = df_model[col]
    elif col not in result.columns and col in df.columns:
        result[col] = df[col]

# Sort by hybrid score descending
result = result.sort_values("hybrid_score", ascending=False)

# Save all anomalies
anomalies_df = result[result["final_anomaly"] == 1]
anomalies_df.to_csv(f"{OUTPUT_DIR}/anomalies_enrollment.csv", index=False)
print(f"Saved {len(anomalies_df):,} anomalies to {OUTPUT_DIR}/anomalies_enrollment.csv")

# Save full scored dataset
result.to_csv(f"{OUTPUT_DIR}/all_scores_enrollment.csv", index=False)
print(f"Saved full scored data to {OUTPUT_DIR}/all_scores_enrollment.csv")

# Print top 20
print()
print("=" * 60)
print("TOP 20 ANOMALIES")
print("=" * 60)
display_cols = id_cols + [
    "umur",
    "jml_keluarga",
    "hybrid_score",
    "anomaly_source",
    "reason",
]
display_cols = [c for c in display_cols if c in anomalies_df.columns]
print(anomalies_df[display_cols].head(20).to_string(index=False))
print()


# --- Visualizations ---
print("=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) AE Score Distribution
ax = axes[0, 0]
ax.hist(df_model["ae_score"], bins=100, color="steelblue", alpha=0.7, log=True)
ax.axvline(
    df_model.loc[df_model["final_anomaly"] == 1, "ae_score"].min(),
    color="red", linestyle="--", label="Anomaly boundary"
)
ax.set_title("AE Reconstruction Error Distribution")
ax.set_xlabel("AE Score (MSE)")
ax.set_ylabel("Count (log)")
ax.legend()

# (b) IF Score Distribution
ax = axes[0, 1]
ax.hist(df_model["if_score"], bins=100, color="darkorange", alpha=0.7, log=True)
ax.set_title("Isolation Forest Score Distribution")
ax.set_xlabel("IF Score (inverted)")
ax.set_ylabel("Count (log)")

# (c) Hybrid Score Distribution
ax = axes[0, 2]
ax.hist(df_model["hybrid_score"], bins=100, color="seagreen", alpha=0.7)
ax.axvline(threshold, color="red", linestyle="--", label=f"P{ANOMALY_PERCENTILE} threshold")
ax.set_title("Hybrid Score Distribution")
ax.set_xlabel("Hybrid Score")
ax.set_ylabel("Count")
ax.legend()

# (d) AE vs IF scatter
ax = axes[1, 0]
sample = df_model.sample(min(10000, len(df_model)), random_state=42)
colors = sample["final_anomaly"].map({0: "steelblue", 1: "red"})
ax.scatter(sample["ae_rank"], sample["if_rank"], c=colors, alpha=0.3, s=5)
ax.set_xlabel("AE Rank")
ax.set_ylabel("IF Rank")
ax.set_title("AE vs IF Rank (red=anomaly)")

# (e) Anomaly source breakdown
ax = axes[1, 1]
source_counts = df_model[df_model["final_anomaly"] == 1]["anomaly_source"].value_counts()
source_counts.plot(kind="bar", ax=ax, color=["#e74c3c", "#f39c12", "#3498db"])
ax.set_title("Anomaly Sources")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)

# (f) Top reasons
ax = axes[1, 2]
all_reasons = (
    df_model[df_model["reason"] != ""]["reason"]
    .str.split("; ")
    .explode()
    .str.strip()
)
all_reasons = all_reasons[all_reasons != ""]
reason_counts = all_reasons.value_counts().head(10)
reason_counts.plot(kind="barh", ax=ax, color="teal")
ax.set_title("Top Anomaly Reasons")
ax.set_xlabel("Count")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/anomaly_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Dashboard saved to {OUTPUT_DIR}/anomaly_dashboard.png")

# Age distribution by anomaly
fig, ax = plt.subplots(figsize=(10, 5))
df_model[df_model["final_anomaly"] == 0]["umur"].hist(
    bins=50, alpha=0.6, label="Normal", ax=ax, color="steelblue", density=True
)
df_model[df_model["final_anomaly"] == 1]["umur"].hist(
    bins=50, alpha=0.6, label="Anomaly", ax=ax, color="red", density=True
)
ax.set_title("Age Distribution: Normal vs Anomaly")
ax.set_xlabel("Age")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/age_distribution.png", dpi=150)
plt.close()
print(f"Age distribution saved to {OUTPUT_DIR}/age_distribution.png")

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print(f"Results in: {OUTPUT_DIR}/")
print(f"  - anomalies_enrollment.csv   (flagged records)")
print(f"  - all_scores_enrollment.csv  (full scored dataset)")
print(f"  - anomaly_dashboard.png      (6-panel dashboard)")
print(f"  - age_distribution.png       (age comparison)")
print(f"  - ae_training_history.png    (AE loss curve)")
