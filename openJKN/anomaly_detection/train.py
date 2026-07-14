"""
Training pipeline: fits Autoencoder + Isolation Forest and saves artifacts.

Saves to saved_models/:
  - autoencoder.keras       (Keras model)
  - isolation_forest.joblib (sklearn model)
  - scaler.joblib           (MinMaxScaler)
  - feature_names.joblib    (list of feature column names)
  - threshold.joblib        (anomaly score threshold)
"""

import os
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from . import config


def build_autoencoder(input_dim: int) -> Model:
    """Build the autoencoder architecture."""
    inp = Input(shape=(input_dim,), name="input")

    # Encoder
    x = Dense(64, activation="relu", name="enc_1")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu", name="enc_2")(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation="relu", name="enc_3")(x)
    bottleneck = Dense(config.AE_ENCODING_DIM, activation="relu", name="bottleneck")(x)

    # Decoder
    x = Dense(16, activation="relu", name="dec_1")(bottleneck)
    x = Dense(32, activation="relu", name="dec_2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu", name="dec_3")(x)
    x = BatchNormalization()(x)
    out = Dense(input_dim, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=config.AE_LEARNING_RATE), loss="mse")
    return model


def train_pipeline(X: np.ndarray, feature_names: list) -> dict:
    """
    Full training pipeline.

    Args:
        X: feature matrix (unscaled), shape (n_samples, n_features)
        feature_names: list of feature column names

    Returns:
        dict with all trained artifacts (also saved to disk)
    """
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # --- Scale ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Train/val split ---
    X_train, X_val = train_test_split(
        X_scaled, test_size=config.TEST_SIZE, random_state=42
    )
    print(f"Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}")

    # --- Autoencoder ---
    autoencoder = build_autoencoder(X_scaled.shape[1])
    autoencoder.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.AE_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    history = autoencoder.fit(
        X_train, X_train,
        epochs=config.AE_EPOCHS,
        batch_size=config.AE_BATCH_SIZE,
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1,
    )

    # --- AE scores (on full data) ---
    X_pred = autoencoder.predict(X_scaled, batch_size=config.AE_BATCH_SIZE)
    ae_scores = np.mean((X_scaled - X_pred) ** 2, axis=1)
    ae_feature_errors = (X_scaled - X_pred) ** 2

    # --- Isolation Forest ---
    iso_forest = IsolationForest(
        n_estimators=config.IF_N_ESTIMATORS,
        contamination=config.IF_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_scaled)
    if_raw_scores = iso_forest.decision_function(X_scaled)
    if_scores = -if_raw_scores  # higher = more anomalous

    # --- Threshold (from hybrid rank score) ---
    from scipy.stats import rankdata

    ae_ranks = rankdata(ae_scores) / len(ae_scores)
    if_ranks = rankdata(if_scores) / len(if_scores)
    hybrid_scores = (ae_ranks + if_ranks) / 2
    threshold = np.percentile(hybrid_scores, config.ANOMALY_PERCENTILE)

    # --- Save everything ---
    autoencoder.save(os.path.join(config.MODEL_DIR, "autoencoder.keras"))
    joblib.dump(iso_forest, os.path.join(config.MODEL_DIR, "isolation_forest.joblib"))
    joblib.dump(scaler, os.path.join(config.MODEL_DIR, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(config.MODEL_DIR, "feature_names.joblib"))
    joblib.dump(threshold, os.path.join(config.MODEL_DIR, "threshold.joblib"))

    print(f"\nModels saved to {config.MODEL_DIR}/")
    print(f"Threshold (P{config.ANOMALY_PERCENTILE}): {threshold:.6f}")

    return {
        "autoencoder": autoencoder,
        "iso_forest": iso_forest,
        "scaler": scaler,
        "feature_names": feature_names,
        "threshold": threshold,
        "history": history,
        "ae_scores": ae_scores,
        "if_scores": if_scores,
        "hybrid_scores": hybrid_scores,
        "ae_feature_errors": ae_feature_errors,
        "X_scaled": X_scaled,
    }
