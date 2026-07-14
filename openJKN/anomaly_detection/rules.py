"""
Business rules for enrollment anomaly detection.
Separated so domain experts / backend team can update rules
without touching the model code.
"""

import pandas as pd


def apply_rules(df: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Apply business rules to the scored dataframe.

    Args:
        df: df_model with features and scores
        df_original: original df with raw columns (peran, status_kawin, etc.)

    Returns:
        df with 'rule_flag' and 'reason' columns added
    """
    df = df.copy()
    df["rule_flag"] = 0
    df["reason"] = ""

    def _hard(mask, text):
        df.loc[mask, "rule_flag"] = 1
        df.loc[mask, "reason"] += text + "; "

    def _soft(mask, text):
        df.loc[mask, "reason"] += text + "; "

    # --- Hard rules - force anomaly flag ---

    # Family size impossibly large
    _hard(df["jml_keluarga"] > 50, "KELUARGA_>50_ANGGOTA")

    # Active with impossible age
    _hard((df["is_active"] == 1) & (df["umur"] > 110), "AKTIF_UMUR_>110")

    # Negative age (future birth date)
    _hard(df["umur"] < 0, "UMUR_NEGATIF")

    # Family head is a child
    _hard(
        df_original["peran"].isin(["PESERTA", "SUAMI", "ISTRI"]) & (df["umur"] < 12),
        "KEPALA_KELUARGA_ANAK",
    )

    # --- Soft rules - explanation only ---

    # Family without a head
    _soft(
        (df["kepala_exists"] == 0) & (df["jml_keluarga"] > 1),
        "TANPA_KEPALA_KELUARGA",
    )

    # Low active ratio in large family
    _soft(
        (df["rasio_aktif"] < 0.2) & (df["jml_keluarga"] > 5),
        "RASIO_AKTIF_RENDAH",
    )

    # Large family
    _soft(df["jml_keluarga"] > 10, "KELUARGA_BESAR")

    # Child role but self-paying (PBPU)
    if "jenis_peserta_PBPU" in df.columns:
        _soft(
            (df_original["peran"] == "ANAK") & (df["jenis_peserta_PBPU"] == 1),
            "ANAK_TAPI_PBPU",
        )

    # Married but very young
    _soft(df["kawin_tapi_muda"] == 1, "KAWIN_UMUR_<16")

    # Child role but adult age
    _soft(df["anak_tapi_dewasa"] == 1, "ANAK_TAPI_UMUR_>25")

    # Active elderly
    _soft(
        (df["is_active"] == 1) & (df["umur"].between(90, 110)),
        "AKTIF_LANSIA_>90",
    )

    # Abnormally high kapitasi
    if "kapitasi" in df.columns and df["kapitasi"].max() > 0:
        q99 = df["kapitasi"].quantile(0.99)
        _soft(df["kapitasi"] > q99, "KAPITASI_SANGAT_TINGGI")

    return df
