"""
Central configuration for the anomaly detection pipeline.
Backend team: adjust paths and thresholds here.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "DATA",
    "Data Sampel 2015-2023",
    "Data Sampel Reguler Edisi 2024",
    "data",
    "2015202301_kepesertaan.dta",
)
MODEL_DIR = os.path.join(BASE_DIR, "anomaly_detection", "saved_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_enrollment")

# --- Column mapping (BPJS coded → readable) ---
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

# --- Feature engineering ---
REFERENCE_DATE = "2023-01-01"  # match data period for age calculation

CATEGORICAL_COLS = [
    "peran",
    "jenis_kelamin",
    "status_kawin",
    "kelas_rawat",
    "jenis_peserta",
]

NUMERICAL_FEATURES = [
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

# --- Autoencoder ---
AE_EPOCHS = 50
AE_BATCH_SIZE = 256
AE_LEARNING_RATE = 1e-3
AE_PATIENCE = 10
AE_ENCODING_DIM = 8
TEST_SIZE = 0.2

# --- Isolation Forest ---
IF_CONTAMINATION = 0.01
IF_N_ESTIMATORS = 200

# --- Anomaly threshold ---
ANOMALY_PERCENTILE = 99  # top 1% flagged
