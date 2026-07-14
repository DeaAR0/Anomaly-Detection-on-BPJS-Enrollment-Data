# Anomaly Detection on BPJS Enrollment Data

Detecting anomalous enrollment records in BPJS Kesehatan data using a hybrid of deep learning and rule-based methods.

## Background

BPJS Kesehatan manages health insurance enrollment for millions of participants in Indonesia. Enrollment data can contain anomalies ranging from data entry errors (negative age, impossible family sizes) to more complex pattern deviations that are hard to catch with manual rules alone. This project combines an autoencoder, Isolation Forest, and domain-specific business rules to flag these records automatically.

## Method

Three components work together:

1. **Autoencoder** — learns the reconstruction pattern of normal enrollment records. Records with high reconstruction error are likely anomalous.
2. **Isolation Forest** — independently detects outliers in the feature space without needing labeled data.
3. **Business Rules** — hard and soft rules based on domain knowledge (e.g. active member with age > 110, family head younger than 12).

Scores from the autoencoder and Isolation Forest are converted to percentile ranks and averaged, so neither model dominates without arbitrary weighting. Final anomaly flags combine the hybrid score with hard rule overrides.

## Project Structure
```
openJKN/
├── anomaly_detection/
│   ├── config.py          # all paths, thresholds, and hyperparameters
│   ├── features.py        # feature engineering and encoding (shared by train and score)
│   ├── rules.py           # business rules (hard and soft)
│   ├── train.py           # autoencoder + isolation forest training pipeline
│   ├── score.py           # scoring pipeline for new data using saved models
│   └── evaluate.py        # visualization and summary utilities
├── run_train.py           # entry point: train models on enrollment data
├── run_score.py           # entry point: score a new .dta or .csv file
└── anomaly_detection_enrollment.py  # self-contained single-file version (exploratory)

outputs_enrollment/
├── anomaly_dashboard.png      # 6-panel summary dashboard
├── age_distribution.png       # age distribution: normal vs anomaly
└── ae_training_history.png    # autoencoder loss curve
```


## How to Run

**Install dependencies:**

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn scipy joblib
```
Train on enrollment data:

```
cd openJKN
python run_train.py
```
This trains the autoencoder and Isolation Forest, saves the models to anomaly_detection/saved_models/, and writes scored outputs to outputs_enrollment/.

Score a new file:

```
cd openJKN
python run_score.py path/to/new_data.dta
# or
python run_score.py path/to/new_data.csv
```

## Outputs
| File | Description |
|---|---|
| `anomalies_enrollment.csv` | All flagged records with scores, source, and reason |
| `all_scores_enrollment.csv` | Full dataset with anomaly scores for every record |
| `anomaly_dashboard.png` | Score distributions, AE vs IF scatter, anomaly sources, top rule reasons |
| `age_distribution.png` | Age distribution comparison between normal and anomaly records |
| `ae_training_history.png` | Autoencoder training and validation loss per epoch |

The raw enrollment data (2015202301_kepesertaan.dta) is not included in this repository as it contains sensitive participant information. Place the file at:


DATA/Data Sampel 2015-2023/Data Sampel Reguler Edisi 2024/data/2015202301_kepesertaan.dta
The data path can be changed in openJKN/anomaly_detection/config.py.

## Requirements
Python 3.8+
TensorFlow 2.x
scikit-learn
pandas
numpy
matplotlib
seaborn
scipy
joblib

