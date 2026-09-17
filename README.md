# Detecting AI-Generated Speech Using Traditional Speech Processing Features

Course project for Spoken Language Processing (Summer 2026), Birzeit University.

This project investigates whether traditional speech-processing features (MFCC, spectral features, LFCC) can distinguish real human speech (bonafide) from AI-generated speech (spoof), using a subset of the ASVspoof 2021 Deepfake (DF) dataset. We compare SVM, Random Forest, and a CNN across multiple feature representations, and test generalization to unseen spoofing attack types.

## Authors

- Lara Fuqaha
- Taymaa Nasser

## Key Results

| Model | Feature Set | Test Accuracy |
|---|---|---|
| SVM (RBF, C=10) | MFCC + Spectral | **92.1%** |
| Random Forest | MFCC + Spectral | 86.9% |
| SVM | LFCC | 89.6% |
| Random Forest | LFCC | 83.9% |
| CNN | Mel-spectrogram | 87.5% |

Our best model (SVM on MFCC + spectral features) achieved 92.1% test accuracy. A generalization experiment, in which two spoofing attack types (A16, A09) were entirely excluded from training, showed accuracy dropping from 94.5% (seen attacks) to 88.8% (unseen attacks) for SVM, indicating partial but incomplete generalization.

The full write-up, including methodology, detailed results, and discussion, is in the project report).


## Dataset

[ASVspoof 2021 Deepfake (DF) evaluation dataset](https://zenodo.org/record/4835108), accessed via a [Kaggle mirror](https://www.kaggle.com/datasets/pankajsomkuwar/asvspoof-2021-df). Ground-truth labels obtained from the official [ASVspoof 2021 keys and metadata](https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz).

Due to the dataset's size (~34 GB) and severe class imbalance in the full evaluation partition (14,869 bonafide vs. 519,059 spoof samples), we constructed a balanced subset of 2,000 bonafide and 2,000 spoof samples (stratified, fixed random seed) for feature extraction and model training.

## Pipeline Overview

1. **Dataset access and subset construction** — done on Kaggle (raw audio never downloaded locally; see `notebooks/kaggle_feature_extraction.ipynb`)
2. **Feature extraction** — MFCC + spectral (38-dim), LFCC (30-dim), and mel-spectrograms (128x128, for CNN), all extracted on Kaggle and saved as CSV/compressed arrays
3. **Preprocessing** — z-score standardization (fit on training data only)
4. **Model training** — SVM and Random Forest, tuned via 5-fold cross-validated grid search (`GridSearchCV`); CNN trained directly on Kaggle with GPU
5. **Evaluation** — accuracy, precision, recall, F1, confusion matrices, ROC/AUC
6. **Generalization test** — two attack types held out entirely from training, evaluated separately

## Reproducing the Results

### Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn librosa
```

### Running locally (using pre-extracted features)

```bash
python models/train_models.py       # SVM + RF on MFCC + spectral features
python models/train_lfcc.py         # SVM + RF on LFCC features
python models/generalization_test.py # Seen vs. unseen attack type evaluation
```

Each script loads pre-extracted features from `data/`, trains and tunes the models, prints evaluation metrics, and saves plots to `results/`.

### Re-running feature extraction (optional)

Feature extraction requires the raw ASVspoof 2021-DF audio, which is only accessed via Kaggle (not included in this repository due to size). To re-run or modify extraction, open `notebooks/kaggle_feature_extraction.ipynb` on Kaggle with the dataset attached, and re-run the relevant cells.

## Tools and Libraries

- `librosa` — audio loading, MFCC, spectral feature extraction
- `spafe` — LFCC extraction
- `scikit-learn` — SVM, Random Forest, GridSearchCV, evaluation metrics
- `TensorFlow/Keras` — CNN implementation
- `matplotlib`, `seaborn` — plots
