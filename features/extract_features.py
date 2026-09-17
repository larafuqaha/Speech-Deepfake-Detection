"""
Feature extraction for ASVspoof2021-DF real vs. AI-generated speech detection.

Note: this function was run on Kaggle (where the raw .flac audio lives),
not locally. It's included here for documentation/reproducibility —
running it locally requires the raw ASVspoof2021-DF dataset.
"""

import glob
import librosa
import numpy as np
from tqdm import tqdm

def find_flac(filename):
    matches = glob.glob(f"{base}/*/ASVspoof2021_DF_eval/flac/{filename}.flac")
    return matches[0] if matches else None

def extract_features(filepath, sr=16000, n_mfcc=13):
    y, _ = librosa.load(filepath, sr=sr)
    
    frame_length = int(0.025*sr)
    hop_length = int(0.010*sr)
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=frame_length, hop_length=hop_length)
    
    # Time-domain
    energy = np.array([
        np.sum(np.abs(y[i:i+frame_length]**2))
        for i in range(0, len(y)-frame_length, hop_length)
    ])
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
    
    # Spectral features (new)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    feat_vector = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),                          # 26
        [energy.mean(), energy.std()],                                  # 2
        [zcr.mean(), zcr.std()],                                        # 2
        [spectral_centroid.mean(), spectral_centroid.std()],            # 2
        [spectral_bandwidth.mean(), spectral_bandwidth.std()],          # 2
        [spectral_rolloff.mean(), spectral_rolloff.std()],              # 2
        [spectral_flux.mean(), spectral_flux.std()],                    # 2
    ])
    return feat_vector  # total: 38 features

X, y_labels = [], []
skipped = 0

for _, row in tqdm(subset.iterrows(), total=len(subset)):
    filepath = find_flac(row["filename"])
    if filepath is None:
        skipped += 1
        continue
    try:
        feat = extract_features(filepath)
        X.append(feat)
        y_labels.append(0 if row["label"] == "bonafide" else 1)
    except Exception as e:
        skipped += 1

X = np.array(X)
y_labels = np.array(y_labels)
print(f"Extracted: {X.shape}, Skipped: {skipped}")