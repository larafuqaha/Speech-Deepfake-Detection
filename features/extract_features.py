"""
Feature extraction for ASVspoof2021-DF real vs. AI-generated speech detection.

Note: this function was run on Kaggle (where the raw .flac audio lives),
not locally. It's included here for documentation/reproducibility —
running it locally requires the raw ASVspoof2021-DF dataset.
"""

import librosa
import numpy as np


def extract_features(filepath, sr=16000, n_mfcc=13):
    y, _ = librosa.load(filepath, sr=sr)

    frame_length = int(0.025 * sr)  # 25ms window
    hop_length = int(0.010 * sr)    # 10ms shift

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length
    )

    energy = np.array([
        np.sum(np.abs(y[i:i + frame_length] ** 2))
        for i in range(0, len(y) - frame_length, hop_length)
    ])

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )

    feat_vector = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        [energy.mean(), energy.std()],
        [zcr.mean(), zcr.std()]
    ])
    return feat_vector