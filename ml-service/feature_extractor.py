
import numpy as np
import librosa
import os


# =====================================================
# Core feature logic (shared)
# =====================================================
def _extract_features_core(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    if len(y) < sr:  # at least 1 second
        raise ValueError("Audio too short to extract features")

    features = []

    # MFCCs (40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(mfcc.mean(axis=1))
    features.extend(mfcc.std(axis=1))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(centroid.mean())

    # Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(rolloff.mean())

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(zcr.mean())

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features.append(rms.mean())

    # Pitch (F0)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    if f0 is not None:
        f0 = f0[~np.isnan(f0)]
        if len(f0) > 0:
            features.append(f0.mean())
            features.append(f0.std())
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


# =====================================================
# Used by FastAPI (audio already in memory)
# =====================================================
def extract_features_from_array(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    return _extract_features_core(y, sr)


# =====================================================
# Used by training / batch scripts (file path)
# =====================================================
def extract_features_from_file(wav_path: str) -> np.ndarray:
    if not os.path.isfile(wav_path):
        raise ValueError(f"Audio file not found: {wav_path}")

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    return _extract_features_core(y, sr)