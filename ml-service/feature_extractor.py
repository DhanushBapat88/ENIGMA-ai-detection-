import numpy as np
import librosa
import os

def extract_features(wav_path: str):
    if not os.path.isfile(wav_path):
        raise ValueError(f"Audio file not found: {wav_path}")

    # 1. Load audio
    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    # Safety: very short audio is useless
    if len(y) < sr:  # < 1 second
        raise ValueError("Audio too short to extract features")

    features = []

    # 2. MFCCs (40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(mfcc.mean(axis=1))
    features.extend(mfcc.std(axis=1))

    # 3. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(centroid.mean())

    # 4. Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(rolloff.mean())

    # 5. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(zcr.mean())

    # 6. RMS Energy
    rms = librosa.feature.rms(y=y)
    features.append(rms.mean())

    # 7. Pitch (F0)
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
#if __name__ == "__main__":
   # feats = extract_features("test.wav")
   # print("Type:", type(feats))
    #print("Shape:", feats.shape)
    #print("NaN present:", np.isnan(feats).any())
   # print("First 10 values:", feats[:10])