'''
import os
import pickle
import base64
import numpy as np

from audio_pipeline import process_audio_pipeline
from feature_extractor import extract_features_from_array

print("\n==============================")
print("Running LOCAL Prediction (ALL FILES)...")
print("==============================\n")

# ---------- LOAD TRAINED MODEL ----------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------- DATA FOLDER ----------
DATA_DIR = "data"

# ---------- SUPPORTED AUDIO TYPES ----------
SUPPORTED_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma")

print("=== Batch Prediction Results ===\n")

# ---------- LOOP THROUGH ALL FILES ----------
for root, dirs, files in os.walk(DATA_DIR):

    if len(files) == 0:
        continue

    folder_name = os.path.basename(root)
    print(f"[Folder: {folder_name}]")

    for fname in files:

        if not fname.lower().endswith(SUPPORTED_EXT):
            continue

        path = os.path.join(root, fname)

        try:
            print(f"Processing: {path}")

            # ---------- READ FILE ----------
            with open(path, "rb") as f:
                audio_bytes = f.read()

            # ---------- BASE64 ----------
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

            # ---------- AUDIO PIPELINE ----------
            y = process_audio_pipeline(b64_audio)

            if y is None:
                print("❌ Failed pipeline")
                continue

            # ---------- FEATURE EXTRACTION ----------
            features = extract_features_from_array(y)

            # ---------- MODEL PREDICTION ----------
            X = features.reshape(1, -1)

            prediction = model.predict(X)[0]
            confidence = float(np.max(model.predict_proba(X)))

            label = "AI_GENERATED" if int(prediction) == 1 else "HUMAN"

            print(
                f"{fname} | Predicted: {label} | Confidence: {round(confidence,3)}\n"
            )

        except Exception as e:
            print(f"❌ Error Processing {fname}: {str(e)}")

    print()

print("=========== DONE ===========\n")

import joblib
import numpy as np
from feature_extractor import extract_features_from_array

model = joblib.load("model.pkl")

def predict_audio(waveform, sr):
    # Feature extraction
    features = extract_features_from_array(waveform, sr)

    # Reshape for model
    X = features.reshape(1, -1)

    # Prediction
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label = "AI_GENERATED" if int(pred) == 1 else "HUMAN"
    confidence = float(np.max(proba))

    return label, confidence
    '''
import joblib
import numpy as np
from feature_extractor import extract_features_from_array

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")

# ---------- PREDICTION FUNCTION ----------
def predict_audio(waveform, sr):
    try:
        # Feature extraction
        features = extract_features_from_array(waveform, sr)

        # Reshape
        X = features.reshape(1, -1)

        # Predict
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        label = "AI_GENERATED" if int(pred) == 1 else "HUMAN"
        confidence = float(np.max(proba))

        return label, confidence

    except Exception as e:
        print("Prediction error:", e)
        return "ERROR", 0.0
