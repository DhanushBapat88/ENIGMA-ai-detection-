import os
import pickle
import numpy as np
import base64

from audio_pipeline import process_audio_pipeline
from feature_extractor import extract_features, extract_features_from_array

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data")


def predict_from_b64(model, b64_string):
    y = process_audio_pipeline(b64_string)
    if y is None:
        raise RuntimeError("Pipeline failed to decode/process audio")

    feats = extract_features_from_array(y, sr=16000).reshape(1, -1)
    probs = model.predict_proba(feats)[0]
    pred = int(np.argmax(probs))

    label = "AI_GENERATED" if pred == 1 else "HUMAN"
    confidence = float(probs[pred])

    return label, confidence, probs


def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ model.pkl missing. Run train_model.py first.")
        return

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Demo: use first .mp3 in DATA_DIR (if present) — read bytes -> base64 -> predict
    def _get_demo_b64_from_mp3():
        for root, _, files in os.walk(DATA_DIR):
            for fname in files:
                if fname.lower().endswith('.mp3'):
                    path = os.path.join(root, fname)
                    try:
                        with open(path, 'rb') as fh:
                            b = fh.read()
                        return base64.b64encode(b).decode('ascii')
                    except Exception as e:
                        print(f"Could not read {path}: {e}")
                        return None
        return None

    demo_b64 = _get_demo_b64_from_mp3()
    if demo_b64 is not None:
        print("Using .mp3 file from data for demo prediction.")
    else:
        print("No .mp3 found in data; falling back to embedded WAV base64 demo.")
        demo_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="

    try:
        label, confidence, probs = predict_from_b64(model, demo_b64)
        print(f"Demo base64 -> Predicted: {label}, Confidence: {confidence:.2f}, Probs: {probs}")
    except Exception as e:
        print(f"Demo prediction failed: {e}")

    # Optional: run batch file predictions if data is present
    if os.path.isdir(DATA_DIR):
        print("\n=== Batch Prediction Results ===\n")
        for folder, expected_label in [("human", "HUMAN"), ("ai", "AI_GENERATED")]:
            class_dir = os.path.join(DATA_DIR, folder)

            print(f"[Folder: {folder}]")

            if not os.path.isdir(class_dir):
                print(f"  Folder missing: {class_dir}")
                continue

            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(".wav"):
                    continue

                wav_path = os.path.join(class_dir, fname)

                try:
                    feats = extract_features(wav_path).reshape(1, -1)
                    probs = model.predict_proba(feats)[0]
                    pred = int(np.argmax(probs))
                    pred_label = "AI_GENERATED" if pred == 1 else "HUMAN"
                    confidence = float(probs[pred])

                    print(
                        f"{fname:20s} | "
                        f"Predicted: {pred_label:12s} | "
                        f"Confidence: {confidence:.2f} | "
                        f"Expected: {expected_label}"
                    )

                except Exception as e:
                    print(f"{fname:20s} | ERROR: {e}")

            print()


if __name__ == "__main__":
    main()