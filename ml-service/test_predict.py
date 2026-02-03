import os
import pickle
import numpy as np

from feature_extractor import extract_features

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data")

def predict_file(model, wav_path):
    features = extract_features(wav_path).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    pred = int(np.argmax(probs))

    label = "AI_GENERATED" if pred == 1 else "HUMAN"
    confidence = float(probs[pred])

    return label, confidence, probs

def main():
    # Load trained model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("\n=== Batch Prediction Results ===\n")

    for folder, expected_label in [("human", "HUMAN"), ("ai", "AI_GENERATED")]:
        class_dir = os.path.join(DATA_DIR, folder)

        print(f"[Folder: {folder}]")

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(".wav"):
                continue

            wav_path = os.path.join(class_dir, fname)

            try:
                pred_label, confidence, probs = predict_file(model, wav_path)

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