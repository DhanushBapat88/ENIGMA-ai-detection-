# import os
# import pickle
# import numpy as np

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# from feature_extractor import extract_features

# BASE_DIR = os.path.dirname(__file__)
# DATA_DIR = os.path.join(BASE_DIR, "data")
# MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# def load_dataset():
#     X = []
#     y = []

#     for label_name, label_value in [("human", 0), ("ai", 1)]:
#         class_dir = os.path.join(DATA_DIR, label_name)

#         if not os.path.isdir(class_dir):
#             raise RuntimeError(f"Missing folder: {class_dir}")

#         for fname in os.listdir(class_dir):
#             if not fname.lower().endswith(".wav"):
#                 continue

#             wav_path = os.path.join(class_dir, fname)

#             try:
#                 features = extract_features(wav_path)
#                 X.append(features)
#                 y.append(label_value)
#             except Exception as e:
#                 print(f"[WARN] Skipping {wav_path}: {e}")

#     if len(X) == 0:
#         raise RuntimeError("No training data found")

#     return np.array(X), np.array(y)

# def main():
#     print("[INFO] Loading dataset...")
#     X, y = load_dataset()

#     print(f"[INFO] Samples: {len(X)}, Feature dim: {X.shape[1]}")

#     X_train, y_train = X, y

#     model = RandomForestClassifier(
#         n_estimators=200,
#         random_state=42,
#         n_jobs=-1
#     )

#     print("[INFO] Training model...")
#     model.fit(X_train, y_train)

#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump(model, f)

#     print(f"[INFO] Model saved to {MODEL_PATH}")

# if __name__ == "__main__":
#     main()

import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from feature_extractor import extract_features_from_file

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def load_dataset():
    X = []
    y = []

    for label_name, label_value in [("human", 0), ("ai", 1)]:
        class_dir = os.path.join(DATA_DIR, label_name)

        if not os.path.isdir(class_dir):
            raise RuntimeError(f"Missing folder: {class_dir}")

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(".wav"):
                continue

            wav_path = os.path.join(class_dir, fname)

            try:
                features = extract_features_from_file(wav_path)
                X.append(features)
                y.append(label_value)
            except Exception as e:
                print(f"[WARN] Skipping {wav_path}: {e}")

    if len(X) == 0:
        raise RuntimeError("No training data found")

    return np.array(X), np.array(y)


def main():
    print("[INFO] Loading dataset...")
    X, y = load_dataset()

    print(f"[INFO] Samples: {len(X)}, Feature dim: {X.shape[1]}")

    # Using full dataset (no split for now)
    X_train, y_train = X, y

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"[INFO] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()