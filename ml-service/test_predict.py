import joblib
import numpy as np
from feature_extractor import extract_features_from_array

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")

# ---------- PREDICTION FUNCTION ----------
def predict_audio(waveform, sr):
    try:
        # 1. Feature extraction
        features = extract_features_from_array(waveform, sr)

        # 2. Reshape for model
        X = features.reshape(1, -1)

        # 3. Predict
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        label = "AI_GENERATED" if int(pred) == 1 else "HUMAN"
        confidence = float(np.max(proba))

        # 4. Explanation (honest & defensible)
        if label == "AI_GENERATED":
            explanation = (
                "Unnatural pitch consistency and robotic speech "
                "pattern detected" 
            )
        else:
            explanation = (
                "The audio shows natural pitch variation, energy fluctuations, "
                "and spectral features typically found in human speech."
            )

        return {
            "label": label,
            "confidence": confidence,
            "explanation": explanation
        }

    except Exception as e:
        return {
            "label": "ERROR",
            "confidence": 0.0,
            "explanation": f"Prediction failed due to internal error: {str(e)}"
        }
