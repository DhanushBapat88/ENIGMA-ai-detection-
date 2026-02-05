# ENIGMA - Voice AI Detector 🔊🤖

ENIGMA is a production‑style AI vs Human Voice Detection System that demonstrates a full pipeline from audio ingestion to machine‑learning prediction using FastAPI, an audio preprocessing pipeline, and an automated n8n workflow.

Repository layout:

├── n8n/
│   └── workflow.json          # n8n webhook → FastAPI integration
│
├── ml-service/
│   ├── api.py                 # FastAPI prediction service
│   ├── audio_pipeline.py      # Base64 → waveform processing
│   ├── feature_extractor.py   # Audio feature extraction
│   ├── train_model.py         # RandomForest training script
│   ├── test_pipeline.py       # Audio pipeline validation
│   ├── test_predict.py        # Batch prediction script
│   ├── model.pkl              # Trained ML model
│   └── data/
│       ├── human/             # Human voice samples
│       └── ai/                # AI‑generated voice samples
│
└── README.md

Overview:

This project simulates a real‑world architecture where voice audio is received via webhook automation, processed through an audio normalization pipeline, transformed into numerical features, and classified using a trained RandomForest model.

Core Goals

Detect whether a voice is Human or AI Generated

Provide a secure API with authentication

Demonstrate scalable ML + Backend integration

Support automation workflows (n8n)



System Architecture:

Client / Tester / n8n
        │
        ▼
FastAPI (/process)
        │
        ▼
Base64 Decode
        │
        ▼
Audio Pipeline (pydub + librosa)
        │
        ▼
Feature Extraction
        │
        ▼
RandomForest Model
        │
        ▼
Prediction + Confidence


Replace `model.pkl` with your trained model and implement real feature extraction.

---

## ▶️ Quick start (local)
1. Install Python packages (recommended virtualenv):

   ```bash
   pip install numpy librosa scikit-learn pydub soundfile
   ```

2. Install FFmpeg and ensure it is on your PATH (required by `pydub`). See https://ffmpeg.org/.

3. Generate a model (if none exists):

   ```bash
   python ml-service/train_model.py
   ```

4. Run the demo prediction (uses an `.mp3` in `ml-service/data/` if present; otherwise falls back to a tiny WAV base64):

   ```bash
   python ml-service/test_predict.py
   ```

---

## 🧪 Testing & expected behavior
- `test_pipeline.py` verifies Base64 decoding → resample → trim → pad and prints array shape/duration.
- `test_predict.py` performs an in-memory prediction: it will look for `.mp3` in `ml-service/data/`, encode it to base64, run it through the pipeline + feature extractor, and return a prediction using `model.pkl`.

Notes:
- Feature extraction expects >= 1s input; pipeline pads to 3s by default.
- `model.pkl` must be present (run training if missing).
---

## 🚧 Known issues & TODO (prioritized)
1. Data collection: **Collect 200 human + 200 AI** voice samples. (Current: ~2/2) — Highest priority
2. Add evaluation: train/validation split, metrics (accuracy, precision, recall, ROC AUC), and threshold selection in `train_model.py` — Important
3. Implement API server (FastAPI recommended) with `/detect-voice` endpoint, JSON responses, and proper error handling — Important
4. Implement API key authentication server-side (workflow currently expects it) — Important
5. Add `requirements.txt`, CI (pytest), and GitHub Actions to run tests — Medium
6. Improve explainability and README documentation for decisions & thresholds — Medium

---

## 👥 Contributors
- Member 1 — ML model, feature extractor: **(status: feature extraction & RF training done; evaluation missing)**
- Member 2 — Audio pipeline: **(status: pipeline implemented; trimming & padding done; integration to API pending)**
- Member 3 — Backend/API: **(status: workflow references API, server missing)**
- Member 4 — Integration & docs: **(status: partial; README cleaned; explanation pending)**



---

## 🤝 How to help / contribute
- Add more labeled audio files under `ml-service/data/human` and `ml-service/data/ai` (prefer `.wav` or `.mp3`) and name them consistently.
- Implement an API in `ml-service/` (FastAPI suggested) with an authenticated `/detect-voice` endpoint.
- Add proper train/validation code and a script to output evaluation metrics and a chosen decision threshold.
- Create `requirements.txt` and add CI with `pytest`.

---
hello, to work on just put the command
pip install fastapi uvicorn
python -m uvicorn api:app --reload --port 5000
u will get to see:
Uvicorn running on http://127.0.0.1:5000
Application startup complete
and next, go to browser and test
http://localhost:5000/docs
Swagger UI will open 
Next go to  Postman
there  POST
http://localhost:5678/webhook-test/process-audio
and next select body 
in that raw ->JSON->there u need to insert
{
  "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
  "message": "testing"
}
then press send u will get the output:
{
    "status": "success",
    "prediction": "HUMAN",
    "confidence": 0.5,
    "sample_rate": 16000
}


