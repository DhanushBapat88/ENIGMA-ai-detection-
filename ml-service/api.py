from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import base64
import tempfile
import os

from audio_pipeline import process_audio_pipeline
from test_predict import predict_audio

app = FastAPI(title="AI Generated Voice Detection API")

API_KEY = "Qkef6vU90hKcU2Fvsaa2"

# -------- Request Schema --------
class AudioRequest(BaseModel):
    message: Optional[str] = None
    audio_url: str

# -------- Health check --------
@app.get("/")
def home():
    return {"status": "API running"}

# -------- API Endpoint --------
@app.post("/process")
def process_audio(
    payload: AudioRequest,
    authorization: str = Header(None)
):
    # Auth check
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Authorization")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization")

    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Download audio
    try:
        r = requests.get(payload.audio_url, timeout=15)
        r.raise_for_status()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to download audio")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(r.content)
        temp_path = tmp.name

    try:
        # Convert to base64
        with open(temp_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Run pipeline
        waveform = process_audio_pipeline(audio_base64)
        if waveform is None:
            raise HTTPException(status_code=500, detail="Audio processing failed")

        sr = 16000

        # Predict
        label, confidence = predict_audio(waveform, sr)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_path)

    return {
        "status": "success",
        "prediction": label,
        "confidence": round(float(confidence), 3),
        "sample_rate": sr
    }
