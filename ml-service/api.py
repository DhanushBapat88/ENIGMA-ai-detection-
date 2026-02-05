from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import os

from audio_pipeline import process_audio_pipeline
from test_predict import predict_audio

app = FastAPI(title="AI Generated Voice Detection API")

# GUVI key
API_KEY = "sk_guvi_voice_2026"

# -------- Request Schema --------
class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str


# -------- API Endpoint --------
@app.post("/process")
def process_audio(
    payload: AudioRequest,
    x_api_key: str = Header(None)
):
    # 🔐 API KEY CHECK
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API Key")

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 🔊 Decode base64 audio
    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    # 💾 Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    try:
        # 🎧 Run pipeline
        with open(temp_path, "rb") as f:
            b64_audio = base64.b64encode(f.read()).decode()

        waveform = process_audio_pipeline(b64_audio)

        if waveform is None:
            raise HTTPException(status_code=500, detail="Audio processing failed")

        sr = 16000

        # 🤖 Predict
        label, confidence = predict_audio(waveform, sr)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.remove(temp_path)

    # ✅ Response (GUVI format)
    return {
        "status": "success",
        "prediction": label,
        "confidence": round(confidence, 3)
    }
