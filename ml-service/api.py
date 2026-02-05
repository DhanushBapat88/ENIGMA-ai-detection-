# from fastapi import FastAPI, Header, HTTPException
# from pydantic import BaseModel
# import requests
# import base64
# import tempfile
# import os

# from audio_pipeline import process_audio_pipeline
# from test_predict import predict_audio

# app = FastAPI(title="AI Generated Voice Detection API")

# API_KEY = "Qkef6vU90hKcU2Fvsaa2"


# # -------- Request Schema --------
# from pydantic import BaseModel
# from typing import Optional

# class AudioRequest(BaseModel):
#     message: Optional[str] = None
#     audio_url: str

# # -------- API Endpoint --------
# @app.post("/process")
# def process_audio(
#     payload: AudioRequest,
#     authorization: str = Header(None)
# ):
#     # 1️⃣ Auth check
#     if authorization is None:
#         raise HTTPException(status_code=401, detail="Missing Authorization")

#     token = authorization.replace("Bearer ", "")
#     if token != API_KEY:
#         raise HTTPException(status_code=401, detail="Invalid API Key")

#     # 2️⃣ Download audio
#     try:
#         r = requests.get(payload.audio_url, timeout=10)
#         r.raise_for_status()
#     except Exception:
#         raise HTTPException(status_code=400, detail="Failed to download audio")

#     # 3️⃣ Save temp audio
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
#         tmp.write(r.content)
#         temp_path = tmp.name

#     try:
#         # 4️⃣ Convert to base64
#         with open(temp_path, "rb") as f:
#             audio_base64 = base64.b64encode(f.read()).decode("utf-8")

#         # 5️⃣ Run pipeline
#         waveform, sr = process_audio_pipeline(audio_base64)

#         # 6️⃣ Predict
#         label, confidence = predict_audio(waveform, sr)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         os.remove(temp_path)

#     # 7️⃣ Response (tester-friendly)
#     return {
#         "status": "success",
#         "prediction": label,
#         "confidence": round(confidence, 3),
#         "sample_rate": sr
#     }

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


# -------- API Endpoint --------
@app.post("/process")
def process_audio(
    payload: AudioRequest,
    authorization: str = Header(None)
):
    # 1️⃣ Authorization check
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Authorization")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization")

    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2️⃣ Download audio
    try:
        r = requests.get(payload.audio_url, timeout=10)
        r.raise_for_status()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to download audio")

    # 3️⃣ Save temp audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(r.content)
        temp_path = tmp.name

    try:
        # 4️⃣ Convert audio to base64
        with open(temp_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 5️⃣ Run audio pipeline (RETURNS ONLY y)
        waveform = process_audio_pipeline(audio_base64)

        if waveform is None:
            raise HTTPException(status_code=500, detail="Audio processing failed")

        sr = 16000  # fixed by pipeline

        # 6️⃣ Predict
        label, confidence = predict_audio(waveform, sr)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.remove(temp_path)

    # 7️⃣ Tester-friendly response
    return {
        "status": "success",
        "prediction": label,
        "confidence": round(confidence, 3),
        "sample_rate": sr
    }
