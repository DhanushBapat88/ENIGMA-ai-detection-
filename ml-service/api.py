from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from pydantic import BaseModel,Field, field_validator
import base64
import os
import tempfile

import os

from audio_pipeline import process_audio_pipeline
from test_predict import predict_audio

# ✅ CREATE APP FIRST
app = FastAPI(title="AI Generated Voice Detection API")

# GUVI key
API_KEY = "sk_guvi_voice_2026"

# -------- Authorization Dependency --------
def verify_api_key(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing"
        )

    # Accept both:
    # "Bearer <API_KEY>" and "<API_KEY>"
    token = authorization.replace("Bearer ", "").strip()

    if token != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return token


# -------- Request Schema --------
from pydantic import BaseModel, Field, field_validator
from typing import Literal
import base64

from pydantic import BaseModel, Field, field_validator
from typing import Literal
import base64
import binascii


class AudioRequest(BaseModel):
    language: Literal[
        "tamil",
        "english",
        "malayalam",
        "hindi",
        "telugu"
    ] = Field(..., description="Supported languages only")

    audio_format: Literal["mp3"] = Field(
        ..., description="Only mp3 format is supported"
    )

    audio_base64: str = Field(
        ..., description="Base64-encoded MP3 audio (single line)"
    )

    # ✅ Proper base64 + MP3 validation
    @field_validator("audio_base64")
    @classmethod
    def validate_base64_mp3(cls, value: str):
        try:
            # Remove whitespace / accidental newlines
            # handling new line and spaces here
            value = value.strip().replace("\n", "").replace("\r", "")
            decoded = base64.b64decode(value)
        except binascii.Error:
            raise ValueError("audio_base64 is not valid base64")

        # 🔍 MP3 signature check
        # ID3 → tagged MP3
        # FF FB / FF F3 / FF F2 → raw MP3 frames
        if not (
            decoded.startswith(b"ID3") or
            decoded[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
        ):
            raise ValueError("audio_base64 does not contain MP3 audio")

        return value



@app.post("/process")
def process_audio(
    payload: AudioRequest,
    x_api_key: str = Header(..., alias="x-api-key")
):
    # 🔐 API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 🔍 Validate audio format (optional but good)
    if payload.audio_format.lower() not in ["mp3", "wav", "m4a"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # 🔁 Decode base64 → bytes
    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    # 🔁 Re-encode to clean base64 (single-line, safe)
    audio_base64_clean = base64.b64encode(audio_bytes).decode("utf-8")

    # 🎧 Run pipeline
    waveform = process_audio_pipeline(audio_base64_clean)

    if waveform is None:
        raise HTTPException(status_code=500, detail="Audio processing failed")

    # 🤖 Predict
    sr = 16000

    result = predict_audio(waveform, sr)

    return {
        "status": "success",
        "language": payload.language,
        "prediction": result["label"],
        "confidenceScore": round(result["confidence"], 3),
        "explanation": result["explanation"]
    }




@app.post("/encode-audio")
def encode_audio(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Upload mp3, wav, m4a, or flac."
        )

    try:
        # Read file bytes
        audio_bytes = file.file.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Convert to base64 (single line)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        file.file.close()

    return {
        "audio_base64": audio_base64
    }
