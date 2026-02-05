import base64
import requests

with open("test.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "language": "en",
    "audio_format": "wav",
    "audio_base64": audio_b64
}

r = requests.post("http://127.0.0.1:8000/process", json=payload)
print(r.status_code)
print(r.json())