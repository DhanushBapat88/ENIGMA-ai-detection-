import base64
import io
import numpy as np
import librosa
from pydub import AudioSegment


def process_audio_pipeline(b64_string, sr=16000, duration=4.0):
    try:
        # ---------- STAGE 1: Decode base64 ----------
        audio_bytes = base64.b64decode(b64_string)

        # ---------- STAGE 2: Load ANY format using pydub ----------
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

        # ---------- STAGE 3: Convert to mono + target rate ----------
        audio = audio.set_frame_rate(sr).set_channels(1)

        # ---------- STAGE 4: Export to wav buffer ----------
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)

        # ---------- STAGE 5: Load with librosa ----------
        y, sr = librosa.load(buffer, sr=sr)

        # ---------- STAGE 6: Fix duration ----------
        target_length = int(sr * duration)

        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)))

        return y

    except Exception as e:
        print("Audio pipeline error:", e)
        return None