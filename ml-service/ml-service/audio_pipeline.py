import base64
import io
import librosa
import numpy as np
from pydub import AudioSegment

def process_audio_pipeline(b64_string, sr=16000, duration=3.0):
    """
    MASTER PIPELINE (Member 2 Tasks)
    1. Base64 to Bytes
    2. Format Conversion & Resampling (Pydub)
    3. Trim Silence (Librosa)
    4. Pad/Crop to 3.0s (NumPy)
    """
    try:
        # --- STAGE 1: Decoding ---
        audio_bytes = base64.b64decode(b64_string)
        audio_io = io.BytesIO(audio_bytes)
        
        # --- STAGE 2: Standardization ---
        # Converts any format (mp3/ogg/aac) to 16kHz Mono WAV
        audio = AudioSegment.from_file(audio_io)
        audio = audio.set_frame_rate(sr).set_channels(1)
        
        # Export to buffer to read into librosa
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        
        # Load as NumPy array
        y, _ = librosa.load(buffer, sr=sr)

        # --- STAGE 3: Cleaning (The code you just shared) ---
        # 1. Trim Silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # 2. Fix Duration (Padding or Cropping)
        target_length = int(sr * duration)
        if len(y_trimmed) > target_length:
            y_final = y_trimmed[:target_length]
        else:
            y_final = np.pad(y_trimmed, (0, target_length - len(y_trimmed)), mode='constant')
            
        print("✅ Pipeline Complete: Audio is standardized and ready for ML model.")
        return y_final

    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        return None
    