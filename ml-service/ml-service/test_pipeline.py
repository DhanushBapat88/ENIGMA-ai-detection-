import base64
from audio_pipeline import process_audio_pipeline  # Ensure your file is named audio_pipeline.py

def test_member_2():
    # 1. Create a tiny test Base64 string (This is a 1-second silent WAV)
    # You can also use a real base64 string from a file
    test_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="
    
    print("🚀 Starting Pipeline Test...")
    result = process_audio_pipeline(test_b64)
    
    if result is not None:
        print(f"✅ SUCCESS!")
        print(f"📊 Array Shape: {result.shape}")
        print(f"⏱️ Duration Check: {len(result)/16000:.1f} seconds")
    else:
        print("❌ FAILED: Check if FFmpeg is working and path is correct.")

if __name__ == "__main__":
    test_member_2()