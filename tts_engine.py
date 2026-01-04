import io
import soundfile as sf
from kokoro_onnx import Kokoro
import os

class TTSEngine:
    def __init__(self):
        print("⏳ Loading Kokoro TTS Model...")
        
        model_path = "models/kokoro-v1.0.onnx"
        voices_path = "models/voices-v1.0.bin"
        
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            raise FileNotFoundError("❌ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบโฟลเดอร์ /models")

        # โหลดโมเดล (รันครั้งแรกจะช้านิดนึง)
        self.kokoro = Kokoro(model_path, voices_path)
        print("✅ Kokoro TTS Loaded! Ready to speak.")

    def generate_audio_bytes(self, text: str, voice: str = "af_bella", speed: float = 1.0):
        """
        แปลง Text -> Audio Bytes (WAV format)
        """
        # สร้างเสียง (Return เป็น numpy array และ sample rate)
        samples, sample_rate = self.kokoro.create(
            text, 
            voice=voice, 
            speed=speed, 
            lang="en-us"
        )

        # แปลง Numpy Array -> WAV File ใน Memory (BytesIO)
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format='WAV')
        buffer.seek(0) # รีเซ็ตเข็มอ่านไปที่จุดเริ่มต้น
        
        return buffer