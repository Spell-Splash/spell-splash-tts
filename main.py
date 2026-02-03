from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from tts_engine import TTSEngine
from contextlib import asynccontextmanager

# Global Variable สำหรับเก็บ Engine
tts_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_engine
    
    # 1. ปริ้นท์ Dashboard Link ก่อนเลย
    print("\n" + "="*60)
    print("✅  TTS Service Ready!")
    print("🔗  Open Docs: http://localhost:5001/docs")
    print("="*60 + "\n")

    try:
        # 2. เริ่มโหลดโมเดล
        print("⏳ Loading Kokoro TTS Model...")
        tts_engine = TTSEngine()
        print("✅ Kokoro TTS Loaded! Ready to speak.")
        yield
    except Exception as e:
        print(f"❌ Failed to load TTS Engine: {e}")
        # ถ้าโหลดไม่ผ่าน Server อาจจะรันไม่ขึ้น หรือควร Handle error
        raise e
    finally:
        print("🛑 Shutting down TTS Service...")

app = FastAPI(title="Spell Splash TTS Service", lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "spell-splash-tts", "model": "Kokoro v0.19 (ONNX)"}

@app.get("/tts")
async def text_to_speech(
    text: str = Query(..., description="ข้อความที่ต้องการให้พูด"),
    voice: str = Query("af_bella", description="เสียงที่ต้องการ (เช่น af_bella, af_sarah, am_michael)"),
    speed: float = Query(1.0, description="ความเร็วเสียง")
):
    """
    Generate Audio from Text using Kokoro
    """
    if not tts_engine:
        raise HTTPException(status_code=503, detail="TTS Engine is not ready")

    try:
        # เรียก Engine สร้างเสียง
        audio_buffer = tts_engine.generate_audio_bytes(text, voice, speed)
        
        # ส่งไฟล์กลับไป (Streaming)
        return StreamingResponse(
            audio_buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": f"inline; filename={text}.wav"}
        )

    except Exception as e:
        print(f"❌ Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)