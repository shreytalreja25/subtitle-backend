# FastAPI-based API for subtitle generation using Whisper (no OpenAI)
# Save this as: main.py

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import whisper
import srt
import re
from datetime import timedelta
import moviepy as mp
import os

app = FastAPI()

# Enable CORS for frontend (v0.dev or localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Subtitle Processing Pipeline ---

def extract_audio(video_path: str, audio_path: str):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def transcribe_audio(audio_path: str, model_size: str = "base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=False)
    return result['segments']

def simple_cleanup(text: str):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if not text.endswith((".", "?", "!")):
        text += "."
    return text[0].upper() + text[1:] if text else text

def generate_srt(segments, output_file="subtitles.srt"):
    subtitles = []
    for i, seg in enumerate(segments):
        start = timedelta(seconds=seg["start"])
        end = timedelta(seconds=seg["end"])
        text = simple_cleanup(seg["text"])
        subtitles.append(srt.Subtitle(index=i + 1, start=start, end=end, content=text))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))


# --- API Route ---

@app.post("/generate-subtitles/")
async def generate_subtitles(
    file: UploadFile = File(...),
    model_size: str = Form("base")
):
    video_path = "input_video.mp4"
    audio_path = "audio.wav"
    srt_output = "subtitles.srt"

    # Save uploaded video
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extract_audio(video_path, audio_path)
        segments = transcribe_audio(audio_path, model_size)
        generate_srt(segments, srt_output)
        return FileResponse(srt_output, media_type="text/plain", filename="subtitles.srt")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for path in [video_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)

# --- Run the server (only for local testing) ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

