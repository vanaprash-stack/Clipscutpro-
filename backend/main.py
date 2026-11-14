import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / "work"
OUTPUT_DIR = APP_DIR / "outputs"
WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Clipscutpro API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500",
                   "http://127.0.0.1:8000", "http://localhost:8000",
                   "null"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

MAX_BYTES = 500 * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
_model = None

def run_cmd(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout
    except FileNotFoundError:
        raise HTTPException(500, "FFmpeg/ffprobe was not found. Install FFmpeg and add it to PATH.")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "Media command failed")[-1800:]
        raise HTTPException(500, detail)

def get_duration(path: Path) -> float:
    out = run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]).strip()
    try:
        return float(out)
    except ValueError:
        raise HTTPException(400, "Could not read the video's duration.")

def transcribe(path: Path):
    global _model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(500, "Whisper is not installed. Follow backend setup in README.txt.")
    if _model is None:
        # First run downloads the small model. CPU/int8 keeps the starter accessible.
        _model = WhisperModel(os.getenv("WHISPER_MODEL", "base"), device="cpu", compute_type="int8")
    segments, info = _model.transcribe(str(path), vad_filter=True, word_timestamps=False)
    rows = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text and seg.end > seg.start:
            rows.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return rows

HOOKS = {
    "important", "key", "secret", "mistake", "tip", "trick", "remember",
    "because", "finally", "best", "worst", "surprising", "problem",
    "solution", "why", "how", "never", "always", "lesson", "learn"
}

def choose_highlights(segments: list[dict], total: float, target: int, limit: int = 5):
    """Simple transcript-based baseline, not a trained semantic highlight model."""
    if not segments:
        return []
    candidates = []
    for i, seg in enumerate(segments):
        start = max(0.0, seg["start"] - 3.0)
        end = min(total, max(seg["end"] + 3.0, start + target))
        if end - start < min(12, target * 0.5):
            continue
        text = " ".join(s["text"] for s in segments
                        if s["start"] < end and s["end"] > start)
        words = re.findall(r"[a-z']+", text.lower())
        if not words:
            continue
        hook_count = sum(1 for w in words if w in HOOKS)
        punctuation = text.count("?") + text.count("!")
        # Prefer coherent speech windows and verbal emphasis; score is heuristic.
        score = min(len(words), 90) / 90 + hook_count * 0.16 + punctuation * 0.12
        candidates.append({"start": start, "end": end, "text": text, "score": score})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    for item in candidates:
        if all(item["end"] <= old["start"] or item["start"] >= old["end"] for old in selected):
            selected.append(item)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda x: x["start"])
    return selected

def render_clip(source: Path, output: Path, start: float, end: float, aspect: str):
    if aspect == "9:16":
        vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    elif aspect == "1:1":
        vf = "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080"
    else:
        vf = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080"
    run_cmd([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source),
        "-t", f"{end-start:.3f}", "-vf", vf, "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
        "-movflags", "+faststart", str(output)
    ])

@app.get("/api/health")
def health():
    return {"ok": True, "service": "Clipscutpro API"}

@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    clip_length: int = Form(45),
    aspect_ratio: str = Form("9:16"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "Unsupported file type. Use MP4, MOV, WEBM, or MKV.")
    if clip_length not in {30, 45, 60}:
        raise HTTPException(400, "Choose a supported clip length.")
    if aspect_ratio not in {"9:16", "1:1", "16:9"}:
        raise HTTPException(400, "Choose a supported aspect ratio.")

    job_id = uuid.uuid4().hex
    source = WORK_DIR / f"{job_id}{suffix}"
    size = 0
    try:
        with source.open("wb") as dest:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "Video is larger than the 500 MB starter limit.")
                dest.write(chunk)

        duration = get_duration(source)
        if duration < 8:
            raise HTTPException(400, "Please upload a video at least 8 seconds long.")
        transcript = transcribe(source)
        picks = choose_highlights(transcript, duration, clip_length)
        if not picks:
            raise HTTPException(422, "No speech highlights found. Try a video with clear spoken audio.")

        clips = []
        for idx, pick in enumerate(picks, 1):
            out_name = f"{job_id}_clip_{idx}.mp4"
            output = OUTPUT_DIR / out_name
            render_clip(source, output, pick["start"], pick["end"], aspect_ratio)
            snippet = re.sub(r"\s+", " ", pick["text"]).strip()
            title = (snippet[:58] + "…") if len(snippet) > 58 else snippet
            clips.append({
                "title": title or f"Clip {idx}",
                "start": round(pick["start"], 2),
                "end": round(pick["end"], 2),
                "duration": round(pick["end"] - pick["start"], 2),
                "download_url": f"/outputs/{out_name}",
            })
        return {"job_id": job_id, "clips": clips,
                "note": "Highlights are ranked using a basic transcript heuristic. Review clips before publishing."}
    finally:
        try:
            source.unlink(missing_ok=True)
        except Exception:
            pass
        await file.close()
