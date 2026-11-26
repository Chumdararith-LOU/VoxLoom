"""Single-take and Batch Gap-Fill voice cloning endpoints + ZIP export."""

from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..audio_processing import postprocess_clip
from ..config import get_outputs_dir
from ..models_manager import engine_manager
from ..profiles import get_profile

router = APIRouter(prefix="/api/generate", tags=["generate"])


class PostProcessOptions(BaseModel):
    trim_silence: bool = True
    pad_sec: float = Field(default=0.05, ge=0.0, le=0.5)
    loudnorm: bool = True
    target_lufs: float = Field(default=-20.0, ge=-30.0, le=-10.0)
    sample_rate: int = Field(default=48000)
    channels: int = Field(default=1, ge=1, le=2)
    max_atempo: float = Field(default=1.08, ge=1.0, le=1.25)


class SingleGenerateRequest(BaseModel):
    profile_id: str
    text: str = Field(min_length=1)
    model_id: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=1.5)
    temperature: float = Field(default=0.7, ge=0.1, le=1.5)
    language: str = "en"
    postprocess: PostProcessOptions = Field(default_factory=PostProcessOptions)


class BatchGapItem(BaseModel):
    filename: str = Field(min_length=1)
    timeline_start: str = ""
    max_duration: float | None = Field(default=None, gt=0.2)
    text: str = Field(min_length=1)


class BatchGenerateRequest(BaseModel):
    profile_id: str
    items: list[BatchGapItem] = Field(min_length=1)
    model_id: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=1.5)
    temperature: float = Field(default=0.7, ge=0.1, le=1.5)
    language: str = "en"
    postprocess: PostProcessOptions = Field(default_factory=PostProcessOptions)


def _safe_filename(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)
    if not clean.lower().endswith(".wav"):
        clean = f"{clean}.wav"
    return clean


@router.post("/single")
def generate_single(req: SingleGenerateRequest) -> dict:
    try:
        profile = get_profile(req.profile_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    job_id = f"take_{uuid.uuid4().hex[:10]}"
    job_dir = get_outputs_dir() / job_id
    raw_wav = job_dir / "raw" / "take.wav"
    final_wav = job_dir / "take.wav"

    try:
        engine = engine_manager.get_loaded_tts(req.model_id)
        engine.generate(
            text=req.text.strip(),
            ref_audio=Path(profile["ref_audio_path"]),
            ref_text=profile["ref_text"],
            output_wav=raw_wav,
            speed=req.speed,
            temperature=req.temperature,
            language=req.language,
        )
        pp = postprocess_clip(
            raw_wav,
            final_wav,
            trim_silence=req.postprocess.trim_silence,
            pad_sec=req.postprocess.pad_sec,
            loudnorm=req.postprocess.loudnorm,
            target_lufs=req.postprocess.target_lufs,
            sample_rate=req.postprocess.sample_rate,
            channels=req.postprocess.channels,
            max_atempo=req.postprocess.max_atempo,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "job_id": job_id,
        "filename": "take.wav",
        "audio_url": f"/api/generate/outputs/{job_id}/take.wav",
        "result": pp.to_dict(),
    }


@router.post("/batch")
async def generate_batch_stream(req: BatchGenerateRequest) -> StreamingResponse:
    """Load TTS model once and stream SSE progress as each gap clip is generated & normalized."""
    try:
        profile = get_profile(req.profile_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    batch_id = f"batch_{uuid.uuid4().hex[:10]}"
    batch_dir = get_outputs_dir() / batch_id
    raw_dir = batch_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    q: queue.Queue[dict | None] = queue.Queue()

    def worker() -> None:
        try:
            q.put({"type": "status", "batch_id": batch_id, "message": "Ensuring TTS model is loaded in memory..."})
            engine = engine_manager.get_loaded_tts(req.model_id)
            ref_audio = Path(profile["ref_audio_path"])
            ref_text = profile["ref_text"]

            results = []
            total = len(req.items)
            for idx, item in enumerate(req.items, start=1):
                fname = _safe_filename(item.filename)
                q.put({
                    "type": "item_start",
                    "index": idx,
                    "total": total,
                    "filename": fname,
                    "timeline_start": item.timeline_start,
                })

                raw_wav = raw_dir / fname
                final_wav = batch_dir / fname

                engine.generate(
                    text=item.text.strip(),
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    output_wav=raw_wav,
                    speed=req.speed,
                    temperature=req.temperature,
                    language=req.language,
                )

                pp = postprocess_clip(
                    raw_wav,
                    final_wav,
                    trim_silence=req.postprocess.trim_silence,
                    pad_sec=req.postprocess.pad_sec,
                    max_duration=item.max_duration,
                    max_atempo=req.postprocess.max_atempo,
                    loudnorm=req.postprocess.loudnorm,
                    target_lufs=req.postprocess.target_lufs,
                    sample_rate=req.postprocess.sample_rate,
                    channels=req.postprocess.channels,
                )

                # If duration still exceeded max at 1.08x, automatically regenerate once per requirement
                regenerated = False
                if pp.needs_regeneration:
                    regenerated = True
                    q.put({
                        "type": "item_retry",
                        "index": idx,
                        "filename": fname,
                        "message": f"{fname} exceeded max duration even at {req.postprocess.max_atempo:.2f}x; regenerating once...",
                    })
                    engine.generate(
                        text=item.text.strip(),
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        output_wav=raw_wav,
                        speed=min(1.5, req.speed * 1.05),
                        temperature=req.temperature,
                        language=req.language,
                    )
                    pp = postprocess_clip(
                        raw_wav,
                        final_wav,
                        trim_silence=req.postprocess.trim_silence,
                        pad_sec=req.postprocess.pad_sec,
                        max_duration=item.max_duration,
                        max_atempo=req.postprocess.max_atempo,
                        loudnorm=req.postprocess.loudnorm,
                        target_lufs=req.postprocess.target_lufs,
                        sample_rate=req.postprocess.sample_rate,
                        channels=req.postprocess.channels,
                    )

                row = {
                    "index": idx,
                    "filename": fname,
                    "timeline_start": item.timeline_start,
                    "text": item.text,
                    "max_duration": item.max_duration,
                    "actual_duration": pp.final_duration,
                    "trimmed_duration": pp.trimmed_duration,
                    "atempo_applied": pp.atempo_applied,
                    "status": pp.status,
                    "regenerated": regenerated,
                    "audio_url": f"/api/generate/outputs/{batch_id}/{fname}",
                }
                results.append(row)
                q.put({"type": "item_done", "item": row})

            q.put({
                "type": "complete",
                "batch_id": batch_id,
                "zip_url": f"/api/generate/outputs/{batch_id}.zip",
                "output_dir": str(batch_dir),
                "results": results,
            })
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_generator():
        while True:
            evt = await asyncio.to_thread(q.get)
            if evt is None:
                break
            yield f"data: {json.dumps(evt)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/outputs/{job_id}.zip")
def download_batch_zip(job_id: str) -> FileResponse:
    safe_job = Path(job_id).name
    batch_dir = get_outputs_dir() / safe_job
    if not batch_dir.is_dir():
        raise HTTPException(status_code=404, detail="Batch output directory not found")

    zip_path = get_outputs_dir() / f"{safe_job}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for wav in sorted(batch_dir.glob("*.wav")):
            zf.write(wav, arcname=wav.name)

    return FileResponse(zip_path, media_type="application/zip", filename=f"{safe_job}.zip")


@router.get("/outputs/{job_id}/{filename}")
def download_output_audio(job_id: str, filename: str) -> FileResponse:
    safe_job = Path(job_id).name
    safe_name = Path(filename).name
    path = get_outputs_dir() / safe_job / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Generated audio file not found")
    return FileResponse(path, media_type="audio/wav", filename=safe_name)
