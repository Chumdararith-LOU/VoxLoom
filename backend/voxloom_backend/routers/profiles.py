"""Voice profile upload, waveform analysis, auto-segment selection, transcription, and CRUD."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..audio_processing import (
    auto_pick_cleanest_segment,
    extract_waveform_peaks,
    probe_audio,
    slice_reference_wav,
)
from ..config import get_uploads_dir
from ..models_manager import engine_manager
from ..profiles import (
    create_profile_from_slice,
    delete_profile,
    get_profile,
    list_profiles,
    update_profile,
)

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


class TranscribeSliceRequest(BaseModel):
    upload_id: str
    start_sec: float = Field(ge=0.0)
    duration_sec: float = Field(gt=0.5, le=60.0)
    language: str | None = None
    stt_model_id: str | None = None


class CreateProfileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    upload_id: str
    start_sec: float = Field(ge=0.0)
    duration_sec: float = Field(gt=0.5, le=60.0)
    ref_text: str = Field(min_length=1)
    language: str = "en"
    notes: str = ""


class UpdateProfileRequest(BaseModel):
    name: str | None = None
    ref_text: str | None = None
    notes: str | None = None


def _resolve_upload(upload_id: str) -> Path:
    # Sanitize upload_id to prevent path traversal
    safe_id = Path(upload_id).name
    matches = list(get_uploads_dir().glob(f"{safe_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Uploaded audio not found: {safe_id}")
    return matches[0]


@router.get("")
def get_all_profiles() -> dict:
    return {"profiles": list_profiles()}


@router.post("/upload")
async def upload_source_audio(file: UploadFile = File(...)) -> dict:
    """Save uploaded audio/video, extract waveform peaks, and suggest cleanest 10-14s slice."""
    suffix = Path(file.filename or "recording.wav").suffix.lower() or ".wav"
    if suffix not in {".wav", ".m4a", ".mp3", ".ogg", ".flac", ".aac", ".mp4", ".mov", ".webm"}:
        raise HTTPException(status_code=400, detail=f"Unsupported media format: {suffix}")

    upload_id = f"src_{uuid.uuid4().hex[:10]}"
    dest = get_uploads_dir() / f"{upload_id}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        meta = probe_audio(dest)
        waveform = extract_waveform_peaks(dest, num_bars=240)
        suggestion = auto_pick_cleanest_segment(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Failed to analyze media: {exc}") from exc

    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "metadata": meta.to_dict(),
        "waveform": waveform,
        "suggestion": suggestion.to_dict(),
    }


@router.get("/upload/{upload_id}/audio")
def stream_uploaded_audio(upload_id: str) -> FileResponse:
    path = _resolve_upload(upload_id)
    return FileResponse(path)


@router.post("/transcribe-slice")
def transcribe_slice(req: TranscribeSliceRequest) -> dict:
    """Slice temporary 24kHz ref WAV and transcribe locally with Whisper."""
    src = _resolve_upload(req.upload_id)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        slice_reference_wav(src, req.start_sec, req.duration_sec, tmp_path)
        text = engine_manager.transcribe_reference(
            tmp_path,
            model_id=req.stt_model_id,
            language=req.language,
        )
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("")
def create_profile(req: CreateProfileRequest) -> dict:
    src = _resolve_upload(req.upload_id)
    try:
        profile = create_profile_from_slice(
            name=req.name,
            source_audio=src,
            start_sec=req.start_sec,
            duration_sec=req.duration_sec,
            ref_text=req.ref_text,
            language=req.language,
            notes=req.notes,
        )
        return {"profile": profile}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.patch("/{profile_id}")
def patch_profile(profile_id: str, req: UpdateProfileRequest) -> dict:
    try:
        profile = update_profile(profile_id, name=req.name, ref_text=req.ref_text, notes=req.notes)
        return {"profile": profile}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{profile_id}")
def remove_profile(profile_id: str) -> dict:
    try:
        delete_profile(profile_id)
        return {"deleted": profile_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{profile_id}/audio")
def get_profile_audio(profile_id: str) -> FileResponse:
    try:
        profile = get_profile(profile_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(profile["ref_audio_path"], media_type="audio/wav", filename="ref.wav")
