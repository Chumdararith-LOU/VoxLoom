"""Local Voice Profile persistence (ref.wav + exact ref_text + metadata)."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .audio_processing import get_duration, slice_reference_wav
from .config import get_profiles_dir


@dataclass
class VoiceProfile:
    id: str
    name: str
    ref_text: str
    ref_audio_path: str
    duration_sec: float
    language: str
    created_at: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _index_file() -> Path:
    return get_profiles_dir() / "profiles.json"


def _load_index() -> list[dict]:
    idx = _index_file()
    if not idx.exists():
        return []
    try:
        return json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_index(items: list[dict]) -> None:
    idx = _index_file()
    idx.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def list_profiles() -> list[dict]:
    items = _load_index()
    # Filter out any profile whose ref.wav was deleted externally
    return [item for item in items if Path(item["ref_audio_path"]).exists()]


def get_profile(profile_id: str) -> dict:
    for item in _load_index():
        if item["id"] == profile_id:
            if not Path(item["ref_audio_path"]).exists():
                raise FileNotFoundError(f"Reference audio missing for profile {profile_id}")
            return item
    raise KeyError(f"Voice profile not found: {profile_id}")


def create_profile_from_slice(
    *,
    name: str,
    source_audio: Path,
    start_sec: float,
    duration_sec: float,
    ref_text: str,
    language: str = "en",
    notes: str = "",
) -> dict:
    profile_id = f"vp_{uuid.uuid4().hex[:10]}"
    profile_dir = get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    ref_wav = profile_dir / "ref.wav"
    slice_reference_wav(source_audio, start_sec, duration_sec, ref_wav)
    actual_dur = round(get_duration(ref_wav), 2)

    (profile_dir / "ref_transcript.txt").write_text(ref_text.strip(), encoding="utf-8")

    profile = VoiceProfile(
        id=profile_id,
        name=name.strip(),
        ref_text=ref_text.strip(),
        ref_audio_path=str(ref_wav),
        duration_sec=actual_dur,
        language=language,
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=notes.strip(),
    )
    items = _load_index()
    items.insert(0, profile.to_dict())
    _save_index(items)
    return profile.to_dict()


def update_profile(profile_id: str, *, name: str | None = None, ref_text: str | None = None, notes: str | None = None) -> dict:
    items = _load_index()
    for item in items:
        if item["id"] == profile_id:
            if name is not None:
                item["name"] = name.strip()
            if ref_text is not None:
                item["ref_text"] = ref_text.strip()
                ref_dir = Path(item["ref_audio_path"]).parent
                (ref_dir / "ref_transcript.txt").write_text(item["ref_text"], encoding="utf-8")
            if notes is not None:
                item["notes"] = notes.strip()
            _save_index(items)
            return item
    raise KeyError(f"Voice profile not found: {profile_id}")


def delete_profile(profile_id: str) -> None:
    items = _load_index()
    remaining = [item for item in items if item["id"] != profile_id]
    if len(remaining) == len(items):
        raise KeyError(f"Voice profile not found: {profile_id}")
    _save_index(remaining)
    profile_dir = get_profiles_dir() / profile_id
    if profile_dir.exists():
        shutil.rmtree(profile_dir, ignore_errors=True)
