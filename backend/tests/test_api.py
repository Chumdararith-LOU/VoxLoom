"""FastAPI integration tests covering system, profiles, and mocked single/batch TTS generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wavfile
from fastapi.testclient import TestClient

from voxloom_backend.engines.base import TTSEngine
from voxloom_backend.main import create_app
from voxloom_backend.models_manager import engine_manager


class DummyTTSEngine(TTSEngine):
    backend_name = "mock"

    def __init__(self) -> None:
        self._loaded = True
        self._repo = "mock/qwen3-tts"

    def load(self, model_repo: str) -> None:
        self._loaded = True
        self._repo = model_repo

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def loaded_model_repo(self) -> str | None:
        return self._repo

    def generate(self, *, text: str, ref_audio: Path, ref_text: str, output_wav: Path, **kwargs) -> Path:
        sr = 24000
        dur = 1.6
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        pcm = (0.25 * np.sin(2 * np.pi * 300.0 * t) * 32767).astype(np.int16)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(output_wav), sr, pcm)
        return output_wav


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VOXLOOM_DATA_DIR", str(tmp_path / "voxloom_data"))
    dummy = DummyTTSEngine()
    monkeypatch.setattr(engine_manager, "get_loaded_tts", lambda model_id=None: dummy)
    app = create_app()
    return TestClient(app)


def _create_sample_wav_bytes() -> bytes:
    sr = 24000
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * 240.0 * t) * 32767).astype(np.int16)
    import io

    buf = io.BytesIO()
    wavfile.write(buf, sr, pcm)
    return buf.getvalue()


def test_health_hardware_and_models(client: TestClient) -> None:
    r = client.get("/api/system/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    hw = client.get("/api/system/hardware")
    assert hw.status_code == 200
    assert hw.json()["recommended_backend"] in ("mlx", "cuda", "cpu")

    models = client.get("/api/models")
    assert models.status_code == 200
    assert len(models.json()["models"]) >= 4


def test_profile_lifecycle_and_generation(client: TestClient) -> None:
    wav_bytes = _create_sample_wav_bytes()
    up = client.post(
        "/api/profiles/upload",
        files={"file": ("voice_sample.wav", wav_bytes, "audio/wav")},
    )
    assert up.status_code == 200
    upload_id = up.json()["upload_id"]

    created = client.post(
        "/api/profiles",
        json={
            "name": "Test Narrator",
            "upload_id": upload_id,
            "start_sec": 0.0,
            "duration_sec": 3.0,
            "ref_text": "This is a clean reference clip for testing.",
            "language": "en",
            "notes": "Unit test",
        },
    )
    assert created.status_code == 200
    profile_id = created.json()["profile"]["id"]

    # Single take generation
    single = client.post(
        "/api/generate/single",
        json={
            "profile_id": profile_id,
            "text": "Hello from VoxLoom local studio.",
        },
    )
    assert single.status_code == 200
    audio_url = single.json()["audio_url"]
    assert client.get(audio_url).status_code == 200

    # Batch gap-fill SSE generation + ZIP download
    batch = client.post(
        "/api/generate/batch",
        json={
            "profile_id": profile_id,
            "items": [
                {"filename": "gap_01.wav", "timeline_start": "0m10s", "max_duration": 3.0, "text": "Line one."},
                {"filename": "gap_02.wav", "timeline_start": "0m20s", "max_duration": 3.0, "text": "Line two."},
            ],
        },
    )
    assert batch.status_code == 200
    events = [
        json.loads(line.removeprefix("data: "))
        for line in batch.text.splitlines()
        if line.startswith("data: ")
    ]
    complete = [e for e in events if e.get("type") == "complete"]
    assert len(complete) == 1
    assert len(complete[0]["results"]) == 2
    assert complete[0]["results"][0]["status"] == "PASS"

    zip_resp = client.get(complete[0]["zip_url"])
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"

    # Delete profile
    deleted = client.delete(f"/api/profiles/{profile_id}")
    assert deleted.status_code == 200


EXPIRED_OAUTH_TOKEN = (
    "hf_oauth_eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJleHAiOjE3MDAwMDAwMDAsInN1YiI6InRlc3QifQ.expired_signature"
)


def test_expired_hf_oauth_token_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure cached/env Hugging Face OAuth tokens are never sent implicitly on public Hub requests."""
    from huggingface_hub.utils._headers import build_hf_headers

    from voxloom_backend import configure_hf_environment, get_hf_token

    fake_hf_home = tmp_path / "hf_home"
    fake_hf_home.mkdir(parents=True, exist_ok=True)
    (fake_hf_home / "token").write_text(EXPIRED_OAUTH_TOKEN, encoding="utf-8")

    monkeypatch.delenv("VOXLOOM_HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(fake_hf_home))
    monkeypatch.setenv("HF_TOKEN", EXPIRED_OAUTH_TOKEN)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", EXPIRED_OAUTH_TOKEN)

    configure_hf_environment()

    assert get_hf_token() is False
    explicit_headers = build_hf_headers(token=get_hf_token())
    implicit_headers = build_hf_headers()
    assert "authorization" not in {k.lower() for k in explicit_headers}
    assert "authorization" not in {k.lower() for k in implicit_headers}


def test_live_hub_with_expired_oauth_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify live Hugging Face Hub metadata + file download succeed even with a poisoned expired OAuth token."""
    import socket

    from voxloom_backend import (
        configure_hf_environment,
        hf_model_info,
        hf_snapshot_download,
    )

    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        pytest.skip("Offline environment: skipping live Hugging Face Hub verification")

    fake_hf_home = tmp_path / "hf_home_live"
    fake_hf_home.mkdir(parents=True, exist_ok=True)
    (fake_hf_home / "token").write_text(EXPIRED_OAUTH_TOKEN, encoding="utf-8")

    monkeypatch.delenv("VOXLOOM_HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(fake_hf_home))
    monkeypatch.setenv("HF_HUB_CACHE", str(fake_hf_home / "hub"))
    monkeypatch.setenv("HF_TOKEN", EXPIRED_OAUTH_TOKEN)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", EXPIRED_OAUTH_TOKEN)

    configure_hf_environment()

    repo = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    info = hf_model_info(repo, files_metadata=True)
    assert info.id == repo

    local_dir = Path(hf_snapshot_download(repo, allow_patterns=["config.json"]))
    assert (local_dir / "config.json").is_file()

    # Also verify fallback when a user-supplied VOXLOOM_HF_TOKEN itself has expired
    monkeypatch.setenv("VOXLOOM_HF_TOKEN", EXPIRED_OAUTH_TOKEN)
    fallback_info = hf_model_info(repo)
    assert fallback_info.id == repo
