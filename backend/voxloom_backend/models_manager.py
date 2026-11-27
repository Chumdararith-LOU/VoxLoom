"""Model registry, Hugging Face Hub downloader with progress, and singleton engine manager."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import scan_cache_dir
from tqdm.auto import tqdm as base_tqdm

from . import get_hf_cache_dir, hf_model_info, hf_snapshot_download
from .engines.base import STTEngine, TTSEngine
from .hardware import detect_hardware


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    kind: str  # "tts" | "stt"
    backend: str  # "mlx" | "torch"
    repo_id: str
    size_gb: float
    min_ram_gb: int
    description: str
    license: str


MODEL_CATALOG: list[ModelSpec] = [
    ModelSpec(
        id="qwen3-tts-mlx-8bit",
        name="Qwen3-TTS 1.7B Base (MLX 8-bit)",
        kind="tts",
        backend="mlx",
        repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        size_gb=3.1,
        min_ram_gb=16,
        description="High-fidelity 8-bit quantized voice cloning optimized for Apple Silicon Metal.",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="qwen3-tts-mlx-4bit",
        name="Qwen3-TTS 1.7B Base (MLX 4-bit)",
        kind="tts",
        backend="mlx",
        repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
        size_gb=1.8,
        min_ram_gb=8,
        description="Lightweight 4-bit voice cloning for 8GB Apple Silicon Macs.",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="qwen3-tts-mlx-0.6b-8bit",
        name="Qwen3-TTS 0.6B Base (MLX 8-bit Fast)",
        kind="tts",
        backend="mlx",
        repo_id="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        size_gb=0.9,
        min_ram_gb=8,
        description="Ultra-fast compact voice cloning for Apple Silicon.",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="qwen3-tts-torch-0.6b",
        name="Qwen3-TTS 0.6B Base (CUDA 4–6GB / CPU)",
        kind="tts",
        backend="torch",
        repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        size_gb=1.5,
        min_ram_gb=6,
        description="Fast low-VRAM voice cloning ideal for RTX 3050 (4–6 GB VRAM) laptops and CPU systems.",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="qwen3-tts-torch-base",
        name="Qwen3-TTS 1.7B Base (CUDA 8GB+ / CPU)",
        kind="tts",
        backend="torch",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        size_gb=3.6,
        min_ram_gb=12,
        description="Highest-fidelity PyTorch checkpoint for 8 GB+ NVIDIA CUDA GPUs (BF16/FP16) and CPU.",
        license="Apache-2.0",
    ),
    ModelSpec(
        id="whisper-v3-turbo-mlx",
        name="Whisper Large v3 Turbo (MLX)",
        kind="stt",
        backend="mlx",
        repo_id="mlx-community/whisper-large-v3-turbo",
        size_gb=1.6,
        min_ram_gb=8,
        description="Fast local reference-clip transcription for Apple Silicon.",
        license="MIT",
    ),
    ModelSpec(
        id="faster-whisper-turbo",
        name="Faster-Whisper Large v3 Turbo (CUDA / CPU)",
        kind="stt",
        backend="torch",
        repo_id="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        size_gb=1.6,
        min_ram_gb=8,
        description="Fast CTranslate2 speech recognition for Windows/Linux GPUs & CPUs.",
        license="MIT",
    ),
    ModelSpec(
        id="faster-whisper-base",
        name="Faster-Whisper Base (Lightweight CPU)",
        kind="stt",
        backend="torch",
        repo_id="Systran/faster-whisper-base",
        size_gb=0.15,
        min_ram_gb=4,
        description="Ultra-compact local transcription model for CPU-only systems.",
        license="MIT",
    ),
]


def get_model_spec(model_id: str) -> ModelSpec:
    for spec in MODEL_CATALOG:
        if spec.id == model_id or spec.repo_id == model_id:
            return spec
    raise KeyError(f"Unknown model id: {model_id}")


def get_cached_repos() -> dict[str, int]:
    """Return {repo_id: size_on_disk_bytes} for locally cached HF models."""
    cached: dict[str, int] = {}
    try:
        info = scan_cache_dir(cache_dir=get_hf_cache_dir())
        for repo in info.repos:
            cached[repo.repo_id] = repo.size_on_disk
    except Exception:
        pass
    return cached


class EngineManager:
    """Thread-safe singleton holding the active TTS model hot in memory."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tts_engine: TTSEngine | None = None
        self._loaded_model_id: str | None = None

    def _create_tts_engine(self, backend: str) -> TTSEngine:
        if backend == "mlx":
            from .engines.mlx_engine import MLXTTSEngine

            return MLXTTSEngine()
        from .engines.torch_engine import TorchTTSEngine

        hw = detect_hardware()
        return TorchTTSEngine(device="cuda" if hw.cuda_available else "cpu")

    def _create_stt_engine(self, backend: str) -> STTEngine:
        if backend == "mlx":
            from .engines.mlx_engine import MLXSTTEngine

            return MLXSTTEngine()
        from .engines.torch_engine import FasterWhisperSTTEngine

        return FasterWhisperSTTEngine()

    def list_models(self) -> list[dict]:
        hw = detect_hardware()
        cached = get_cached_repos()
        items = []
        for spec in MODEL_CATALOG:
            compatible = (
                (spec.backend == "mlx" and hw.os_name == "Darwin" and hw.arch in ("arm64", "aarch64"))
                or (spec.backend == "torch")
            )
            disk_bytes = cached.get(spec.repo_id, 0)
            items.append(
                {
                    **asdict(spec),
                    "compatible": compatible,
                    "recommended": spec.id in (hw.recommended_tts_model, hw.recommended_stt_model),
                    "downloaded": disk_bytes > 0,
                    "cached_size_gb": round(disk_bytes / (1024**3), 2) if disk_bytes > 0 else 0.0,
                    "loaded": self._loaded_model_id == spec.id,
                }
            )
        return items

    def status(self) -> dict:
        with self._lock:
            return {
                "loaded": self._tts_engine is not None and self._tts_engine.is_loaded,
                "model_id": self._loaded_model_id,
                "repo_id": self._tts_engine.loaded_model_repo if self._tts_engine else None,
                "backend": self._tts_engine.backend_name if self._tts_engine else None,
            }

    def download_model(
        self,
        model_id: str,
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict:
        spec = get_model_spec(model_id)
        if on_progress:
            on_progress(
                {
                    "stage": "starting",
                    "model_id": spec.id,
                    "percent": 0.0,
                    "downloaded_mb": 0.0,
                    "total_mb": round(spec.size_gb * 1024.0, 1),
                    "speed_mb_s": 0.0,
                    "eta_sec": None,
                    "current_file": "",
                    "message": f"Resolving {spec.repo_id}…",
                }
            )

        files = []
        try:
            info = hf_model_info(spec.repo_id, files_metadata=True)
            files = info.siblings or []
        except Exception:
            pass

        total_files = max(1, len(files))
        total_bytes = sum((getattr(s, "size", 0) or 0) for s in files)
        if total_bytes <= 0:
            total_bytes = int(spec.size_gb * (1024**3))

        blobs_dir = (
            Path(get_hf_cache_dir()) / ("models--" + spec.repo_id.replace("/", "--")) / "blobs"
        )

        tracker_lock = threading.Lock()
        tracker = {
            "tqdm_bytes": 0,
            "current_file": "",
        }
        stop_event = threading.Event()

        class _ProgressTqdm(base_tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)
                self._is_bytes = kwargs.get("unit") in ("B", "iB") or bool(kwargs.get("unit_scale"))
                desc = str(kwargs.get("desc") or "").strip()
                if desc and not desc.startswith("Fetching"):
                    with tracker_lock:
                        tracker["current_file"] = desc

            def update(self, n=1):
                if self._is_bytes and n and n > 0:
                    with tracker_lock:
                        tracker["tqdm_bytes"] += int(n)
                return super().update(n)

        def _measure_blobs_bytes() -> int:
            if not blobs_dir.is_dir():
                return 0
            total = 0
            try:
                for p in blobs_dir.iterdir():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
            except OSError:
                pass
            return total

        def _monitor_loop() -> None:
            start_t = time.monotonic()
            prev_t = start_t
            prev_bytes = _measure_blobs_bytes()
            ema_speed = 0.0

            while not stop_event.wait(0.45):
                now = time.monotonic()
                disk_bytes = _measure_blobs_bytes()
                with tracker_lock:
                    cur_bytes = max(disk_bytes, tracker["tqdm_bytes"])
                    cur_file = tracker["current_file"]

                # Cap at 99.5% until snapshot_download returns
                bounded = min(cur_bytes, int(total_bytes * 0.995))
                dt = max(0.05, now - prev_t)
                inst_speed = max(0.0, (cur_bytes - prev_bytes) / dt)
                ema_speed = inst_speed if ema_speed == 0.0 else (0.35 * inst_speed + 0.65 * ema_speed)

                if cur_bytes <= prev_bytes and (now - start_t) > 1.0:
                    avg_speed = max(0.0, cur_bytes / max(0.1, now - start_t))
                    speed_bps = ema_speed if ema_speed > 0 else avg_speed
                else:
                    speed_bps = ema_speed

                prev_t = now
                prev_bytes = cur_bytes

                percent = min(99.5, (bounded / max(1, total_bytes)) * 100.0)
                remaining_bytes = max(0, total_bytes - bounded)
                eta_sec = int(remaining_bytes / speed_bps) if speed_bps > 10_000 else None

                if on_progress:
                    on_progress(
                        {
                            "stage": "downloading",
                            "model_id": spec.id,
                            "percent": round(percent, 1),
                            "downloaded_mb": round(bounded / (1024**2), 1),
                            "total_mb": round(total_bytes / (1024**2), 1),
                            "speed_mb_s": round(speed_bps / (1024**2), 2),
                            "eta_sec": eta_sec,
                            "current_file": cur_file,
                            "total_files": total_files,
                            "message": f"Downloading {spec.name}…",
                        }
                    )

        monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        monitor_thread.start()
        try:
            local_path = hf_snapshot_download(spec.repo_id, tqdm_class=_ProgressTqdm)
        finally:
            stop_event.set()
            monitor_thread.join(timeout=1.5)

        # If this is an MLX Whisper model, also fetch the HF WhisperProcessor config so STT works offline
        if spec.kind == "stt" and spec.backend == "mlx":
            if on_progress:
                on_progress(
                    {
                        "stage": "downloading",
                        "model_id": spec.id,
                        "percent": 99.5,
                        "downloaded_mb": round(total_bytes / (1024**2), 1),
                        "total_mb": round(total_bytes / (1024**2), 1),
                        "speed_mb_s": 0.0,
                        "eta_sec": 0,
                        "current_file": "WhisperProcessor config",
                        "message": "Fetching Whisper processor config…",
                    }
                )
            try:
                from .engines.mlx_engine import MLXSTTEngine

                MLXSTTEngine()._ensure_whisper_processor(spec.repo_id)
            except Exception:
                pass

        if on_progress:
            on_progress(
                {
                    "stage": "complete",
                    "model_id": spec.id,
                    "percent": 100.0,
                    "downloaded_mb": round(total_bytes / (1024**2), 1),
                    "total_mb": round(total_bytes / (1024**2), 1),
                    "speed_mb_s": 0.0,
                    "eta_sec": 0,
                    "current_file": "",
                    "message": f"Cached locally at {local_path}",
                }
            )
        return {"model_id": spec.id, "repo_id": spec.repo_id, "local_path": local_path}

    def load_tts(self, model_id: str | None = None) -> dict:
        hw = detect_hardware()
        target_id = model_id or hw.recommended_tts_model
        spec = get_model_spec(target_id)
        if spec.kind != "tts":
            raise ValueError(f"{target_id} is not a TTS model.")

        with self._lock:
            if (
                self._tts_engine is not None
                and self._tts_engine.is_loaded
                and self._loaded_model_id == spec.id
            ):
                return self.status()

            if self._tts_engine is not None:
                self._tts_engine.unload()

            engine = self._create_tts_engine(spec.backend)
            engine.load(spec.repo_id)
            self._tts_engine = engine
            self._loaded_model_id = spec.id
            return self.status()

    def unload_tts(self) -> dict:
        with self._lock:
            if self._tts_engine is not None:
                self._tts_engine.unload()
                self._tts_engine = None
                self._loaded_model_id = None
            return self.status()

    def get_loaded_tts(self, model_id: str | None = None) -> TTSEngine:
        with self._lock:
            if self._tts_engine is None or not self._tts_engine.is_loaded:
                self.load_tts(model_id)
            elif model_id and self._loaded_model_id != model_id:
                self.load_tts(model_id)
            assert self._tts_engine is not None
            return self._tts_engine

    def transcribe_reference(self, audio_path, model_id: str | None = None, language: str | None = None) -> str:
        hw = detect_hardware()
        target_id = model_id or hw.recommended_stt_model
        spec = get_model_spec(target_id)
        engine = self._create_stt_engine(spec.backend)
        return engine.transcribe(audio_path, model_repo=spec.repo_id, language=language)


engine_manager = EngineManager()
