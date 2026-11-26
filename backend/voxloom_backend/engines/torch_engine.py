"""Windows / Linux cross-platform inference backend powered by PyTorch (CUDA/CPU) + faster-whisper."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import STTEngine, TTSEngine


class TorchTTSEngine(TTSEngine):
    """Qwen3-TTS voice cloning backend for NVIDIA CUDA GPUs and CPU fallback."""

    def __init__(self, device: str | None = None) -> None:
        self._model = None
        self._repo: str | None = None
        self._device = device
        self.backend_name = device or "torch"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model_repo(self) -> str | None:
        return self._repo

    def _resolve_device_and_dtype(self):
        import torch

        if self._device:
            device = self._device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device.startswith("cuda") and torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.backend_name = "cuda"
        else:
            device = "cpu"
            dtype = torch.float32
            self.backend_name = "cpu"
        return device, dtype

    def load(self, model_repo: str) -> None:
        if self._model is not None and self._repo == model_repo:
            return
        self.unload()

        device, dtype = self._resolve_device_and_dtype()

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch TTS dependencies are not installed. Install with: "
                "`uv pip install '.[torch]'` (or download the Windows CUDA/CPU release)."
            ) from exc

        self._model = Qwen3TTSModel.from_pretrained(
            model_repo,
            device_map=device,
            dtype=dtype,
        )
        self._repo = model_repo

    def unload(self) -> None:
        if self._model is not None:
            self._model = None
            self._repo = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def generate(
        self,
        *,
        text: str,
        ref_audio: Path,
        ref_text: str,
        output_wav: Path,
        speed: float = 1.0,
        temperature: float = 0.7,
        language: str = "en",
    ) -> Path:
        if self._model is None:
            raise RuntimeError("TTS model is not loaded. Load a model before generating.")

        lang_map = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
        }
        qwen_lang = lang_map.get(language.lower(), "Auto")

        wavs, sample_rate = self._model.generate_voice_clone(
            text=text,
            language=qwen_lang,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            temperature=temperature,
        )
        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        audio_np = np.asarray(audio, dtype=np.float32).squeeze()

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), audio_np, int(sample_rate), subtype="PCM_16")
        return output_wav


class FasterWhisperSTTEngine(STTEngine):
    """Fast local transcription for Windows / Linux (CUDA float16 or CPU int8)."""

    backend_name = "faster-whisper"

    def transcribe(
        self,
        audio_path: Path,
        model_repo: str = "turbo",
        language: str | None = None,
    ) -> str:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install with `uv pip install '.[torch]'`."
            ) from exc

        device = "cpu"
        compute_type = "int8"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
        except Exception:
            pass

        size = "turbo" if "turbo" in model_repo else ("base" if "base" in model_repo else model_repo)
        model = WhisperModel(size, device=device, compute_type=compute_type)
        try:
            segments, _ = model.transcribe(str(audio_path), language=language, vad_filter=True)
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            del model
            gc.collect()
