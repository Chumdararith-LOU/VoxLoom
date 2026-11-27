"""Apple Silicon (Metal) inference backend powered by mlx-audio."""

from __future__ import annotations

import gc
import shutil
from pathlib import Path

from .. import get_hf_token, hf_snapshot_download
from .base import STTEngine, TTSEngine


class MLXTTSEngine(TTSEngine):
    backend_name = "mlx"

    def __init__(self) -> None:
        self._model = None
        self._repo: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model_repo(self) -> str | None:
        return self._repo

    def load(self, model_repo: str) -> None:
        if self._model is not None and self._repo == model_repo:
            return
        self.unload()
        from mlx_audio.tts.utils import load_model

        if Path(model_repo).is_dir():
            local_dir = Path(model_repo)
        else:
            try:
                local_dir = Path(hf_snapshot_download(model_repo, local_files_only=True))
            except Exception:
                local_dir = Path(hf_snapshot_download(model_repo))

        self._model = load_model(local_dir)
        self._repo = model_repo

    def unload(self) -> None:
        if self._model is not None:
            self._model = None
            self._repo = None
            gc.collect()
            try:
                import mlx.core as mx

                mx.clear_cache()
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

        from mlx_audio.tts.generate import generate_audio

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        stem = output_wav.stem

        ok = generate_audio(
            model=self._model,
            text=text,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            output_path=str(output_wav.parent),
            file_prefix=stem,
            audio_format="wav",
            join_audio=True,
            play=False,
            verbose=False,
            speed=speed,
            temperature=temperature,
            lang_code=language,
        )
        if not ok or not output_wav.exists():
            raise RuntimeError(f"MLX TTS generation failed to produce {output_wav}")
        return output_wav


class MLXSTTEngine(STTEngine):
    backend_name = "mlx"

    def _ensure_whisper_processor(self, model_repo: str) -> str:
        """Ensure mlx-community Whisper snapshots include HuggingFace processor files."""
        if Path(model_repo).is_dir():
            snapshot_dir = Path(model_repo)
        else:
            try:
                snapshot_dir = Path(hf_snapshot_download(model_repo, local_files_only=True))
            except Exception:
                snapshot_dir = Path(hf_snapshot_download(model_repo))

        if not (snapshot_dir / "preprocessor_config.json").exists():
            from transformers import WhisperProcessor

            base_repo = "openai/whisper-large-v3-turbo"
            if "medium" in model_repo:
                base_repo = "openai/whisper-medium"
            elif "small" in model_repo:
                base_repo = "openai/whisper-small"
            elif "base" in model_repo:
                base_repo = "openai/whisper-base"

            token = get_hf_token()
            try:
                processor = WhisperProcessor.from_pretrained(base_repo, token=token)
            except Exception:
                processor = WhisperProcessor.from_pretrained(base_repo, token=False)
            processor.save_pretrained(str(snapshot_dir))
        return str(snapshot_dir)

    def transcribe(
        self,
        audio_path: Path,
        model_repo: str = "mlx-community/whisper-large-v3-turbo",
        language: str | None = None,
    ) -> str:
        from mlx_audio.stt import load as load_stt_model

        local_dir = self._ensure_whisper_processor(model_repo)
        model = load_stt_model(local_dir)
        try:
            kwargs = {}
            if language:
                kwargs["language"] = language
            result = model.generate(str(audio_path), **kwargs)
            text = getattr(result, "text", None) or str(result)
            return text.strip()
        finally:
            del model
            gc.collect()
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass
