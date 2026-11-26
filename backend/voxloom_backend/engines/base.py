"""Pluggable inference engine abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TTSEngine(ABC):
    """Abstract local Text-to-Speech voice cloning engine."""

    backend_name: str = "base"

    @abstractmethod
    def load(self, model_repo: str) -> None:
        """Load model weights into accelerator/system memory once."""

    @abstractmethod
    def unload(self) -> None:
        """Free accelerator/system memory."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model weights are currently hot in memory."""

    @property
    @abstractmethod
    def loaded_model_repo(self) -> str | None:
        """Return currently loaded Hugging Face repo ID."""

    @abstractmethod
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
        """Generate cloned speech WAV into output_wav."""


class STTEngine(ABC):
    """Abstract local Speech-to-Text engine for transcribing reference clips."""

    backend_name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: Path, model_repo: str, language: str | None = None) -> str:
        """Return clean transcript text for a reference audio clip."""
