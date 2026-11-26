"""Inference engine backends (MLX for macOS Apple Silicon, PyTorch for Windows/Linux CUDA & CPU)."""

from .base import STTEngine, TTSEngine

__all__ = ["TTSEngine", "STTEngine"]
