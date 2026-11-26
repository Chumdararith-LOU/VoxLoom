"""Cross-platform hardware & accelerator detection (Apple Silicon MLX, NVIDIA CUDA, CPU)."""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
from dataclasses import asdict, dataclass

from .config import find_binary


@dataclass
class HardwareProfile:
    os_name: str
    os_version: str
    arch: str
    ram_gb: float
    recommended_backend: str  # "mlx" | "cuda" | "cpu"
    available_backends: list[str]
    accelerator_name: str
    vram_gb: float | None
    mlx_audio_installed: bool
    torch_installed: bool
    cuda_available: bool
    ffmpeg_available: bool
    recommended_tts_model: str
    recommended_stt_model: str

    def to_dict(self) -> dict:
        return asdict(self)


def _get_total_ram_gb() -> float:
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return round(int(out) / (1024**3), 1)
        if system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024**3), 1)
        # Linux / POSIX
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024**3), 1)
    except Exception:
        return 16.0


def _detect_apple_chip() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return "Apple Silicon (Metal)"


def _detect_nvidia_gpu() -> tuple[bool, str | None, float | None]:
    # Try PyTorch first if available
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / (1024**3), 1)
                return True, name, vram_gb
        except Exception:
            pass

    # Fallback to nvidia-smi (useful before PyTorch CUDA is installed)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        if out:
            first = out.splitlines()[0].split(",")
            name = first[0].strip()
            vram_gb = round(float(first[1].strip()) / 1024.0, 1)
            return True, name, vram_gb
    except Exception:
        pass

    return False, None, None


def detect_hardware() -> HardwareProfile:
    os_name = platform.system()
    os_version = platform.release()
    arch = platform.machine().lower()
    ram_gb = _get_total_ram_gb()

    is_apple_silicon = os_name == "Darwin" and arch in ("arm64", "aarch64")
    mlx_audio_installed = importlib.util.find_spec("mlx_audio") is not None
    torch_installed = importlib.util.find_spec("torch") is not None
    cuda_available, cuda_name, vram_gb = _detect_nvidia_gpu()
    ffmpeg_available = find_binary("ffmpeg") is not None and find_binary("ffprobe") is not None

    available: list[str] = []
    if is_apple_silicon and mlx_audio_installed:
        available.append("mlx")
    if cuda_available and torch_installed:
        available.append("cuda")
    if torch_installed:
        available.append("cpu")

    if is_apple_silicon:
        recommended = "mlx"
        accelerator = _detect_apple_chip()
        vram_gb = ram_gb  # Unified memory
        rec_tts = (
            "qwen3-tts-mlx-4bit" if ram_gb <= 12.0 else "qwen3-tts-mlx-8bit"
        )
        rec_stt = "whisper-v3-turbo-mlx"
    elif cuda_available:
        recommended = "cuda"
        accelerator = cuda_name or "NVIDIA CUDA GPU"
        rec_tts = "qwen3-tts-torch-base"
        rec_stt = "faster-whisper-turbo"
    else:
        recommended = "cpu"
        accelerator = platform.processor() or f"CPU ({arch})"
        rec_tts = "qwen3-tts-torch-base"
        rec_stt = "faster-whisper-base"

    return HardwareProfile(
        os_name=os_name,
        os_version=os_version,
        arch=arch,
        ram_gb=ram_gb,
        recommended_backend=recommended,
        available_backends=available,
        accelerator_name=accelerator,
        vram_gb=vram_gb,
        mlx_audio_installed=mlx_audio_installed,
        torch_installed=torch_installed,
        cuda_available=cuda_available,
        ffmpeg_available=ffmpeg_available,
        recommended_tts_model=rec_tts,
        recommended_stt_model=rec_stt,
    )
