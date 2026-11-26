"""Cross-platform paths, settings, and binary discovery."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path


def get_app_data_dir() -> Path:
    """Return the OS-native application data directory for VoxLoom."""
    override = os.environ.get("VOXLOOM_DATA_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "VoxLoom"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        base = Path(appdata) / "VoxLoom"
    else:
        xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        base = Path(xdg) / "voxloom"

    base.mkdir(parents=True, exist_ok=True)
    return base


def get_profiles_dir() -> Path:
    d = get_app_data_dir() / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_outputs_dir() -> Path:
    d = get_app_data_dir() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_uploads_dir() -> Path:
    d = get_app_data_dir() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_binary(name: str) -> str | None:
    """Locate ffmpeg/ffprobe: bundled sidecar first, then system PATH."""
    ext = ".exe" if platform.system() == "Windows" else ""
    candidates: list[Path] = []

    # 1. Explicit environment override
    env_key = f"VOXLOOM_{name.upper()}_PATH"
    if os.environ.get(env_key):
        candidates.append(Path(os.environ[env_key]))

    # 2. Adjacent to frozen PyInstaller binary or Tauri sidecar directory
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / f"{name}{ext}")
        candidates.append(exe_dir / "bin" / f"{name}{ext}")

    # 3. Project-local binaries folder (dev mode)
    repo_bin = Path(__file__).resolve().parents[2] / "bin" / f"{name}{ext}"
    candidates.append(repo_bin)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    # 4. Standard PATH (plus common Homebrew locations on macOS)
    found = shutil.which(name)
    if found:
        return found

    if platform.system() == "Darwin":
        for brew_dir in ("/opt/homebrew/bin", "/usr/local/bin"):
            p = Path(brew_dir) / name
            if p.is_file():
                return str(p)

    return None


def get_ffmpeg_path() -> str:
    path = find_binary("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg binary not found. Install ffmpeg (macOS: `brew install ffmpeg`, "
            "Windows: `winget install Gyan.FFmpeg`) or bundle it with the app."
        )
    return path


def get_ffprobe_path() -> str:
    path = find_binary("ffprobe")
    if not path:
        raise RuntimeError(
            "ffprobe binary not found. Install ffmpeg (which includes ffprobe) "
            "or bundle it with the app."
        )
    return path
