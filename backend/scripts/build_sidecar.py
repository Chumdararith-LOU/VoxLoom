"""Bundle the Python FastAPI engine + static ffmpeg/ffprobe into a Tauri resource bundle.

Uses PyInstaller --onedir so Windows PyTorch CUDA (cu124) DLLs avoid the 2GB PE --onefile
limit and launch instantaneously with zero temp-folder extraction delay.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
ENGINE_OUT_DIR = ROOT / "src-tauri" / "resources" / "engine"

METADATA_PACKAGES = [
    "transformers",
    "huggingface-hub",
    "tokenizers",
    "safetensors",
    "numpy",
    "scipy",
    "soundfile",
    "tqdm",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "pyyaml",
]


def _copy_ffmpeg_binaries(dest_dir: Path) -> None:
    ext = ".exe" if platform.system() == "Windows" else ""
    for tool in ("ffmpeg", "ffprobe"):
        env_override = os.environ.get(f"VOXLOOM_BUNDLE_{tool.upper()}")
        found = env_override if (env_override and Path(env_override).is_file()) else shutil.which(tool)
        if not found and platform.system() == "Darwin":
            for brew_dir in ("/opt/homebrew/bin", "/usr/local/bin"):
                p = Path(brew_dir) / tool
                if p.is_file():
                    found = str(p)
                    break
        if not found:
            raise RuntimeError(
                f"Required media tool '{tool}' not found on PATH. "
                "Install ffmpeg before bundling."
            )
        target = dest_dir / f"{tool}{ext}"
        shutil.copy2(found, target)
        # Ensure owner-writable executable mode so macOS xattr/codesign never hits EACCES
        os.chmod(target, 0o755)
        print(f"Bundled {tool} -> {target}")


def main() -> None:
    entry = BACKEND_DIR / "scripts" / "engine_entry.py"
    dist_dir = BACKEND_DIR / "dist" / "voxloom-engine"

    if ENGINE_OUT_DIR.exists():
        shutil.rmtree(ENGINE_OUT_DIR)
    ENGINE_OUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name",
        "voxloom-engine",
        "--hidden-import",
        "uvicorn.logging",
        "--hidden-import",
        "uvicorn.loops.auto",
        "--hidden-import",
        "uvicorn.protocols.http.auto",
        "--hidden-import",
        "uvicorn.lifespan.on",
        "--collect-submodules",
        "voxloom_backend",
        "--collect-all",
        "soundfile",
    ]

    for pkg in METADATA_PACKAGES:
        cmd.extend(["--copy-metadata", pkg])

    # Collect platform-specific AI runtime assets (Metal shaders on macOS, CUDA/STT libs on Windows)
    for optional_pkg in ("mlx", "mlx_audio", "qwen_tts", "faster_whisper", "ctranslate2"):
        if importlib.util.find_spec(optional_pkg) is not None:
            cmd.extend(["--collect-all", optional_pkg])

    cmd.append(str(entry))
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=BACKEND_DIR, check=True)

    shutil.copytree(dist_dir, ENGINE_OUT_DIR)
    _copy_ffmpeg_binaries(ENGINE_OUT_DIR)

    # PyInstaller places libmlx.dylib directly in _internal/, where MLX expects mlx.metallib
    # alongside it on macOS Apple Silicon (otherwise mlx::core::metal::Device::Device traps).
    mlx_metallib = ENGINE_OUT_DIR / "_internal" / "mlx" / "lib" / "mlx.metallib"
    if mlx_metallib.is_file():
        target_metallib = ENGINE_OUT_DIR / "_internal" / "mlx.metallib"
        shutil.copy2(mlx_metallib, target_metallib)
        print(f"Copied mlx.metallib -> {target_metallib}")

    print(f"Engine bundle ready at: {ENGINE_OUT_DIR}")


if __name__ == "__main__":
    main()
