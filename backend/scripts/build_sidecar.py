"""Bundle the Python FastAPI engine into a standalone Tauri sidecar binary via PyInstaller."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
TAURI_BIN_DIR = ROOT / "src-tauri" / "binaries"


def get_target_triple() -> str:
    """Return the Rust target triple Tauri expects for externalBin sidecars."""
    try:
        out = subprocess.check_output(["rustc", "-vV"], text=True)
        for line in out.splitlines():
            if line.startswith("host:"):
                return line.split("host:")[1].strip()
    except Exception:
        pass

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        return "aarch64-apple-darwin" if machine in ("arm64", "aarch64") else "x86_64-apple-darwin"
    if system == "Windows":
        return "x86_64-pc-windows-msvc"
    return "x86_64-unknown-linux-gnu"


def main() -> None:
    triple = get_target_triple()
    ext = ".exe" if platform.system() == "Windows" else ""
    TAURI_BIN_DIR.mkdir(parents=True, exist_ok=True)

    entry = BACKEND_DIR / "voxloom_backend" / "main.py"
    dist_name = f"voxloom-engine-{triple}"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--name",
        dist_name,
        "--hidden-import",
        "uvicorn.logging",
        "--hidden-import",
        "uvicorn.loops.auto",
        "--hidden-import",
        "uvicorn.protocols.http.auto",
        "--hidden-import",
        "uvicorn.lifespan.on",
        str(entry),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=BACKEND_DIR, check=True)

    built_bin = BACKEND_DIR / "dist" / f"{dist_name}{ext}"
    dest_bin = TAURI_BIN_DIR / f"{dist_name}{ext}"
    shutil.copy2(built_bin, dest_bin)
    print(f"Sidecar binary ready at: {dest_bin}")


if __name__ == "__main__":
    main()
