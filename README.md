# VoxLoom — Local Voice Cloning & Timeline Gap-Fill Studio

**VoxLoom** is a 100% local, privacy-first desktop application for voice cloning and video/podcast dialogue gap-filling built with **Tauri v2 + React/TypeScript** and a **cross-platform Python FastAPI AI engine**.

Clone your voice from an 8–15 second reference sample, keep the **Qwen3-TTS 1.7B** model hot in memory, and generate loudness-normalized, duration-fitted voiceover clips with zero cloud fees or uploads.

---

## Features

- **Cross-Platform Local AI Engine**
  - **macOS (Apple Silicon M1–M4)**: Native Metal acceleration via `mlx-audio` (`Qwen3-TTS-12Hz-1.7B-Base-8bit` / `4bit` + `whisper-large-v3-turbo`).
  - **Windows & Linux (NVIDIA CUDA / CPU)**: PyTorch CUDA (BF16/FP16) with automatic CPU fallback (`Qwen/Qwen3-TTS-12Hz-1.7B-Base` + `faster-whisper`).
- **Interactive Voice Profile Studio**
  - Drag-and-drop `.m4a`, `.wav`, `.mp3`, `.mp4` recordings or capture directly from your microphone.
  - Canvas waveform trimmer with **Auto-Pick Cleanest 10–14s** silence/RMS analyzer so clips never cut mid-word.
  - One-click local Whisper transcription for exact `--ref_text` alignment.
- **Single Take & A/B Comparison**
  - Instant repeat synthesis while keeping the model warm in memory, with side-by-side playback against your original reference clip.
- **Batch Timeline Gap-Fill Desk**
  - Import JSON or CSV script lists with per-line `max_duration` targets.
  - Automated `ffmpeg` post-processing: leading/trailing silence trim (50 ms padding), gentle `atempo` duration fitting (capped at `1.08x`, auto-retry + flag if still over), EBU R128 loudness normalization (`-20 LUFS` / `-16 LUFS`, 48 kHz mono), and one-click ZIP export.
- **In-App Hugging Face Model Manager**
  - No huge weights committed to Git: models download directly from Hugging Face Hub with live status and cache locally for offline operation.

---

## Hardware Support Matrix

| Platform | Accelerator | Recommended TTS Model | Recommended RAM |
| :--- | :--- | :--- | :--- |
| macOS Apple Silicon (16GB+) | Metal (`mlx-audio`) | `Qwen3-TTS-12Hz-1.7B-Base-8bit` (~3.1 GB) | 16 GB+ Unified |
| macOS Apple Silicon (8GB) | Metal (`mlx-audio`) | `Qwen3-TTS-12Hz-1.7B-Base-4bit` (~1.8 GB) | 8–12 GB Unified |
| Windows / Linux (NVIDIA GPU) | CUDA (`torch`) | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (~3.6 GB) | 8 GB+ VRAM |
| Windows / Linux (CPU-only) | CPU (`torch`) | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` + `faster-whisper-base` | 16 GB RAM |

---

## Quick Start (Development)

### Prerequisites
- **Node.js 20+** and **pnpm**
- **Rust** (`rustup` / `cargo`) for Tauri desktop builds
- **Python 3.10–3.13** and **[uv](https://docs.astral.sh/uv/)**
- **FFmpeg** (`brew install ffmpeg` on macOS, `winget install Gyan.FFmpeg` on Windows)

### 1. Install Frontend Dependencies
```bash
pnpm install
```

### 2. Install Local AI Backend Dependencies

**macOS (Apple Silicon MLX):**
```bash
cd backend
uv sync --extra mlx --extra dev
```

**Windows / Linux (CUDA or CPU):**
```bash
cd backend
uv sync --extra torch --extra dev
```

### 3. Run in Development Mode
In Terminal 1 (starts the local FastAPI engine on `http://127.0.0.1:8765`):
```bash
pnpm backend:dev
```

In Terminal 2 (launches the Tauri desktop app, or run `pnpm dev` for browser mode at `http://localhost:1420`):
```bash
pnpm tauri:dev
```

### 4. Run Automated Tests
```bash
pnpm backend:test
pnpm build
```

---

## Building Standalone GitHub Releases (`.dmg` / `.msi` / `.exe`)

GitHub Actions (`.github/workflows/release.yml`) automatically freezes the Python engine via PyInstaller (`backend/scripts/build_sidecar.py`) and builds native installers for both **macOS Apple Silicon** and **Windows x64** whenever you push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

To build locally on your machine:
```bash
cd backend && uv run python scripts/build_sidecar.py
cd .. && pnpm tauri build --config src-tauri/tauri.release.conf.json
```

---

## Responsible Use & Licensing

- **Responsible Use**: Only clone your own voice or voices for which you have explicit permission. VoxLoom includes a first-launch consent dialog and processes all audio strictly on-device.
- **App Code**: Licensed under the [Apache License 2.0](LICENSE).
- **Third-Party Models & Tools**:
  - Qwen3-TTS (`Qwen/Qwen3-TTS-12Hz-1.7B-Base` / MLX conversions): Apache-2.0
  - Whisper / Faster-Whisper: MIT
  - FFmpeg: LGPL/GPL (respect FFmpeg license terms when bundling static binaries)
