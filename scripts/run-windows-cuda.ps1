# One-command launcher for VoxLoom with full NVIDIA CUDA 12.4 GPU acceleration on Windows 10/11
# Ideal for RTX 3050 (4-8GB VRAM) / RTX 3060 / RTX 40-series GPUs.
$ErrorActionPreference = "Stop"

Write-Host "=== VoxLoom Windows NVIDIA CUDA 12.4 Launcher ===" -ForegroundColor Cyan

# Ensure prerequisites
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..." -ForegroundColor Yellow
    winget install --id astral-sh.uv -e --source winget
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Installing FFmpeg..." -ForegroundColor Yellow
    winget install --id Gyan.FFmpeg -e --source winget
}
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
    Write-Host "Installing pnpm..." -ForegroundColor Yellow
    npm install -g pnpm
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
pnpm install

Write-Host "Setting up Python backend with PyTorch CUDA 12.4..." -ForegroundColor Cyan
Set-Location "$RepoRoot\backend"
uv sync --extra torch --prerelease=allow
uv pip install --reinstall torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple

uv run --no-sync python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host "`nStarting VoxLoom AI Engine (http://127.0.0.1:8765) and Studio UI (http://localhost:1420)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RepoRoot\backend'; uv run --no-sync uvicorn voxloom_backend.main:app --host 127.0.0.1 --port 8765"
Set-Location $RepoRoot
Start-Process "http://localhost:1420"
pnpm dev
