"""FastAPI application entrypoint and sidecar CLI."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .models_manager import engine_manager
from .routers import generate, models, profiles, system


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    engine_manager.unload_tts()


def create_app() -> FastAPI:
    app = FastAPI(
        title="VoxLoom Local AI Engine",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:1420",
            "http://127.0.0.1:1420",
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(system.router)
    app.include_router(models.router)
    app.include_router(profiles.router)
    app.include_router(generate.router)
    return app


app = create_app()


def _ensure_valid_stdio() -> None:
    """Prevent crashes when launched without a console (macOS Finder / Windows GUI)."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _configure_tls_certificates() -> None:
    """Ensure HTTPS requests to Hugging Face Hub work inside frozen Windows/macOS bundles."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import certifi

        ca_bundle = certifi.where()
        if os.path.isfile(ca_bundle):
            os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
            os.environ.setdefault("CURL_CA_BUNDLE", ca_bundle)
    except Exception:
        pass
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


def cli() -> None:
    _ensure_valid_stdio()
    _configure_tls_certificates()
    parser = argparse.ArgumentParser(description="Run VoxLoom local FastAPI backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    # Print structured ready marker so the Tauri host can discover the listening address
    print(f"VOXLOOM_ENGINE_STARTING http://{args.host}:{args.port}", flush=True)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        use_colors=False,
    )


if __name__ == "__main__":
    cli()
