"""System health and hardware inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..config import get_app_data_dir
from ..hardware import detect_hardware
from ..models_manager import engine_manager

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "data_dir": str(get_app_data_dir()),
        "engine": engine_manager.status(),
    }


@router.get("/hardware")
def hardware() -> dict:
    return detect_hardware().to_dict()
