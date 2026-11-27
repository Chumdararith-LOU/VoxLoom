"""Model catalog, Hugging Face download (SSE), and hot-memory lifecycle endpoints."""

from __future__ import annotations

import asyncio
import json
import queue
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..models_manager import engine_manager

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelActionRequest(BaseModel):
    model_id: str | None = None


@router.get("")
def list_models() -> dict:
    return {
        "models": engine_manager.list_models(),
        "active": engine_manager.status(),
    }


@router.get("/check-hub")
def check_hub() -> dict:
    """Verify TLS connectivity to Hugging Face Hub and platform AI engine imports."""
    from .. import hf_model_info
    from ..hardware import detect_hardware

    hw = detect_hardware()
    repo = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit" if hw.recommended_backend == "mlx" else "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    try:
        info = hf_model_info(repo)
        if hw.recommended_backend == "mlx":
            import mlx_audio  # noqa: F401
        else:
            import faster_whisper  # noqa: F401
            from qwen_tts import Qwen3TTSModel  # noqa: F401
        return {"ok": True, "repo": info.id, "backend": hw.recommended_backend}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hub/runtime check failed: {exc}") from exc


@router.post("/load")
def load_model(req: ModelActionRequest) -> dict:
    try:
        return engine_manager.load_tts(req.model_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/unload")
def unload_model() -> dict:
    return engine_manager.unload_tts()


@router.post("/download")
async def download_model_stream(req: ModelActionRequest) -> StreamingResponse:
    if not req.model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    q: queue.Queue[dict | None] = queue.Queue()

    def worker() -> None:
        try:
            res = engine_manager.download_model(
                req.model_id,  # type: ignore[arg-type]
                on_progress=lambda evt: q.put(evt),
            )
            q.put({"stage": "done", **res})
        except Exception as exc:
            q.put({"stage": "error", "message": str(exc)})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_generator():
        while True:
            evt = await asyncio.to_thread(q.get)
            if evt is None:
                break
            yield f"data: {json.dumps(evt)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
