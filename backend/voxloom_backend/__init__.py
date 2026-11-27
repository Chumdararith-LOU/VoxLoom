"""VoxLoom local AI backend package."""

from __future__ import annotations

import os
import sys

__version__ = "0.1.6"


def configure_hf_environment() -> None:
    """Prevent expired machine-cached Hugging Face OAuth tokens from breaking public downloads.

    All models in VoxLoom's catalog are public (`private=False, gated=False`).
    If another tool left an expired `hf_oauth_...` token in `~/.cache/huggingface/token`
    or `HF_TOKEN`, `huggingface_hub` sends `Authorization: Bearer ...` by default and
    Hugging Face rejects the request with `401 Client Error: OAuth token has expired`.
    Users who intentionally need a custom token can set `VOXLOOM_HF_TOKEN`.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        os.environ.pop(env_key, None)

    hf_constants = sys.modules.get("huggingface_hub.constants")
    if hf_constants is not None:
        setattr(hf_constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)


def get_hf_token() -> str | bool:
    """Return explicit `VOXLOOM_HF_TOKEN` when set, otherwise `False` (never use implicit cached token)."""
    explicit_token = (os.environ.get("VOXLOOM_HF_TOKEN") or "").strip()
    return explicit_token if explicit_token else False


def get_hf_cache_dir() -> str:
    """Return active Hugging Face Hub cache directory, honoring runtime env overrides."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    from huggingface_hub.constants import HF_HUB_CACHE

    return str(HF_HUB_CACHE)


def _is_unauthorized_error(exc: BaseException) -> bool:
    resp = getattr(exc, "response", None)
    if getattr(resp, "status_code", None) == 401:
        return True
    msg = str(exc).lower()
    return (
        "401 client error" in msg
        or "oauth token has expired" in msg
        or "invalid credentials" in msg
    )


def hf_model_info(repo_id: str, *, files_metadata: bool = False):
    """Fetch Hugging Face model metadata without failing on expired ambient tokens."""
    from huggingface_hub import HfApi

    configure_hf_environment()
    token = get_hf_token()
    try:
        return HfApi(token=token).model_info(repo_id, files_metadata=files_metadata)
    except Exception as exc:
        if token is not False and _is_unauthorized_error(exc):
            return HfApi(token=False).model_info(repo_id, files_metadata=files_metadata)
        raise


def hf_snapshot_download(repo_id: str, **kwargs) -> str:
    """Download or resolve a Hugging Face repo snapshot without implicit expired OAuth tokens."""
    from huggingface_hub import snapshot_download

    configure_hf_environment()
    kwargs.setdefault("cache_dir", get_hf_cache_dir())
    token = kwargs.pop("token", get_hf_token())
    try:
        return snapshot_download(repo_id=repo_id, token=token, **kwargs)
    except Exception as exc:
        if token is not False and _is_unauthorized_error(exc):
            return snapshot_download(repo_id=repo_id, token=False, **kwargs)
        raise


configure_hf_environment()
