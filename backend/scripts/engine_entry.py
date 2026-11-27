"""PyInstaller entrypoint for VoxLoom local AI engine (avoids relative-import errors)."""

import multiprocessing

from voxloom_backend import configure_hf_environment

configure_hf_environment()

from voxloom_backend.main import cli  # noqa: E402

if __name__ == "__main__":
    multiprocessing.freeze_support()
    cli()
