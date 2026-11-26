"""PyInstaller entrypoint for VoxLoom local AI engine (avoids relative-import errors)."""

import multiprocessing

from voxloom_backend.main import cli

if __name__ == "__main__":
    multiprocessing.freeze_support()
    cli()
