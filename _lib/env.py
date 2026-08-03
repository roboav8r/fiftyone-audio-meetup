"""Load this repo's .env into the current process. Use BEFORE importing fiftyone."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def load(path: str | Path | None = None, *, override: bool = True) -> dict[str, str]:
    """Load an env file into os.environ. Defaults to this repo's top-level .env.

    If `path` is omitted and the default .env doesn't exist (e.g. running
    against local OSS FiftyOne, which needs no env file), this is a no-op --
    an explicit `path` that doesn't exist is still an error.

    Returns the loaded dict.
    """
    env_path = Path(path) if path else DEFAULT_ENV_PATH
    if not env_path.is_file():
        if path is None:
            return {}
        raise FileNotFoundError(
            f"No env file at {env_path}. Copy .env.example to .env and fill it in."
        )
    values = dict(dotenv_values(env_path))
    for key, value in values.items():
        if value is None:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return {k: v for k, v in values.items() if v is not None}
