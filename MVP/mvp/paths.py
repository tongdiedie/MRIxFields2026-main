from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_LOADED = False
_ENV_DIR: Optional[Path] = None


def load_env(search_from: Optional[Path] = None) -> None:
    """Load repo-root .env if present.

    Values already in os.environ are kept. Relative paths in the .env file can
    later be resolved with :func:`resolve_path`.
    """
    global _LOADED, _ENV_DIR
    if _LOADED:
        return
    _LOADED = True

    start = search_from or Path.cwd()
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for d in candidates:
        env_path = d / ".env"
        if env_path.exists():
            _ENV_DIR = d
            with env_path.open("r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            return


def resolve_path(path: str | os.PathLike | None) -> Optional[Path]:
    if path is None:
        return None
    load_env()
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if p.is_absolute():
        return p
    if _ENV_DIR is not None:
        return (_ENV_DIR / p).resolve()
    return p.resolve()


def env_path(key: str, default: str | None = None, required: bool = False) -> Path | None:
    load_env()
    val = os.environ.get(key, default)
    if val is None:
        if required:
            raise RuntimeError(f"{key} is not set. Put it in repo-root .env or pass the CLI argument.")
        return None
    return resolve_path(val)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = resolve_path(path)
    assert p is not None
    p.mkdir(parents=True, exist_ok=True)
    return p
