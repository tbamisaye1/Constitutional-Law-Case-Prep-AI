"""Client-compatible id helpers (match frontend newId / annotation ids)."""

from __future__ import annotations

import secrets
import time


def new_id(prefix: str) -> str:
    """Match `argumentsBoard.js` newId: `{prefix}-{epochms}-{base36}`. """
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    suffix = "".join(secrets.choice(alphabet) for _ in range(5))
    return f"{prefix}-{int(time.time() * 1000)}-{suffix}"


def new_annotation_id() -> str:
    """Match library annotation ids: `a-<epochms>-<rand>`."""
    return new_id("a")
