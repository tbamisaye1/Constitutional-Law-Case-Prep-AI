"""DB helpers for MCP tools (own short connections, thread-offloaded)."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

import anyio

from app.db.connection import DatabaseNotConfigured, db_connection
from app.mcp.errors import McpToolError, error_result

T = TypeVar("T")


def run_db(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Open a dict-row connection, run fn(cursor, …), commit on success."""
    try:
        with db_connection() as connection:
            with connection.cursor() as cursor:
                return fn(cursor, *args, **kwargs)
    except DatabaseNotConfigured as exc:
        raise McpToolError(
            "upstream_unavailable",
            "Database is not configured.",
            {"detail": str(exc)},
        ) from exc


async def run_db_async(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run blocking DB work off the event loop."""
    return await anyio.to_thread.run_sync(lambda: run_db(fn, *args, **kwargs))


async def run_blocking(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run any blocking work (FAISS, LLM, PDF) off the event loop."""
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


def clamp_limit(limit: int | None, default: int = 50, maximum: int = 200) -> int:
    if limit is None:
        return default
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(value, maximum))


def tool_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Convert McpToolError to structured error dicts.

    Must preserve the original signature. FastMCP builds the tool input schema
    from the callable it receives; a bare `*args, **kwargs` wrapper publishes
    those names as required inputs and every client call fails.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except McpToolError as exc:
            return exc.as_dict()
        except Exception as exc:  # pragma: no cover - unexpected
            return error_result(
                "upstream_unavailable",
                str(exc) or exc.__class__.__name__,
            )

    # FastMCP reads __signature__ directly in some paths; set it explicitly.
    wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return wrapper
