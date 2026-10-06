"""
ASGI auth wrapper for the Streamable HTTP MCP app.

Two clients:
- Claude Code / Cursor / scripts: Authorization: Bearer <MCP_TOKEN>
- claude.ai custom connector: secret URL /mcp/k/<MCP_TOKEN>

If MCP_TOKEN is unset, every request returns 503 (never fall open).

TODO(oauth): replace the secret-URL approach with OAuth for claude.ai
connectors when the product path is ready. The bearer header stays for
local tools either way.
"""

from __future__ import annotations

import hmac
import re
from typing import Any

from app.config import get_settings

# Starlette Mount may leave the full "/mcp/..." path (root_path="/mcp") or
# strip it to "/k/...". Accept both.
_SECRET_PATH_RE = re.compile(r"^(?:/mcp)?/k/([^/]+)/?$")


def _configured_token() -> str:
    return (get_settings().mcp_token or "").strip()


def _tokens_match(provided: str, expected: str) -> bool:
    if not provided or not expected:
        return False
    if len(provided) != len(expected):
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _json_response(send, status: int, body: bytes) -> Any:
    async def _send_response() -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return _send_response()


def _normalize_mcp_path(path: str) -> str:
    """Strip a leading /mcp so the Streamable HTTP app sees '/' or '/…'."""
    if path == "/mcp":
        return "/"
    if path.startswith("/mcp/"):
        rest = path[4:]
        return rest if rest.startswith("/") else f"/{rest}"
    return path or "/"


class McpAuth:
    """Wrap an ASGI app; require bearer or secret-URL token before forwarding."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        expected = _configured_token()
        if not expected:
            await _json_response(
                send,
                503,
                b'{"error":{"code":"upstream_unavailable","message":"MCP_TOKEN is not configured."}}',
            )
            return

        path = scope.get("path") or ""
        secret_match = _SECRET_PATH_RE.match(path)
        provided = ""
        if secret_match:
            provided = secret_match.group(1)
        else:
            headers = {
                k.decode("latin1").lower(): v.decode("latin1")
                for k, v in scope.get("headers") or []
            }
            auth = headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                provided = auth[7:].strip()

        if not _tokens_match(provided, expected):
            await _json_response(
                send,
                401,
                b'{"error":{"code":"forbidden","message":"Invalid or missing MCP token."}}',
            )
            return

        # Rewrite path for the Streamable HTTP app (root is "/").
        scope = dict(scope)
        scope["path"] = _normalize_mcp_path(path)
        # Secret URL: never leave the raw token in scope for downstream logging.
        if secret_match:
            scope["path"] = "/"
        if scope["path"] in ("", None):
            scope["path"] = "/"
        scope["raw_path"] = scope["path"].encode("utf-8")

        await self.app(scope, receive, send)


class SlashlessMcpMiddleware:
    """
    Rewrite POST/GET /mcp → /mcp/ before Starlette Mount issues a 307.

    Some MCP clients will not follow that redirect.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope.get("path") == "/mcp":
            scope = dict(scope)
            scope["path"] = "/mcp/"
            scope["raw_path"] = b"/mcp/"
        await self.app(scope, receive, send)
