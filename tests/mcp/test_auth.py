"""MCP auth wrapper: bearer, secret URL, unset token."""

from __future__ import annotations

import json

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.config import get_settings
from app.mcp.auth import McpAuth


async def _ok(_request):
    return JSONResponse({"ok": True})


def _make_client(token: str | None) -> TestClient:
    get_settings.cache_clear()
    inner = Starlette(routes=[Route("/", _ok, methods=["POST", "GET"])])
    app = McpAuth(inner)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_unset_token_returns_503(monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "")
    get_settings.cache_clear()
    with _make_client(None) as client:
        response = client.post("/")
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "upstream_unavailable"


def test_missing_bearer_returns_401(monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "a" * 32)
    get_settings.cache_clear()
    with _make_client("a" * 32) as client:
        response = client.post("/")
    assert response.status_code == 401


def test_wrong_bearer_returns_401(monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "a" * 32)
    get_settings.cache_clear()
    with _make_client("a" * 32) as client:
        response = client.post(
            "/",
            headers={"Authorization": "Bearer wrong-token-wrong-token-wrong"},
        )
    assert response.status_code == 401


def test_bearer_accepted(monkeypatch):
    token = "b" * 32
    monkeypatch.setenv("MCP_TOKEN", token)
    get_settings.cache_clear()
    with _make_client(token) as client:
        response = client.post(
            "/",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_secret_path_accepted(monkeypatch):
    token = "c" * 32
    monkeypatch.setenv("MCP_TOKEN", token)
    get_settings.cache_clear()
    with _make_client(token) as client:
        response = client.post(f"/k/{token}")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_secret_path_with_mcp_prefix(monkeypatch):
    """Starlette Mount may leave the full /mcp/k/... path unstripped."""
    token = "d" * 32
    monkeypatch.setenv("MCP_TOKEN", token)
    get_settings.cache_clear()
    with _make_client(token) as client:
        response = client.post(f"/mcp/k/{token}")
    assert response.status_code == 200
    assert response.json()["ok"] is True
