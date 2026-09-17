"""
The HTTP shape of the Vercel Blob calls.

There is no official Python SDK, so these requests are hand-rolled and the
only thing keeping them honest is a test that asserts the method and path.
The delete path in particular was wrong for a while: sending HTTP DELETE to a
blob URL answers 405, and because the only caller swallowed the error, the
FAISS index cleanup failed quietly.

httpx is intercepted with a mock transport, so nothing here touches the real
store or needs credentials beyond a fake token.
"""

from __future__ import annotations

import httpx
import pytest

from app.storage import blob_client

PATHNAME = "case-law-agent/workspaces/w-1/documents/doc.pdf"
BLOB_URL = f"https://store.public.blob.vercel-storage.com/{PATHNAME}"


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    monkeypatch.setattr(blob_client, "_read_write_token", lambda: "vercel_blob_rw_store_faketoken")
    monkeypatch.setattr(blob_client, "store_slug", lambda: "store")


@pytest.fixture
def recorder(monkeypatch):
    """Capture every request httpx would have sent and script the replies."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)

        if request.method == "GET" and request.url.host == "blob.vercel-storage.com":
            return httpx.Response(200, json={"blobs": [{"pathname": PATHNAME, "url": BLOB_URL}]})
        if request.method == "POST" and request.url.path == "/delete":
            return httpx.Response(200, json=None)
        if request.method == "PUT":
            return httpx.Response(200, json={"url": BLOB_URL, "pathname": PATHNAME})
        if request.method == "DELETE":
            # What the real API answers, so a regression fails loudly here.
            return httpx.Response(405, text="Method Not Allowed")
        return httpx.Response(200, content=b"%PDF-1.7 bytes")

    original_client = httpx.Client

    def client_with_mock(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(blob_client.httpx, "Client", client_with_mock)
    return calls


def test_delete_posts_to_the_delete_endpoint(recorder):
    blob_client.delete_blob(PATHNAME)

    methods = [(call.method, call.url.path) for call in recorder]

    assert ("POST", "/delete") in methods
    assert not any(method == "DELETE" for method, _ in methods), (
        "HTTP DELETE on a blob URL answers 405; deletion goes through POST /delete"
    )


def test_delete_sends_the_blob_url_in_the_body(recorder):
    import json

    blob_client.delete_blob(PATHNAME)

    post = next(call for call in recorder if call.method == "POST")
    assert json.loads(post.content) == {"urls": [BLOB_URL]}


def test_delete_of_a_missing_blob_is_a_no_op(monkeypatch):
    """Deleting twice must not raise; the second call has nothing to find."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"blobs": []})

    original_client = httpx.Client
    monkeypatch.setattr(
        blob_client.httpx,
        "Client",
        lambda *args, **kwargs: original_client(*args, **{**kwargs, "transport": httpx.MockTransport(handler)}),
    )

    blob_client.delete_blob(PATHNAME)


def test_put_returns_the_url_for_storing_against_the_document(recorder):
    stored = blob_client.put_blob(PATHNAME, b"%PDF-1.7", content_type="application/pdf")

    assert stored["url"] == BLOB_URL

    put = next(call for call in recorder if call.method == "PUT")
    assert put.headers["x-content-type"] == "application/pdf"
    # Blob's own random suffix stays off: the caller owns the whole pathname.
    assert put.headers["x-add-random-suffix"] == "0"


def test_exact_match_wins_over_a_prefix_collision(recorder):
    """
    A delete must never act on a near-miss.

    The list API is queried by prefix, so a pathname that is a prefix of
    another would otherwise be able to delete the wrong object.
    """
    with httpx.Client() as client:
        match = blob_client._find_blob(
            client,
            {"authorization": "Bearer x", "x-api-version": "7"},
            "case-law-agent/workspaces/w-1/documents/do",
            exact_only=True,
        )

    assert match is None
