"""Client-token minting for direct browser uploads to Vercel Blob."""

from __future__ import annotations

import base64
import json

import pytest

from app.storage import blob_client


def test_generate_client_token_shape(monkeypatch):
    monkeypatch.setenv(
        "BLOB_READ_WRITE_TOKEN",
        "vercel_blob_rw_store_testhost_secretsegment",
    )

    token = blob_client.generate_client_token(
        "case-law-agent/workspaces/w-1/doc-1/abc.pdf",
        allowed_content_types=["application/pdf"],
        maximum_size_in_bytes=50 * 1024 * 1024,
    )

    assert token.startswith("vercel_blob_client_")
    store_and_sealed = token[len("vercel_blob_client_") :]
    store_id, sealed = store_and_sealed.split("_", 1)
    assert store_id == "store"
    signature, payload_b64 = base64.b64decode(sealed).decode("utf-8").split(".", 1)
    payload = json.loads(base64.b64decode(payload_b64))

    assert len(signature) == 64
    assert payload["pathname"].endswith("/abc.pdf")
    assert payload["allowedContentTypes"] == ["application/pdf"]
    assert payload["maximumSizeInBytes"] == 50 * 1024 * 1024
    assert payload["addRandomSuffix"] is False


def test_generate_client_token_requires_read_write_token(monkeypatch):
    monkeypatch.delenv("BLOB_READ_WRITE_TOKEN", raising=False)
    monkeypatch.setattr(blob_client, "_read_write_token", lambda: "")

    with pytest.raises(RuntimeError, match="BLOB_READ_WRITE_TOKEN"):
        blob_client.generate_client_token("case-law-agent/workspaces/w/doc/x.pdf")
