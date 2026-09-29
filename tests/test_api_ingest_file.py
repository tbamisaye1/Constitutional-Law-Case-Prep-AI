"""Serve PDFs saved under uploads/ during /ingest/pdf for Ask AI cite jumps.

These do not need Postgres; they only hit the ingest file route and a stubbed
uploads directory.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_get_uploaded_pdf_serves_bytes(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    pdf_path = uploads / "06b_United_States_v_USDC_Keith.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nfake keith opinion")

    class FakeSettings:
        uploads_dir = uploads

    monkeypatch.setattr("app.api.ingest.get_settings", lambda: FakeSettings())

    with TestClient(app) as client:
        response = client.get("/ingest/file/06b_United_States_v_USDC_Keith.pdf")

    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
    assert "pdf" in response.headers.get("content-type", "").lower()


def test_get_uploaded_pdf_404_when_missing(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    class FakeSettings:
        uploads_dir = uploads

    monkeypatch.setattr("app.api.ingest.get_settings", lambda: FakeSettings())

    with TestClient(app) as client:
        response = client.get("/ingest/file/missing.pdf")

    assert response.status_code == 404


def test_get_uploaded_pdf_rejects_path_traversal(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    class FakeSettings:
        uploads_dir = uploads

    monkeypatch.setattr("app.api.ingest.get_settings", lambda: FakeSettings())

    with TestClient(app) as client:
        response = client.get("/ingest/file/../../etc/passwd")

    assert response.status_code in (400, 404, 422)
