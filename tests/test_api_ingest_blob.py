"""Large-PDF ingest via Blob must not re-open the Vercel body-size trap."""

from pathlib import Path

from fastapi.testclient import TestClient


def test_proxy_ingest_rejects_oversize_before_indexing(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    class FakeSettings:
        openrouter_api_key = "sk-test"
        uploads_dir = uploads

    monkeypatch.setattr("app.api.ingest.get_settings", lambda: FakeSettings())

    from app.main import app

    client = TestClient(app)
    # Just over the 4 MB proxy cap used by /ingest/pdf.
    oversized = b"%PDF-1.4\n" + (b"x" * (4 * 1024 * 1024 + 100))
    response = client.post(
        "/ingest/pdf",
        files={"file": ("big.pdf", oversized, "application/pdf")},
    )
    assert response.status_code == 413
    assert "4.5" in response.json()["detail"] or "MB" in response.json()["detail"]


def test_from_blob_indexes_downloaded_bytes(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    pdf_path = tmp_path / "source.pdf"
    # Minimal PDF with extractable text so pypdf does not return empty chunks.
    pdf_path.write_bytes(
        b"""%PDF-1.1
1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj
2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj
3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj
4 0 obj<< /Length 44 >>stream
BT /F1 24 Tf 50 100 Td (Hello Carpenter) Tj ET
endstream
endobj
5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000361 00000 n 
trailer<< /Size 6 /Root 1 0 R >>
startxref
442
%%EOF
"""
    )

    class FakeSettings:
        openrouter_api_key = "sk-test"
        uploads_dir = uploads

    monkeypatch.setattr("app.api.ingest.get_settings", lambda: FakeSettings())
    monkeypatch.setattr("app.api.ingest.blob_configured", lambda: True)
    monkeypatch.setattr("app.api.ingest.store_slug", lambda: "teststore")
    monkeypatch.setattr(
        "app.api.ingest._is_allowed_blob_url",
        lambda url, pathname: True,
    )

    class FakeResponse:
        content = pdf_path.read_bytes()

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr("app.api.ingest.httpx.Client", FakeClient)

    def fake_ingest(path: Path, source_name: str | None = None):
        from app.rag.chunking import TextChunk

        return [TextChunk(text="Hello Carpenter", source=source_name or path.name, page=1, index=0)]

    monkeypatch.setattr("app.api.ingest.ingest_pdf", fake_ingest)
    monkeypatch.setattr("app.api.ingest.build_or_merge_store", lambda chunks: None)

    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/ingest/from-blob",
        json={
            "filename": "Carpenter.pdf",
            "blob_url": "https://teststore.public.blob.vercel-storage.com/case-law-agent/ingest/abc.pdf",
            "blob_pathname": "case-law-agent/ingest/abc.pdf",
            "size_bytes": pdf_path.stat().st_size,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["filename"] == "Carpenter.pdf"
    assert body["chunks"] == 1
    assert (uploads / "Carpenter.pdf").is_file()
