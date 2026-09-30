"""FAISS hydrate must not wipe production Blob with the bundled demo index."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.storage import blob_index


def test_hydrate_uses_blob_when_present(tmp_path, monkeypatch):
    faiss_dir = tmp_path / "faiss"
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "index.faiss").write_bytes(b"bundled")

    zipped = blob_index._zip_dir(bundled)
    # Replace payload with a distinct index so we can tell Blob won.
    real = tmp_path / "real"
    real.mkdir()
    (real / "index.faiss").write_bytes(b"from-blob")
    zipped = blob_index._zip_dir(real)

    monkeypatch.setattr(blob_index, "blob_configured", lambda: True)
    monkeypatch.setattr(blob_index, "get_blob", lambda path: zipped)

    assert blob_index.hydrate_faiss_index(faiss_dir, bundled) is True
    assert (faiss_dir / "index.faiss").read_bytes() == b"from-blob"


def test_hydrate_raises_when_blob_errors_instead_of_falling_back(tmp_path, monkeypatch):
    faiss_dir = tmp_path / "faiss"
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "index.faiss").write_bytes(b"bundled")

    monkeypatch.setattr(blob_index, "blob_configured", lambda: True)

    def boom(_pathname):
        raise RuntimeError("blob down")

    monkeypatch.setattr(blob_index, "get_blob", boom)

    with pytest.raises(RuntimeError, match="hydrate FAISS from Blob"):
        blob_index.hydrate_faiss_index(faiss_dir, bundled)

    assert not (faiss_dir / "index.faiss").exists()


def test_hydrate_falls_back_to_bundled_when_blob_empty(tmp_path, monkeypatch):
    faiss_dir = tmp_path / "faiss"
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "index.faiss").write_bytes(b"bundled")

    monkeypatch.setattr(blob_index, "blob_configured", lambda: True)
    monkeypatch.setattr(blob_index, "get_blob", lambda path: None)

    assert blob_index.hydrate_faiss_index(faiss_dir, bundled) is True
    assert (faiss_dir / "index.faiss").read_bytes() == b"bundled"
