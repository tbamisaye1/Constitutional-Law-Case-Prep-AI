"""
Persist the FAISS index to Vercel Blob so uploads survive serverless cold starts.

On Vercel the app bundle is read-only, so we work in /tmp and sync index.zip to
Blob. The HTTP plumbing lives in blob_client.py; this module only decides what
to zip and when.
"""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

from app.storage.blob_client import blob_configured, delete_blob, get_blob, put_blob

BLOB_INDEX_PATH = "case-law-agent/faiss_index.zip"


def _zip_dir(source_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir).as_posix())
    return buf.getvalue()


def _unzip_to(data: bytes, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)


def copy_bundled_index(bundled_dir: Path, dest_dir: Path) -> bool:
    """Copy build-time FAISS from the deployment bundle into writable storage."""
    if not (bundled_dir / "index.faiss").exists():
        return False
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(bundled_dir, dest_dir)
    return True


def hydrate_faiss_index(faiss_dir: Path, bundled_dir: Path) -> bool:
    """
    Ensure faiss_dir has an index: try Blob download, else copy bundled build artifact.
    Returns True if an index is available.
    """
    if (faiss_dir / "index.faiss").exists():
        return True

    if blob_configured():
        try:
            data = get_blob(BLOB_INDEX_PATH)
            if data:
                _unzip_to(data, faiss_dir)
                return (faiss_dir / "index.faiss").exists()
        except Exception:
            pass

    return copy_bundled_index(bundled_dir, faiss_dir)


def persist_faiss_index(faiss_dir: Path) -> None:
    """Upload current FAISS directory to Blob (production only when configured)."""
    if not blob_configured():
        return
    if not (faiss_dir / "index.faiss").exists():
        delete_blob(BLOB_INDEX_PATH)
        return
    put_blob(BLOB_INDEX_PATH, _zip_dir(faiss_dir), content_type="application/zip")
