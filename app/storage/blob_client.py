"""
Vercel Blob REST client.

There is no official Python SDK, so this speaks the same HTTP API that
@vercel/blob does. Extracted from blob_index.py, which now uses it, because
uploaded PDFs need the same put/get/delete primitives as the FAISS index.

Auth works two ways. A BLOB_READ_WRITE_TOKEN is what `vercel env pull` gives
you locally. In a deployed function there may be no such token, only a
VERCEL_OIDC_TOKEN plus the store id, so both paths are supported.
"""

from __future__ import annotations

import os

import httpx

from app.config import get_settings

_API_VERSION = "7"
_BLOB_BASE = "https://blob.vercel-storage.com"
_LIST_URL = "https://blob.vercel-storage.com"
_TIMEOUT = 120.0


def _read_write_token() -> str:
    """
    The read-write token from the environment, or from .env.local via Settings.

    Deployed functions get it injected as a real environment variable. Locally
    it only exists in the file `vercel env pull` wrote, which is why Settings
    is the fallback rather than the other way round.
    """
    from_env = os.environ.get("BLOB_READ_WRITE_TOKEN", "").strip()
    if from_env:
        return from_env
    return get_settings().blob_read_write_token.strip()


def blob_configured() -> bool:
    """True when this process has credentials to reach the Blob store."""
    if _read_write_token():
        return True
    oidc = os.environ.get("VERCEL_OIDC_TOKEN", "").strip()
    store = os.environ.get("BLOB_STORE_ID", "").strip()
    return bool(oidc and store)


def auth_token() -> str:
    """
    Bearer token for the Blob API.

    Raises:
        RuntimeError: when neither credential is present.
    """
    read_write = _read_write_token()
    if read_write:
        return read_write
    oidc = os.environ.get("VERCEL_OIDC_TOKEN", "").strip()
    if oidc:
        return oidc
    raise RuntimeError("Blob credentials missing (BLOB_READ_WRITE_TOKEN or VERCEL_OIDC_TOKEN)")


def store_slug() -> str | None:
    """
    Store id used to build the public hostname, lowercased and unprefixed.

    Falls back to reading it out of the read-write token, where it sits in the
    fourth underscore-separated segment.
    """
    store_id = os.environ.get("BLOB_STORE_ID", "").strip()
    if not store_id:
        parts = auth_token().split("_")
        if len(parts) >= 4:
            store_id = parts[3]
    if not store_id:
        return None
    return store_id.removeprefix("store_").lower()


def put_blob(pathname: str, data: bytes, content_type: str) -> dict:
    """
    Write bytes at an exact pathname, replacing whatever was there.

    Blob's own random suffix is disabled so the pathname stays exactly what the
    caller asked for. Callers that need an unguessable URL must put the random
    part in the pathname themselves.

    Returns:
        Blob's response, which carries 'url' and 'downloadUrl' for the object.

    Raises:
        httpx.HTTPStatusError: when Blob rejects the upload.
    """
    headers = {
        "authorization": f"Bearer {auth_token()}",
        "x-api-version": _API_VERSION,
        "x-content-type": content_type,
        "x-content-length": str(len(data)),
        "x-add-random-suffix": "0",
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.put(f"{_BLOB_BASE}/{pathname}", content=data, headers=headers)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            # Older API versions answered with an empty body. The upload still
            # succeeded, so fall back to the derived public URL.
            slug = store_slug()
            url = f"https://{slug}.public.blob.vercel-storage.com/{pathname}" if slug else ""
            return {"url": url, "pathname": pathname}


def get_blob(pathname: str) -> bytes | None:
    """
    Read bytes back, or None when the pathname does not exist.

    Tries the public store hostname first because it is a plain CDN GET with no
    list call. Falls back to the authenticated list-then-download path, which
    also covers stores that are not publicly readable.
    """
    slug = store_slug()
    if slug:
        public_url = f"https://{slug}.public.blob.vercel-storage.com/{pathname}"
        try:
            with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
                response = client.get(public_url)
                if response.status_code == 200 and response.content:
                    return response.content
        except httpx.HTTPError:
            # Fall through to the authenticated path below.
            pass

    if not blob_configured():
        return None

    headers = {
        "authorization": f"Bearer {auth_token()}",
        "x-api-version": _API_VERSION,
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        match = _find_blob(client, headers, pathname)
        if match is None:
            return None
        download_url = match.get("downloadUrl") or match.get("url")
        if not download_url:
            return None
        response = client.get(download_url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content


def delete_blob(pathname: str) -> None:
    """
    Delete a blob, treating an already-missing pathname as success.

    Deletion is a POST to /delete carrying the object's URL. Sending an HTTP
    DELETE to the blob URL itself answers 405, which is how this was written
    originally and why removing the last FAISS source used to fail silently.

    Note that the object stays readable from the CDN for a short while after
    this returns. Use the list API, not a GET, to assert it is gone.
    """
    if not blob_configured():
        return

    headers = {
        "authorization": f"Bearer {auth_token()}",
        "x-api-version": _API_VERSION,
        "content-type": "application/json",
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        match = _find_blob(client, headers, pathname, exact_only=True)
        if match is None:
            return
        blob_url = match.get("url") or match.get("downloadUrl")
        if not blob_url:
            return
        response = client.post(
            f"{_BLOB_BASE}/delete",
            headers=headers,
            json={"urls": [blob_url]},
        )
        if response.status_code != 404:
            response.raise_for_status()


def _find_blob(
    client: httpx.Client,
    headers: dict[str, str],
    pathname: str,
    exact_only: bool = False,
) -> dict | None:
    """
    Look up one blob by pathname via the list API.

    Args:
        exact_only: When True, only an exact pathname match counts. Deletes use
            this so a prefix collision can never remove the wrong object. Reads
            leave it False and accept the first prefix match, which is what the
            FAISS index path has always relied on.
    """
    listed = client.get(_LIST_URL, params={"prefix": pathname, "limit": "1000"}, headers=headers)
    if listed.status_code in (400, 404):
        return None
    listed.raise_for_status()

    blobs = listed.json().get("blobs") or []
    exact = next((blob for blob in blobs if blob.get("pathname") == pathname), None)
    if exact or exact_only:
        return exact
    return blobs[0] if blobs else None
