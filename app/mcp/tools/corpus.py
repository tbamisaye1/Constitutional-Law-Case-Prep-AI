"""FAISS corpus, PDF page text, and prep agent tools."""

from __future__ import annotations

import fnmatch
import io
from typing import Any
from urllib.parse import urlparse

import httpx
from mcp.types import ToolAnnotations
from pypdf import PdfReader

from app.agents.run_chat import run_prep_chat
from app.api.ingest import _index_pdf_bytes
from app.config import get_settings
from app.mcp.dbutil import run_blocking, tool_guard
from app.mcp.errors import McpToolError
from app.mcp.server import mcp
from app.rag.store import is_instant_case_source, list_index_sources, load_store
from app.storage.ingest_files import load_uploaded_pdf


def _parse_pages(pages: Any) -> list[int]:
    if isinstance(pages, int):
        return [pages]
    if isinstance(pages, list):
        out = []
        for item in pages:
            out.extend(_parse_pages(item))
        return out
    if isinstance(pages, str):
        text = pages.strip()
        if "-" in text and "," not in text:
            start_s, end_s = text.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                start, end = end, start
            return list(range(start, end + 1))
        if "," in text:
            return [int(p.strip()) for p in text.split(",") if p.strip()]
        return [int(text)]
    raise McpToolError("invalid", "pages must be an int, range string, or list.")


def _host_allowed(host: str) -> bool:
    settings = get_settings()
    raw = (settings.mcp_ingest_allowed_hosts or "").strip()
    if not raw:
        return False
    host = host.lower()
    for pattern in raw.split(","):
        pattern = pattern.strip().lower()
        if not pattern:
            continue
        if pattern.startswith("*."):
            suffix = pattern[1:]  # .example.com
            if host.endswith(suffix) or host == pattern[2:]:
                return True
        elif fnmatch.fnmatch(host, pattern) or host == pattern:
            return True
    return False


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_sources() -> dict[str, Any]:
    """List FAISS index sources; mark the Instant Case PDF(s)."""

    def _sync() -> dict[str, Any]:
        sources = list_index_sources()
        for row in sources:
            row["instant_case"] = is_instant_case_source(row.get("source") or "")
        return {"sources": sources}

    return await run_blocking(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def search_corpus(
    query: str,
    k: int = 8,
    source: str | None = None,
) -> dict[str, Any]:
    """FAISS similarity search only (no LLM). Returns chunks with scores."""

    def _sync() -> dict[str, Any]:
        store = load_store()
        if store is None:
            raise McpToolError(
                "upstream_unavailable",
                "FAISS index is not available.",
            )
        pairs = store.similarity_search_with_score(query, k=max(1, min(int(k), 40)))
        chunks = []
        for doc, score in pairs:
            meta = doc.metadata or {}
            src = str(meta.get("source") or "unknown")
            if source and source.lower() not in src.lower():
                continue
            chunks.append(
                {
                    "source": src,
                    "page": meta.get("page"),
                    "text": doc.page_content or "",
                    "score": float(score),
                }
            )
        return {"chunks": chunks[: max(1, min(int(k), 40))]}

    return await run_blocking(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def read_source_pages(
    source: str,
    pages: Any,
) -> dict[str, Any]:
    """Exact text of PDF pages (cap 10). Uses the uploaded PDF via Blob/local."""

    def _sync() -> dict[str, Any]:
        page_nums = _parse_pages(pages)
        if not page_nums:
            raise McpToolError("invalid", "No pages requested.")
        if len(page_nums) > 10:
            raise McpToolError("invalid", "Cap is 10 pages per call.")
        filename = source.split(" — ")[-1].strip() if " — " in source else source
        filename = filename.split("/")[-1]
        content = load_uploaded_pdf(filename)
        if content is None:
            # Fall back to local uploads dir.
            settings = get_settings()
            path = settings.uploads_dir / filename
            if path.is_file():
                content = path.read_bytes()
        if content is None:
            raise McpToolError(
                "not_found",
                "PDF bytes not found for this source.",
                {"source": source, "filename": filename},
            )
        reader = PdfReader(io.BytesIO(content))
        out = []
        for page_n in page_nums:
            idx = page_n - 1
            if idx < 0 or idx >= len(reader.pages):
                out.append({"page": page_n, "error": "out_of_range"})
                continue
            text = reader.pages[idx].extract_text() or ""
            out.append({"page": page_n, "text": text})
        return {"source": source, "pages": out}

    return await run_blocking(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def ask_prep_agent(
    message: str,
    matter_id: str | None = None,
    grounding_source: str = "documents",
    model_tier: str = "standard",
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Thin wrapper over the same prep graph used by POST /chat."""

    def _sync() -> dict[str, Any]:
        if grounding_source not in ("documents", "web_plus"):
            raise McpToolError("invalid", "grounding_source must be documents or web_plus.")
        if model_tier not in ("standard", "advanced"):
            raise McpToolError("invalid", "model_tier must be standard or advanced.")
        return run_prep_chat(
            message,
            matter_id=matter_id or "",
            grounding_source=grounding_source,  # type: ignore[arg-type]
            model_tier=model_tier,  # type: ignore[arg-type]
            history=history,
        )

    return await run_blocking(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def ingest_pdf_from_url(url: str, filename: str) -> dict[str, Any]:
    """
    Fetch a PDF from an allowlisted host and index it into FAISS.
    50 MB cap. Hosts from MCP_INGEST_ALLOWED_HOSTS.
    """

    def _sync() -> dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme not in ("https", "http"):
            raise McpToolError("invalid", "url must be http(s).")
        host = (parsed.hostname or "").lower()
        if not _host_allowed(host):
            raise McpToolError(
                "forbidden",
                "Host is not in MCP_INGEST_ALLOWED_HOSTS.",
                {"host": host},
            )
        safe_name = filename.split("/")[-1]
        if not safe_name.lower().endswith(".pdf"):
            safe_name = f"{safe_name}.pdf"
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                content = response.content
        except Exception as exc:
            raise McpToolError(
                "upstream_unavailable",
                f"Failed to fetch PDF: {exc}",
            ) from exc
        if len(content) > 50 * 1024 * 1024:
            raise McpToolError("invalid", "PDF exceeds 50 MB cap.")
        if not content.startswith(b"%PDF"):
            raise McpToolError("invalid", "Response does not look like a PDF.")
        try:
            result = _index_pdf_bytes(content, safe_name)
        except Exception as exc:
            raise McpToolError(
                "upstream_unavailable",
                f"Index failed: {exc}",
            ) from exc
        return {"filename": safe_name, "result": result}

    return await run_blocking(_sync)
