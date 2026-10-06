"""
FastMCP instance for Case Prep.

Mounted at /mcp with Streamable HTTP (stateless + JSON) for Vercel.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

SERVER_INSTRUCTIONS = """\
Case Prep MCP — moot-court prep data backed by Postgres (same DB as the web app).

1. Call health first. If no workspace is resolved, call find_my_data, then set_default_workspace.
2. Read a row before editing it, and pass its updated_at back as expected_updated_at.
   On a conflict error, re-read, merge, and retry; never blindly overwrite.
3. Prefer the structured Arguments tools over upsert_library_record for the arguments board.
4. Ground legal claims with search_corpus and read_source_pages before writing them into notes.
   Cite record pages as R. <page>.
5. Destructive tools: run the dry run, show the user the diff, and only then confirm.
"""

mcp = FastMCP(
    "case-prep",
    instructions=SERVER_INSTRUCTIONS,
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
)

# Tool modules register via @mcp.tool on import.
from app.mcp.tools import (  # noqa: E402, F401
    annotations,
    arguments,
    cases,
    corpus,
    discovery,
    library,
    notebook,
    notes,
    recovery,
    search,
)
