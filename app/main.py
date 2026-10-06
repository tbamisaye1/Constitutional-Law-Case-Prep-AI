"""
FastAPI entrypoint.

CORS is open in local/dev so Vite (5173) can call us directly if the proxy
is skipped. Tighten origins before any shared deploy.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api import backups, chat, documents, export, health, ingest, matters, sync
from app.mcp.auth import McpAuth, SlashlessMcpMiddleware
from app.mcp.server import mcp

# >>> DEMO_START — Erin screen-share mockup; remove with: ./demo/remove_everything.sh
_DEMO_HTML = Path(__file__).resolve().parents[1] / "demo" / "index.html"
# >>> DEMO_END

# Build the Streamable HTTP app before lifespan so session_manager exists.
_mcp_http_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="Constitutional Law Case Prep AI",
    version="0.1.0",
    description="Agentic prep backend for YUMC / AMCA. Author: Tobi Bamisaye.",
    lifespan=lifespan,
)

# >>> DEMO_START — widen CORS for Vercel preview/prod; remove with ./demo/remove_everything.sh
_cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
# Allow any *.vercel.app deployment (Erin demo link)
_cors_origin_regex = r"https://.*\.vercel\.app"
# >>> DEMO_END

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Outermost: stop Starlette Mount from 307-redirecting /mcp → /mcp/.
app.add_middleware(SlashlessMcpMiddleware)

app.include_router(health.router)
app.include_router(matters.router)
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(export.router)
app.include_router(sync.router)
app.include_router(backups.router)
app.include_router(documents.router)

# Remote MCP (Streamable HTTP). Auth never falls open when MCP_TOKEN is unset.
# Mounted at both /mcp and /mcp/ so clients that omit/require the trailing slash
# both work without a 307 that some MCP clients will not follow.
_mcp_authed = McpAuth(_mcp_http_app)
app.mount("/mcp", _mcp_authed)


# >>> DEMO_START
@app.get("/")
def demo_root():
    """Erin demo: send visitors straight to the Q&A viewer."""
    if _DEMO_HTML.is_file():
        return FileResponse(_DEMO_HTML)
    return {"detail": "demo/index.html missing", "health": "/health", "chat": "POST /chat"}


@app.get("/demo")
def demo_viewer():
    """LOCAL DEMO ONLY — polished viewer for screen-share. Remove via ./demo/remove_everything.sh"""
    if not _DEMO_HTML.is_file():
        return {"detail": "demo/index.html missing"}
    return FileResponse(_DEMO_HTML)
# >>> DEMO_END
