"""
FastAPI entrypoint.

CORS is open in local/dev so Vite (5173) can call us directly if the proxy
is skipped. Tighten origins before any shared deploy.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api import chat, export, health, ingest, matters

# >>> DEMO_START — Erin screen-share mockup; remove with: ./demo/remove_everything.sh
_DEMO_HTML = Path(__file__).resolve().parents[1] / "demo" / "index.html"
# >>> DEMO_END

app = FastAPI(
    title="Constitutional Law Case Prep AI",
    version="0.1.0",
    description="Agentic prep backend for YUMC / AMCA. Author: Tobi Bamisaye.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(matters.router)
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(export.router)

# >>> DEMO_START
@app.get("/demo")
def demo_viewer():
    """LOCAL DEMO ONLY — polished viewer for screen-share. Remove via ./demo/remove_everything.sh"""
    if not _DEMO_HTML.is_file():
        return {"detail": "demo/index.html missing"}
    return FileResponse(_DEMO_HTML)
# >>> DEMO_END
