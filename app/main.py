"""
FastAPI entrypoint.

CORS is open in local/dev so Vite (5173) can call us directly if the proxy
is skipped. Tighten origins before any shared deploy.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, export, health, ingest, matters

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
