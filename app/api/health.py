import psycopg
from fastapi import APIRouter

from app.config import get_settings
from app.db.connection import DatabaseNotConfigured
from app.db.migrate import migration_status
from app.storage.blob_client import blob_configured

router = APIRouter(tags=["health"])


def _database_health() -> dict:
    """
    Whether Postgres is reachable and the schema is current.

    Pending migrations are reported rather than applied. The frontend uses this
    to explain why sync is off, and a deploy check can read it to catch a
    database that was never migrated.
    """
    settings = get_settings()
    if not settings.database_configured:
        return {"configured": False, "reachable": False, "detail": "DATABASE_URL not set"}

    try:
        status = migration_status()
    except DatabaseNotConfigured as error:
        return {"configured": False, "reachable": False, "detail": str(error)}
    except psycopg.Error as error:
        return {"configured": True, "reachable": False, "detail": str(error).strip()}

    return {
        "configured": True,
        "reachable": True,
        "migrationsPending": status["pending"],
    }


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "constitutional-law-case-prep-ai",
        "author": "Tobi Bamisaye",
        "database": _database_health(),
        "blobStorage": {"configured": blob_configured()},
    }


@router.get("/health/db")
def database_health():
    """Database-only probe, cheap enough to poll and safe to alert on."""
    detail = _database_health()
    return {"status": "ok" if detail.get("reachable") else "degraded", "database": detail}
