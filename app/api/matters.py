"""
Matters = one moot problem set (e.g. Bronner 2026-27).

The Bronner seed stays in code because every workspace starts from the same
problem and the frontend needs something real before anyone has synced. A
workspace can then edit that matter or add its own, which arrives through
/sync and overrides the seed by id.

Reading matters does not require a workspace key. Without one you get the seed,
which is what an older client or a plain curl should see.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends

from app.api.deps import optional_workspace_id
from app.db.connection import DatabaseNotConfigured, db_connection
from app.db.repository import to_epoch_ms

router = APIRouter(prefix="/matters", tags=["matters"])

SEED = [
    {
        "id": "bronner-2026",
        "title": "Bobby Bronner v. United States",
        "season": "AMCA 2026–27",
        "issues": [
            {"id": "q1", "label": "Fourth Amendment", "short": "Pole camera / search"},
            {"id": "q2", "label": "Article II", "short": "Detention authority"},
        ],
    }
]


def _workspace_matters(workspace_id: str) -> list[dict]:
    """
    Matters this workspace has saved.

    Returns an empty list when the database is unreachable. Matters are
    reference data with a usable fallback, so a database blip should degrade to
    the seed rather than break the page.
    """
    try:
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, title, season, issues, updated_at
                    FROM matters
                    WHERE workspace_id = %s AND deleted_at IS NULL
                    ORDER BY updated_at DESC
                    """,
                    (workspace_id,),
                )
                return [
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "season": row["season"],
                        "issues": row["issues"],
                        "updatedAt": to_epoch_ms(row["updated_at"]),
                    }
                    for row in cursor.fetchall()
                ]
    except (DatabaseNotConfigured, psycopg.Error):
        return []


def _merged_matters(workspace_id: str | None) -> list[dict]:
    """Seed matters with any workspace copy of the same id taking precedence."""
    if workspace_id is None:
        return list(SEED)

    saved = _workspace_matters(workspace_id)
    by_id = {matter["id"]: matter for matter in SEED}
    for matter in saved:
        by_id[matter["id"]] = matter
    return list(by_id.values())


@router.get("")
def list_matters(workspace_id: str | None = Depends(optional_workspace_id)):
    return {"matters": _merged_matters(workspace_id)}


@router.get("/{matter_id}")
def get_matter(matter_id: str, workspace_id: str | None = Depends(optional_workspace_id)):
    for matter in _merged_matters(workspace_id):
        if matter["id"] == matter_id:
            return matter
    return {"error": "not_found", "matter_id": matter_id}
