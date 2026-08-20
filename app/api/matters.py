"""
Matters = one moot problem set (e.g. Bronner 2026–27).

Hard-coded seed for now so the frontend has something real to show.
Later: SQLite / Postgres.
"""

from fastapi import APIRouter

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


@router.get("")
def list_matters():
    return {"matters": SEED}


@router.get("/{matter_id}")
def get_matter(matter_id: str):
    for m in SEED:
        if m["id"] == matter_id:
            return m
    return {"error": "not_found", "matter_id": matter_id}
