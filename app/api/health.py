from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok", "service": "constitutional-law-case-prep-ai", "author": "Tobi Bamisaye"}
