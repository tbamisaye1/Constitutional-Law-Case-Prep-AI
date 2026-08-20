"""Export endpoint: turn a note into a downloadable Word doc."""

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.export.to_docx import note_to_docx

router = APIRouter(prefix="/export", tags=["export"])


class ExportNoteRequest(BaseModel):
    title: str = Field(min_length=1)
    body: str = ""


@router.post("/note.docx")
def export_note(body: ExportNoteRequest):
    data = note_to_docx(body.title, body.body)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in body.title)[:60]
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe or "note"}.docx"'},
    )
