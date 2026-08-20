"""
Agent state for the moot prep graph.

TypedDict + add_messages so each node returns a partial update and LangGraph
merges it. Evidence + grounding fields let retrieve → reason → verify share
data without stuffing everything into message text alone.
"""

from typing import Annotated, NotRequired, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.grounding.schemas import Claim, EvidenceHit, GroundingStatus


class PrepState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    matter_id: str
    # Filled by retrieve; reason and verify read these.
    evidence: NotRequired[list[EvidenceHit]]
    grounding_status: NotRequired[GroundingStatus]
    claims: NotRequired[list[Claim]]
    grounding_notes: NotRequired[str]
