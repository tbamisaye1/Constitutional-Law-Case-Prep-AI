"""
Agent state for the moot prep graph.

TypedDict + add_messages so each node returns a partial update and LangGraph
merges it. Evidence + grounding fields let retrieve → reason → verify share
data without stuffing everything into message text alone.
"""

from typing import Annotated, Literal, NotRequired, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.grounding.schemas import Claim, EvidenceHit, GroundingStatus

# documents = FAISS RAG only (default). web_plus = corpus + OpenRouter web search.
GroundingSource = Literal["documents", "web_plus"]
# standard = cheap default model. advanced = stronger GPT mini for hard questions.
ModelTier = Literal["standard", "advanced"]


class PrepState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    matter_id: str
    # Ask AI mode. Omitted / documents keeps today's RAG path unchanged.
    grounding_source: NotRequired[GroundingSource]
    # Ask AI model tier from the UI Advanced responses switch.
    model_tier: NotRequired[ModelTier]
    # Filled by retrieve; reason and verify read these.
    evidence: NotRequired[list[EvidenceHit]]
    grounding_status: NotRequired[GroundingStatus]
    claims: NotRequired[list[Claim]]
    grounding_notes: NotRequired[str]
    # Prior Ask AI turns (user/assistant), excluding the current packed question.
    chat_history: NotRequired[list[dict]]
    # Optional notebook chunks the browser searched locally and sent with /chat.
    # Never persisted server-side beyond this request.
    client_notes: NotRequired[list[dict]]
