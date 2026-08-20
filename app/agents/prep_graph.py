"""
Prep graph with anti-hallucination order:

  START → retrieve → reason → verify_claims → END

Retrieve may return empty evidence; reason then abstains instead of guessing.
Verify is deterministic quote-checking (see app/grounding/verify.py).
"""

from langgraph.graph import END, START, StateGraph

from app.agents.nodes.reason import reason_node
from app.agents.nodes.retrieve import retrieve_node
from app.agents.nodes.verify_claims import verify_claims_node
from app.agents.state import PrepState


def build_prep_graph():
    graph = StateGraph(PrepState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("verify_claims", verify_claims_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "verify_claims")
    graph.add_edge("verify_claims", END)
    return graph.compile()


prep_graph = build_prep_graph()
