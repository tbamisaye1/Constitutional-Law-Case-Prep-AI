"""
LOCAL DEMO ONLY — ask questions against the bootstrapped moot-court index.

Usage:
  python demo/bootstrap_moot_index.py   # once
  python demo/chat_cli.py "What test did Youngstown use for executive power?"
  python demo/chat_cli.py               # interactive
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage

from app.agents.prep_graph import prep_graph


def ask(question: str) -> None:
    result = prep_graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "matter_id": "moot-demo",
            "evidence": [],
        }
    )
    last = result["messages"][-1]
    text = getattr(last, "content", str(last))
    print("\n--- reply ---")
    print(text)
    print("\n--- grounding ---")
    print("status:", result.get("grounding_status"))
    print("notes:", result.get("grounding_notes", ""))
    evidence = result.get("evidence") or []
    if evidence:
        print("\n--- evidence (top chunks) ---")
        for e in evidence[:3]:
            page = f" p.{e['page']}" if e.get("page") else ""
            preview = (e.get("text") or "")[:200].replace("\n", " ")
            print(f"  [{e['id']}] {e['source']}{page}: {preview}...")


def main() -> None:
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return
    print("Moot Court RAG demo (type 'quit' to exit)")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"quit", "exit", "q"}:
            break
        ask(q)


if __name__ == "__main__":
    main()
