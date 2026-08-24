# Demo manifest — what Cursor added (safe to delete)

**Purpose:** Erin / portfolio screen-share mockup. **Not** the core Case-Law-Agent product.

When you want to rebuild the RAG agent yourself for practice, run:

```bash
./demo/remove_everything.sh
```

That removes everything in this manifest and restores `app/main.py` to pre-demo state.

---

## Files to delete (all under `demo/`)

| File | What it does |
|------|----------------|
| `bootstrap_moot_index.py` | Indexes 5 case summaries into FAISS |
| `chat_cli.py` | Terminal Q&A against the index |
| `teardown_demo.py` | Deletes `data/faiss_index/` + cached JSON |
| `index.html` | Polished `/demo` web viewer for screen-share |
| `remove_everything.sh` | One-command full removal (this script too) |
| `README.md`, `MANIFEST.md` | Docs |
| `supreme_court_cases.json` | Cached download (gitignored) |

## Code touched outside `demo/`

| Location | What to remove |
|----------|----------------|
| `app/main.py` | Block between `# >>> DEMO_START` and `# >>> DEMO_END` |
| `.gitignore` | Line `demo/supreme_court_cases.json` (optional) |

## Generated data (gitignored)

| Path | What |
|------|------|
| `data/faiss_index/` | FAISS vector index from bootstrap |

---

## What this does NOT remove (your real project)

Keep these if you still want the Case-Law-Agent scaffold to build on:

- `app/agents/` — LangGraph retrieve → reason → verify
- `app/api/chat.py` — `/chat` endpoint
- `app/rag/` — FAISS store, PDF ingest, chunking
- `Constitutional-Law-Case-Prep/src/pages/AgentPage.jsx` — frontend agent room

Those existed before the Erin demo. Delete them separately only if you want a blank slate for the whole agent.

---

## After removal — rebuild path

1. Read `docs/ARCHITECTURE.md` and `docs/TODO.md` in the parent `Case-Law-Agent/` folder.
2. Start with `app/rag/ingest_pdf.py` + `app/rag/store.py` (index one PDF).
3. Wire `app/agents/nodes/retrieve.py` → `reason.py` → `verify_claims.py`.
4. Expose via `app/api/chat.py`; hook up `AgentPage.jsx` or your own UI.
