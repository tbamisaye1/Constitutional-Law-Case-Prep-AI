# Local demo (do not push)

Erin / portfolio RAG mockup. **Tobi pushes this repo himself.**

## Turn on

```bash
cd Constitutional-Law-Case-Prep-AI
source .venv/bin/activate
python demo/bootstrap_moot_index.py
uvicorn app.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/demo**

## Delete everything and rebuild yourself later

One command:

```bash
./demo/remove_everything.sh
```

That removes:
- All files in `demo/` (except `DEMO_REMOVED.txt` stub)
- The `/demo` route and demo imports in `app/main.py`
- `data/faiss_index/` and cached case JSON

**Keeps** the core scaffold (`app/agents/`, `app/rag/`, `/chat`) so you can rebuild the agent as practice.

Full file list: `demo/MANIFEST.md`

## Light teardown (data only, keep scripts)

```bash
python demo/teardown_demo.py
```

Clears the FAISS index but leaves demo scripts and `/demo` route.
