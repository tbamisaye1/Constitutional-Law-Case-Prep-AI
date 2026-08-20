# Constitutional Law Case Prep AI (backend)

**Author:** Tobi Bamisaye

Python FastAPI backend for moot prep: PDF ingest, FAISS RAG, LangGraph prep agent, Word export. Models via OpenRouter (OpenAI-compatible API).

Frontend: [Constitutional-Law-Case-Prep](https://github.com/tbamisaye1/Constitutional-Law-Case-Prep).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# put OPENROUTER_API_KEY in .env
uvicorn app.main:app --reload --port 8000
```

Health check: http://127.0.0.1:8000/health

## Layout

```
app/
  main.py           FastAPI app + CORS
  config.py         env settings
  api/              route modules
  agents/           LangGraph state + prep graph
  rag/              chunk + FAISS store
  grounding/        claim verify + abstain
  llm/              OpenRouter client
  export/           notes → docx
```
