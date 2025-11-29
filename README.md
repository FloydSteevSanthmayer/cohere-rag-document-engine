# cohere-rag-pdf

Retrieval-Augmented Generation for PDFs — Cohere embeddings + FAISS + OpenRouter LLM in a Streamlit UI.

## Author
**Floyd Steev Santhmayer**

## Overview
This repository provides a production-oriented Retrieval-Augmented Generation (RAG) system for extracting knowledge from PDF documents and answering user questions with LLM assistance and evidence provenance. The pipeline uses Cohere for embeddings, FAISS for vector search, and OpenRouter-compatible chat models for answer generation. It includes OCR fallback, provenance-aware chunking, CI, pre-commit, and Docker support.

---

## Quickstart
1. Copy `.env.example` to `.env` and add your API keys (`OPENROUTER_API_KEY`, `COHERE_API_KEY`).
2. Create and activate a virtual environment.
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run locally:
```bash
streamlit run app.py
```

## Docker (quickstart)
Build and run locally with Docker:
```bash
docker build -t cohere-rag-pdf:latest .
docker run -p 8501:8501 --env-file .env cohere-rag-pdf:latest
```

## What’s included
- `app.py` — Streamlit entrypoint (launcher). Replace with full pipeline logic if needed.
- `flowchart_colored.mmd`, `architecture.mmd` — editable Mermaid sources.
- `docs/flowchart_colored.png`, `docs/architecture_colored.png` — rendered diagrams.
- `FLOWCHART_DETAILED.md` — professional, step-by-step explanation of the flowchart.
- `Dockerfile` — for containerized deployment.
- CI & automation: `.github/workflows/ci.yml`, `.github/dependabot.yml`, `.pre-commit-config.yaml`.
- Tests: `tests/` — pytest scaffold.
- `requirements.txt`, `.env.example`, `.gitignore`, `CONTRIBUTING.md`, `LICENSE` (MIT).

---

## Security notes
- Do **not** commit `.env` or any secrets. Use GitHub Secrets for CI and production.
- Rotate keys regularly and enforce usage limits and alerting for embeddings & model usage.

---

## Production considerations
- Add authentication to the app (OAuth / SSO).
- Persist FAISS snapshots to cloud storage (S3) to avoid full rebuild on restart.
- For large-scale use, migrate to a managed vector DB (Pinecone, Weaviate, Milvus) and use batch/async embedding flows.
- Add observability: metrics for embedding latency, cost, and index sizes.

---

## License
MIT © 2025 Floyd Steev Santhmayer
