# FLOWCHART_DETAILED — cohere-rag-pdf

This document provides a professional, detailed explanation of each node in the flowchart and recommended production practices.

1) User Uploads PDFs (UI/Input)
   - Validate file types and sizes; support multiple uploads.
   - Expose chunking parameters in UI for advanced users.

2) Extract Text (PyMuPDF / OCR fallback)
   - Use PyMuPDF for selectable text; fallback to OCR for scanned documents.
   - Capture page-level metadata for provenance and traceability.

3) Chunk Text with Provenance
   - Sentence-aware chunking with tunable chunk_size and overlap settings.
   - Store file, start_page, end_page per chunk to enable auditing of model outputs.

4) Embed Chunks (Cohere)
   - Batch embedding requests, normalize vectors for inner-product search.
   - Implement retry/backoff and monitor embedding latency and failures.

5) Build FAISS Index
   - Use IndexFlatIP for normalized vectors; snapshot indices to persistent storage for faster restarts.

6) Query Flow (Embed Query -> Retrieve -> Compose Prompt)
   - Embed query with same model; retrieve top-k; compose system+user prompt with retrieved excerpts and instructions to rely on evidence.

7) OpenRouter Chat (LLM)
   - Use OpenRouter chat completions endpoint; ensure system instructions discourage hallucination.
   - Consider chunking context to respect model token limits; add truncation/summary strategies if needed.

8) Answer & Follow-ups
   - Display answer, show provenance, and list suggested follow-up questions to guide the user.

Production considerations include secrets management, authentication, index persistence, observability, cost control, and scalability.
