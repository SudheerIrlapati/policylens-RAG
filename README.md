# PolicyLens — Insurance Policy RAG Assistant (Blueprint & Starter Code)

Ask plain-English questions about insurance policy PDFs and get answers with page citations.
This repo is a working starter. Follow the milestones below in order; each one ends with
something you can run. When you finish, replace this README with the portfolio README
template from your build-plan doc.

---

## 1. How it works

```
INGEST (once)     Policy PDFs ──> chunk + page tags ──> embed ──> ChromaDB
                                                                    │ similarity search
ASK (per query)   Question ──> retrieve top-k chunks ──> LLM prompt ──> cited answer
                                                                    │
EVALUATE          50 test questions ──> accuracy, page-hit rate, refusal rate
```

| File | What it does | Day |
|---|---|---|
| `src/config.py` | All settings (models, chunk size, top-k) | 0 |
| `src/ingest.py` | Load PDFs, chunk, embed, save to ChromaDB | 1–3 |
| `src/retriever.py` | Search ChromaDB, filter by policy | 3, 5 |
| `src/chain.py` | Prompt + LLM → answer with citations, refuses when unsure | 4 |
| `tests/test_pipeline.py` | Offline unit tests (no API cost) | 6 |
| `eval/questions.csv` + `eval/run_eval.py` | Your test set and scoring script | 8–9 |
| `src/api.py` | FastAPI: `/health`, `/policies`, `/ask` | 10 |
| `app/streamlit_app.py` | Chat UI with sources | 11 |
| `Dockerfile`, `docker-compose.yml` | Run API + UI in containers | 12 |

---

## 2. Setup (Day 0)

```bash
# 1. Create a GitHub repo named insurance-policy-rag-assistant, then copy these files into it
cd insurance-policy-rag-assistant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your key
cp .env.example .env               # then edit .env and paste your OpenAI key

# 3. Check everything installed (runs offline, should show 6 passed)
pytest -q

# 4. Add 5–10 public policy PDFs to data/  (see data/README.md)

git add . && git commit -m "Project skeleton" && git push
```

Expected cost for the whole project with gpt-4o-mini and text-embedding-3-small: a few dollars.

---

## 3. Milestones

### Milestone 1 — Ingest (Days 1–3)
```bash
python -m src.ingest --preview     # prints 3 chunks, no API cost
python -m src.ingest               # embeds everything into chroma_db/
```
**Check:** the preview shows the right policy name and page number, and clauses aren't cut mid-sentence.
**Experiment:** try `CHUNK_SIZE` 500, 800, 1200 in `.env`. Write down what you notice.

### Milestone 2 — First answers (Day 4)
```bash
python -m src.chain "Is flood damage covered?"
python -m src.chain "What is the capital of France?"   # should refuse
```
**Check:** answers cite pages like `[sample_home_policy p.12]`, and off-topic questions get
"I can't find this in the policy."

### Milestone 3 — Better retrieval (Day 5)
In `src/retriever.py`, try one upgrade and keep it only if answers improve:
- **MMR** (more varied chunks): `vs.max_marginal_relevance_search(question, k=k, fetch_k=20, filter=search_filter)`
- **Hybrid search**: combine BM25 (`langchain_community.retrievers.BM25Retriever`) with the vector results
- **Reranker**: retrieve 15 chunks, rerank with a cross-encoder, keep the top 4

### Milestone 4 — Tests (Day 6)
```bash
pytest -q
```
Add 2–3 tests of your own (for example: the policy filter only returns that policy).

### Milestone 5 — Evaluation (Days 8–9) ⭐ your QA advantage
1. Fill `eval/questions.csv` with **50 rows**: 35 `answerable`, 10 `tricky`
   (exclusions, limits, fine print), 5 `out_of_scope`. Look up each expected answer and
   page yourself in the PDF.
2. Run the baseline:
   ```bash
   python -m eval.run_eval --label baseline
   ```
3. Change **one** thing (chunk size, `TOP_K`, prompt, MMR), re-run `python -m src.ingest`
   if you changed chunking, then:
   ```bash
   python -m eval.run_eval --label chunk500
   ```
4. `eval/results_summary.csv` now holds your before/after table. Read the FAIL rows in
   `eval/results_<label>.csv` to see *why* it failed — that's your interview story.

Optional: add [RAGAS](https://docs.ragas.io) for faithfulness and context-precision scores.

### Milestone 6 — API (Day 10)
```bash
uvicorn src.api:app --reload
# open http://localhost:8000/docs and try POST /ask
```

### Milestone 7 — UI (Day 11)
```bash
# terminal 1
uvicorn src.api:app --reload
# terminal 2
streamlit run app/streamlit_app.py
```
Edit the `EXAMPLES` list in `app/streamlit_app.py` to match questions your policies can answer.

### Milestone 8 — Docker (Day 12)
```bash
python -m src.ingest               # build chroma_db/ first
docker compose up --build
# API: http://localhost:8000/docs   UI: http://localhost:8501
```

### Milestone 9 — Deploy (Day 13)
**Option A, easiest:** Streamlit Community Cloud. You'd change the UI to call `src.chain.answer`
directly instead of the API, commit `chroma_db/`, and add `OPENAI_API_KEY` as a secret.

**Option B, AWS (better for your resume):**
1. Launch an EC2 `t3.micro` (Ubuntu), open ports 8000 and 8501 in the security group.
2. SSH in, install Docker, `git clone` your repo, copy your `.env` and `data/` over.
3. `python -m src.ingest` (or copy `chroma_db/`), then `docker compose up -d --build`.
4. Visit `http://<public-ip>:8501`. Set an AWS billing alert and stop the instance when not demoing.

### Milestone 10 — Polish (Day 14)
- Replace this README with the portfolio template (problem, architecture image, eval table, demo GIF)
- Record a 1–2 minute demo GIF
- Pin the repo on GitHub; add the project to your resume and LinkedIn

---

## 4. Free alternative to OpenAI

Swap in a free model if you want zero cost:
- **LLM:** `pip install langchain-groq`, then in `chain.py` use
  `from langchain_groq import ChatGroq; llm = ChatGroq(model="llama-3.1-8b-instant")`
- **Embeddings:** `pip install langchain-huggingface sentence-transformers`, then use
  `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")` in
  `ingest.py` and `retriever.py`

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `No PDFs found in 'data'` | Put `.pdf` files in `data/` and run from the project root |
| `AuthenticationError` / 401 | Check `OPENAI_API_KEY` in `.env` |
| Answers ignore the policy filter | Re-run `python -m src.ingest` so chunks carry the `policy` tag |
| Empty or garbled text from a PDF | It's probably a scanned image; pick a text-based PDF or add OCR later |
| UI says it can't reach the API | Start `uvicorn` first; in Docker, the UI uses `API_URL=http://api:8000` |
