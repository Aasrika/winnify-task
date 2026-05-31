# WinTeach AI Content Generation Pipeline
### Winnify — Python Fundamentals Pilot

---

## Folder Structure

```
winnify-pipeline/
├── src/
│   ├── pipeline.py       ← entry point, run this
│   ├── ingestion.py      ← document loading and recursive chunking
│   ├── vector_store.py   ← FAISS index and retriever
│   ├── models.py         ← loads EMBED_MODEL and NLI_MODEL once
│   ├── schemas.py        ← Pydantic artifact schemas
│   ├── agents.py         ← LLM client and 4 artifact generators
│   └── validator.py      ← schema + NLI grounding critic
├── data/
│   └── sample_cs50.txt   ← sample source material (replace with real transcript)
├── outputs/              ← generated JSON artifacts saved here (git-ignored)
├── .env                  ← your Groq API key (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## One-Time Setup

### Step 1 — Clone or download the project
```bash
cd Desktop
git clone https://github.com/YOUR_USERNAME/winnify-pipeline.git
cd winnify-pipeline
```
Or just unzip the folder if you downloaded it.

### Step 2 — Create a virtual environment
```bash
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You should see `(venv)` in your terminal prompt.

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
This takes 3–5 minutes the first time (torch + transformers are large).

### Step 4 — Add your Groq API key
Open `.env` and replace the placeholder:
```
GROQ_API_KEY=your_actual_groq_key_here
```
Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.

### Step 5 — Open in VS Code
```bash
code .
```

---

## Running the Pipeline

```bash
cd src
python pipeline.py
```

First run downloads two models (~270 MB total) and caches them. Every run after that starts in seconds.

**To change the subtopic**, open `src/pipeline.py` and edit the bottom two lines:
```python
SOURCE = "../data/sample_cs50.txt"
SUBTOPIC = "Variables and Data Types"
```

Available subtopics in the sample file:
`Variables and Data Types`, `Functions`, `Loops`, `Conditionals`, `Exceptions`, `File Handling`

**To use your own CS50 transcript**, drop a `.txt` or `.pdf` file into `data/` and update `SOURCE`.

Output is saved to `outputs/<subtopic>.json` and previewed in the terminal.

---

## Architecture

```
data/sample_cs50.txt  (or any .txt / .pdf)
        ↓
ingestion.py — recursive_chunk()
  heading-aware, 800 words, 120-word overlap, metadata on every chunk
        ↓
models.py — SentenceTransformer (all-MiniLM-L6-v2)
        ↓
vector_store.py — FAISS IndexFlatIP
        ↓
Retriever — enriched query: "Python Fundamentals: {subtopic} educational concepts examples"
        ↓
agents.py — Artifact Router
  ├── Notes Agent     (detail + pedagogical flow)
  ├── Slides Agent    (4–6 bullets, 3–4 slides)
  ├── Quiz Agent      (explicit per-question Bloom level)
  └── Takeaways Agent (3–5 actionable outcomes)
        ↓
validator.py — Two-layer Critic
  ├── Layer 1: Pydantic schema validation
  │     structure, field types, bullet counts, Bloom diversity
  └── Layer 2: NLI entailment grounding (cross-encoder/nli-deberta-v3-base)
        claim extracted → scored against source chunks as premises
        passes if ≥ 60% of claims are entailed (score ≥ 0.3)
        ↓
  Pass → READY FOR REVIEW   (saved to outputs/)
  Fail → single retry with k=5 → Pass or FLAGGED
```

---

## Design Decisions

### LLM — Groq (Llama-3-8B)
Free tier, fast inference. Temperature 0.3 for deterministic structured JSON. One model for all artifact types — system prompts handle artifact-specific behavior.

### Chunking — Recursive + Heading-Aware
CS50 material has natural pedagogical structure. Recursive chunking respects section headings first, then splits by word count with 120-word overlap to prevent context fragmentation.

### Retrieval — Enriched Query
Subtopic string is expanded to `"Python Fundamentals: {subtopic} educational concepts examples"` before embedding. Produces richer signal than embedding a bare two-word string.

### Quality Check — Two Layers

**Schema (Pydantic):** catches structural failures — malformed JSON, wrong field types, wrong counts, missing Bloom diversity — before anything reaches the content team.

**NLI Entailment (DeBERTa):** unlike cosine similarity, entailment is directional. "Python is weakly typed" can score high similarity against "Python is dynamically typed" but scores low entailment. Each generated claim is checked against source chunks as premises. Artifact passes if ≥ 60% of claims are grounded.

**NLI label order:** verified automatically at startup by `models.py`. A known-entailing pair is scored and the index of the highest score is asserted to be 2. If it ever differs, the pipeline halts with a clear error before any artifact is validated.

### Bloom Enforcement — Two Layers
Prompt assigns a specific Bloom level per question position. Schema validator checks that all four required levels are present. If the model ignores the prompt, schema validation fails and triggers a retry.

---

## Uploading to GitHub

```bash
git init
git add src/ data/ requirements.txt README.md .gitignore
git commit -m "Initial commit: WinTeach AI content generation pipeline"
git remote add origin https://github.com/YOUR_USERNAME/winnify-pipeline.git
git branch -M main
git push -u origin main
```

`.env` is in `.gitignore` and will NOT be pushed. Your API key stays local.

---

## Known Limitations

1. NLI thresholds (0.3 entailment, 60% coverage) are heuristic — need calibration on real labeled data.
2. DeBERTa runs on CPU; slow for batch processing many subtopics. Use GPU or run async for production.
3. No cross-artifact consistency check — notes and quiz may cover different concepts within the same subtopic.
4. PDF extraction via PyPDF2 is text-only; scanned PDFs need OCR.
5. FAISS index is rebuilt on every run — production would serialize it to disk.
6. Groq free tier has rate limits — add retry with exponential backoff for batch runs.