# WinTeach AI Content Generation Pipeline
### Winnify — Python Fundamentals Pilot

An AI pipeline that takes CS50 Python lecture notes as input and generates four types of LMS-ready learning artifacts per topic — Class Notes, Slides, Student Quiz (Bloom's taxonomy), and Key Takeaways — with a two-layer quality check before each artifact is marked Ready for Review.

---

## Folder Structure

```
winnify-pipeline/
├── src/
│   ├── pipeline.py       ← entry point, run this
│   ├── ingestion.py      ← URL fetching, noise filtering, recursive chunking
│   ├── vector_store.py   ← FAISS index and enriched retrieval
│   ├── models.py         ← loads EMBED_MODEL and NLI_MODEL once at startup
│   ├── schemas.py        ← Pydantic artifact schemas with custom validators
│   ├── agents.py         ← Groq LLM client, safe_json parser, 4 generators
│   └── validator.py      ← Pydantic schema + NLI entailment grounding critic
├── sample_output/
│   └── unit_tests.json   ← sample generated output 
├── outputs/              ← all generated JSON artifacts saved here (git-ignored)
├── .env                  ← your Groq API key (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## One-Time Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/Aasrika/winnify-task.git
cd winnify-assignment
```

### Step 2 — Create and activate a virtual environment
```bash
python -m venv venv
```
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You should see `(venv)` in your terminal prompt.

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
First time takes 3–5 minutes. torch and transformers are the large packages.

### Step 4 — Get a free Groq API key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up — no credit card needed
3. Click **API Keys → Create API Key**
4. Copy the key

### Step 5 — Add your key to .env
Open the `.env` file in the project root and replace the placeholder:
```
GROQ_API_KEY=your_actual_groq_key_here
```

---

## Running the Pipeline

```bash
cd src
python pipeline.py
```

On first run, two models download and cache (~270 MB total):
- `all-MiniLM-L6-v2` — embedding model for FAISS retrieval (~90 MB)
- `cross-encoder/nli-deberta-v3-base` — NLI model for grounding validation (~180 MB)

Every run after that starts in seconds from cache.

### What you see

```
NLI label verification:
  Raw scores: [-5.26, 3.93, 0.13]
  Detected entailment index: 1
  Verified: index 1 = entailment. Safe to proceed.

============================================================
WinTeach — Python Fundamentals Content Generator
============================================================

Select a CS50 Python week to generate artifacts for:

  1. Week 0 — Functions and Variables
  2. Week 1 — Conditionals
  3. Week 2 — Loops
  4. Week 3 — Exceptions
  5. Week 4 — Libraries
  6. Week 5 — Unit Tests
  7. Week 6 — File I/O
  8. Week 7 — Regular Expressions
  9. Week 8 — Object-Oriented Programming
  10. Week 9 — Et Cetera

Enter number (1-10):
```

Type a number, press Enter. The pipeline fetches that week's notes from CS50 directly, generates all four artifacts, validates them, shows a preview, and saves the output to `outputs/<topic_name>.json`.

After each run it asks `Generate another topic? (y/n)` — type `y` to continue without restarting.

### Known working topics
Weeks 1–7 (Conditionals through File I/O) and Week 9 (Object-Oriented Programming) all generate successfully with all four artifacts READY FOR REVIEW.

### Known limitations by topic
**Week 8 (Regular Expressions) and Week 10 (Et Cetera)** fail artifact generation due to two compounding issues: regex backslash sequences (`\w`, `\d`, `\s`) are illegal in JSON strings, and both topics are token-dense causing truncation before the JSON closes. The production fix is Groq structured outputs (function calling) which enforces the JSON schema at the token generation level — see Known Limitations below.

---

## Architecture

```
CS50 Notes Page (URL)
        ↓
ingestion.py — fetch_url() + recursive_chunk()
  BeautifulSoup scraping, noise filtering, heading-aware chunking
  800 words per chunk, 120-word overlap, UUID + metadata on every chunk
        ↓
models.py — SentenceTransformer all-MiniLM-L6-v2
  NLI label order verified at startup via probe pair + np.argmax
        ↓
vector_store.py — FAISS IndexFlatIP (cosine similarity)
  Enriched query: "Python Fundamentals: {subtopic} educational concepts examples"
  Retrieves top-k most relevant chunks
        ↓
agents.py — Artifact Router (pipeline.py loops over ARTIFACT_GENERATORS)
  ├── Notes Agent     — detail, code examples, max 300 words, 700 tokens
  ├── Slides Agent    — 3 slides × 3-5 bullets, 800 tokens
  ├── Quiz Agent      — 5 MCQs, explicit Bloom level per question, 700 tokens
  └── Takeaways Agent — 4 sentences under 20 words each, 400 tokens
        ↓
validator.py — Two-layer critic
  ├── Layer 1: Pydantic schema validation
  │     field types, bullet counts, question count, Bloom diversity (all 4 levels present)
  └── Layer 2: NLI entailment grounding (cross-encoder/nli-deberta-v3-base)
        extract claims → score each against source chunks as premises
        claim grounded if best entailment score ≥ 0.3
        artifact passes if ≥ 60% of claims are grounded
        ↓
  Pass → READY FOR REVIEW  (saved to outputs/topic.json)
  Fail → single retry with k=5 chunks → Pass or FLAGGED
```

---

## Design Decisions

### Source material — CS50 HTML notes pages
CS50 offers notes (HTML), slides (Google Slides PDF), and transcripts. The HTML notes pages were chosen because they are plain accessible HTML with no authentication, clean heading structure, and preserved code examples. Google Slides PDFs extract poorly — garbled text, no structure.

### Chunking — Recursive heading-aware
CS50 notes have natural pedagogical boundaries at each heading. Fixed-size chunking ignores these and can split a code example mid-line. Recursive heading-aware chunking respects section boundaries first, then splits by word count with 120-word overlap to prevent context loss at boundaries.

### LLM — Groq Llama-3.1-8b-instant
Free tier, fastest inference available (500+ tokens/sec). Temperature 0.3 for deterministic structured output. Same model for all four agents — system prompts handle the artifact-specific differences. Originally `llama3-8b-8192` but that model was decommissioned by Groq during development and updated to the current model.

### Retrieval — Enriched query
A bare subtopic string like `"Loops"` produces a weak two-word embedding. Expanding to `"Python Fundamentals: Loops educational concepts examples"` produces a richer vector that aligns better with how chunk text is phrased, improving retrieval quality.

### Quality check — Two layers

**Layer 1 — Pydantic schema validation:** Production pipelines fail structurally all the time — malformed JSON, wrong field types, missing fields, wrong counts. Pydantic catches these at generation time with clear error messages before anything reaches the content team.

**Layer 2 — NLI entailment grounding:** Cosine similarity is symmetric and direction-agnostic. It cannot distinguish "Python is dynamically typed" from "Python is weakly typed" — both have high similarity scores but mean different things. NLI entailment is directional: it treats source chunks as premises and generated claims as hypotheses, checking whether the source actually supports the claim. This catches factual drift that similarity cannot. The DeBERTa model's label order is verified at every startup via a known-entailing probe pair — the entailment index is detected automatically, not hardcoded.

### Bloom taxonomy enforcement — Two layers
The quiz system prompt assigns a specific Bloom level to each question by position. The Pydantic schema validator then checks that all four required levels (Remember, Understand, Apply, Analyze) are actually present in the output. If the model ignores the prompt, schema validation fails and triggers a retry.

### Retry strategy — Single retry
One retry with k=5 chunks instead of k=4. No recursive loops. A human reviews every artifact before it reaches students — one retry with richer context fixes most failures, and a FLAGGED artifact goes to the content team with its full validation report attached.

---

## Output Schema

Every artifact in the output JSON contains:

```json
{
  "artifact": {
    "artifact_type": "quiz",
    "topic": "Python Fundamentals",
    "subtopic": "Unit Tests",
    "questions": [
      {
        "question": "What is the purpose of unit tests?",
        "options": ["To debug code", "To test specific aspects", "To run code", "To deploy code"],
        "answer": "To test specific aspects of code",
        "bloom_level": "Remember"
      }
    ]
  },
  "source_chunk_ids": ["uuid1", "uuid2", "uuid3", "uuid4"],
  "validation": {
    "schema_valid": true,
    "schema_message": "OK",
    "nli_coverage_ratio": 1.0,
    "nli_grounding_ok": true,
    "claim_details": [
      {
        "claim": "What is the purpose of unit tests?",
        "best_entailment_score": 2.24,
        "grounded": true
      }
    ],
    "passed": true
  },
  "status": "READY FOR REVIEW"
}
```

`source_chunk_ids` provides full provenance — every artifact is traceable back to the exact source chunks that produced it.

---

## Known Limitations

| Limitation | Root cause | Fix with more time |
|---|---|---|
| Week 8 (Regex) and Week 10 (Et Cetera) fail | Regex backslash sequences illegal in JSON strings + token-dense topics cause truncation | Groq structured outputs (function calling) — enforces schema at token generation level |
| NLI thresholds are heuristic | 0.3 entailment and 60% coverage chosen empirically | Calibrate on human-labeled hallucination examples using precision-recall curves |
| No cross-artifact consistency | Notes and Quiz generated independently may cover different concepts | Post-generation coverage alignment check across all four artifacts |
| DeBERTa slow on CPU | Cross-encoder inference is compute-heavy | Run on GPU or async background worker for batch processing |
| FAISS index rebuilt every run | No persistence layer | Serialize index and chunk metadata to disk after first build |
| Groq rate limits on free tier | Per-minute token cap | Exponential backoff with retry on 429 errors in call_llm() |

---

## Dependencies

```
groq                  — Groq LLM API client
faiss-cpu             — vector similarity search
sentence-transformers — all-MiniLM-L6-v2 embeddings + DeBERTa NLI
pydantic              — artifact schema validation
PyPDF2                — PDF text extraction (fallback)
transformers          — required by sentence-transformers
torch                 — required by sentence-transformers
python-dotenv         — reads GROQ_API_KEY from .env
requests              — CS50 URL fetching
beautifulsoup4        — HTML parsing and noise filtering
```

Install all with:
```bash
pip install -r requirements.txt
```
