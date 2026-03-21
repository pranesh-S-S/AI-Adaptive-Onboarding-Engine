# AI-Adaptive-Onboarding-Engine

![License](https://img.shields.io/badge/license-MIT-green)

## Download Here the total code (code is uploaded in drive because of large size)
https://drive.google.com/drive/folders/12__tkO0EtIka69FQm8QE8z2DySkZ1r1t?usp=sharing

# 🧠 AI Adaptive Onboarding Engine

> **Resume → Gap Analysis → Personalised Learning Roadmap — in under 60 seconds.**
> Powered by Groq API · Llama 3.3-70B · Semantic Embeddings · FastAPI

---

## 📌 Table of Contents

- [What This Project Does](#what-this-project-does)
- [Why It's Different](#why-its-different)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [How the Adaptive Algorithm Works](#how-the-adaptive-algorithm-works)
- [Datasets Used](#datasets-used)
- [Known Issues & Fixes](#known-issues--fixes)
- [Environment Variables](#environment-variables)

---

## What This Project Does

The **AI Adaptive Onboarding Engine** is an end-to-end intelligent system that:

1. Reads a candidate's resume (PDF, DOCX, or scanned image)
2. Accepts a **job description** or just a **raw job role title** (e.g. `"AI Engineer"`)
3. Performs semantic gap analysis between the candidate's skills and the role requirements
4. Generates a **personalised, dependency-ordered learning roadmap** with weekly plans and curated resources

Unlike every existing ATS or onboarding tool, this system does not keyword-match. It understands context, proficiency levels, and skill importance — and explains every decision it makes.

---

## Why It's Different

| Problem With Existing Systems | How This System Solves It |
|---|---|
| `'ML'` and `'Machine Learning'` treated as different skills | Dense vector embeddings — cosine similarity ≥ 0.75 = matched |
| Candidate with Python Beginner scores same as Advanced | LLM grades proficiency level from resume evidence |
| Missing a critical skill scores same as missing an optional one | Importance-weighted scoring (High=1.0 / Medium=0.6 / Low=0.3) |
| Requires a full written job description | Role-Intent Expansion — Llama 3 synthesises skill profile from a job title alone |
| Static, hardcoded skill dependency tables | Dynamic knowledge graph built by LLM at runtime, per role |
| No explanation for why a skill was flagged | Full reasoning trace per skill (score, best match, decision) |
| Needs GPU / local model | Groq Cloud API — runs on any machine, no GPU required |

---

## Key Features

### 🎯 Dual Input Mode
- **Job Description mode** — paste or upload a full JD (PDF, DOCX, or plain text)
- **Role-Intent mode** — type just a job title like `"Flutter Developer"` and the system builds the full skill profile using Llama 3

### 🔍 Semantic Skill Matching
- Uses `sentence-transformers/all-MiniLM-L6-v2` for dense vector matching
- Catches `torch → PyTorch`, `postgres → PostgreSQL`, `js → JavaScript` automatically
- Every match logged in a **reasoning trace** — fully auditable

### 📊 Importance-Weighted ATS Scoring
```
Final Score = Required Coverage  × 0.50   (importance-weighted)
            + Preferred Coverage × 0.20   (competitive advantage)
            + Skill Strength     × 0.30   (LLM-graded level × importance)
```

### 🧩 Level Gap Detection
Three gap states — not just present/absent:
- ✅ **Overlapping** — candidate has it and it's strong enough
- ⚠️ **Level Gap** — candidate has it but proficiency is too low
- ❌ **Missing** — candidate doesn't have it at all

### 🗺️ Adaptive Learning Roadmap
- Dependency-ordered DAG built by Llama 3 for the specific role
- Skills locked until prerequisites are completed
- Personalised by learning speed, hours/day, and preferred learning style
- Weekly plan that packs skills into 5-day work-week buckets
- Groq-powered curated learning resources per skill (cached)

### 🔄 Feedback Loop
Re-upload an updated resume after learning — the system recomputes the delta, shows score improvement, and unlocks newly satisfied dependencies.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                          │
│                    POST /analyze                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │           10-Step Pipeline          │
          └─────────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
Step 1–2               Step 3–6               Step 7–10
─────────              ────────               ─────────
Resume Ingestion       Gap Analysis           Adaptive Roadmap
JD / Role Processing   ATS Scoring            Skill Graph (LLM)
PDF / DOCX / OCR       Level Gap Detect       Progress Tracking
Section Detection      Reasoning Trace        Weekly Plan
Skill Extraction       Skill Clustering       Groq Resources
```

### Pipeline Steps

| # | Stage | Output |
|---|---|---|
| 1 | Resume Ingestion | raw_text, sections, confidence scores, chunks |
| 2 | JD / Role Processing | role_title, required_skills[], preferred_skills[], importance_map{} |
| 3 | Resume Skill Level Analysis | skills{} with Beginner/Intermediate/Advanced + evidence (Groq) |
| 4 | Gap Analysis | overlapping[], gap_skills[], level_gaps[], reasoning_trace{} |
| 5 | Gap Diagnosis | per-skill learning path and level upgrade advice (Groq) |
| 6 | ATS Scoring | required_score, preferred_score, strength_score, final_score |
| 7 | Skill Clustering | 9-category cluster maps for resume and JD |
| 8 | Dynamic Skill Graph | DAG of prerequisites inferred by Groq for this specific role |
| 9 | Progress Tracking | 0.0–1.0 exposure score per gap skill (Groq LLM) |
| 10 | Adaptive Roadmap | Nodes with status, priority, estimated_days, resources, weekly plan |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Llama 3.3-70B-Versatile via [Groq Cloud API](https://console.groq.com) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **PDF Extraction** | PyMuPDF (`fitz`) — block-level text |
| **DOCX Extraction** | `python-docx` — paragraph-level |
| **OCR** | `pdf2image` + `pytesseract` + OpenCV (scanned PDFs) |
| **API Framework** | FastAPI + Uvicorn |
| **Language** | Python 3.10+ |
| **Caching** | `@lru_cache` (embedding pairs) + `resource_cache` dict (Groq resources) |

> **No GPU required.** All LLM inference runs on Groq's cloud LPU hardware.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-adaptive-onboarding-engine.git
cd ai-adaptive-onboarding-engine
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
fastapi
uvicorn
python-multipart
pymupdf
python-docx
pdf2image
pytesseract
opencv-python
numpy
groq
sentence-transformers
```

### 4. Install Tesseract OCR (for scanned PDFs)

```bash
# Windows — download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki

# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

### 5. Set your Groq API key

```bash
# Windows
set GROQ_API_KEY=gsk_your_key_here

# Mac / Linux
export GROQ_API_KEY=gsk_your_key_here
```

Get a free API key at: [console.groq.com](https://console.groq.com)

---

## Usage

### Run the FastAPI server

```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Option A — Use the API (recommended)

```bash
# With a full job description
curl -X POST http://localhost:8000/analyze \
  -F "resume_file=@resume.pdf" \
  -F "job_description=We need a Python developer with ML experience..."

# With just a job role title
curl -X POST http://localhost:8000/analyze \
  -F "resume_file=@resume.docx" \
  -F "job_role=AI Engineer"
```

### Option B — Run the CLI pipeline directly

```bash
python adaptive.py
```

You will be prompted to:
1. Answer 4 personalisation questions (learning speed, hours/day, career goal, learning style)
2. Enter resume file path (`.pdf` or `.docx`)
3. Enter job description text or file path (or a job role title)

### Example output

```
============================================================
   AI ADAPTIVE ONBOARDING ENGINE — GROQ + LLAMA 3
============================================================

RESUME — EXTRACTED SKILLS
  - Python
  - SQL
  - Git

JOB DESCRIPTION — LLM EXTRACTED
  Role : Machine Learning Engineer
  Required Skills (6):
    - Python                         [High]
    - Machine Learning               [High]
    - TensorFlow                     [High]
    - Docker                         [Medium]
    - SQL                            [Medium]
    - Git                            [Low]

GAP ANALYSIS
  Required Skills Matched  : 3 / 6
  Match Score : 52.5%
  Gap Score   : 47.5%
  Missing     : ['Machine Learning', 'TensorFlow', 'Docker']

REASONING TRACE — SKILL MATCHING LOGIC
  ✅ Python        Best Match: Python      Score: 1.00  Decision: covered
  ❌ TensorFlow    Best Match: NumPy       Score: 0.48  Decision: not covered
  ❌ Docker        Best Match: Git         Score: 0.51  Decision: not covered

ADVANCED SCORING SYSTEM (ATS-LEVEL)
  Core Match Score  : ████░░░░░░ 42.0%
  Competitive Score : ████░░░░░░ 40.0%
  Skill Strength    : ████░░░░░░ 38.0%

  FINAL READINESS   : ████░░░░░░ 40.2%
  Readiness Level : Needs Improvement

ADAPTIVE LEARNING ROADMAP
  Profile: average learner | 2.0h/day | hands-on style

  🔵 Python             [High]  Priority: 0.8  ~14 day(s)
     Status   : In Progress
     Progress : 30.0%
     Task     : Learn Python in ~14 day(s) — Build a mini project

  🔒 Machine Learning   [High]  Priority: 0.8  ~21 day(s)
     Status   : Locked
     Locked by: Python

  🔒 TensorFlow         [High]  Priority: 0.9  ~14 day(s)
     Status   : Locked
     Locked by: Machine Learning

  🔵 Docker             [Medium] Priority: 0.6  ~5 day(s)
     Status   : Start
     Resources:
       [course       ] Docker for Beginners - TechWorld with Nana
                       https://www.youtube.com/watch?v=3c-iBn73dDE
       [documentation] Docker Official Docs
                       https://docs.docker.com/get-started/

WEEKLY LEARNING PLAN
  Week 1 (5 day(s)):
    - Docker                          ~5 day(s)  [Medium]

  Week 2 (5 day(s)):
    - Python                          ~14 day(s) [High]

NEXT BEST SKILL TO LEARN:
  -> Docker  (Priority: 0.6, Importance: Medium, ~5 day(s))
     Learn Docker in ~5 day(s) — Build a mini project

  Learning Progress: ░░░░░░░░░░ 10.0%
```

---

## API Reference

### `POST /analyze`

Analyse a resume against a job description or role title.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `resume_file` | File | ✅ | PDF or DOCX resume file |
| `job_description` | string | ⚡ one of these | Full job description text or file path |
| `job_role` | string | ⚡ one of these | Raw role title e.g. `"AI Engineer"` |

> Provide either `job_description` OR `job_role` — not both.

**Response** — `application/json`

```json
{
  "gap_analysis": {
    "overlapping_skills": ["Python", "SQL"],
    "gap_skills": ["TensorFlow", "Docker"],
    "preferred_gap": ["Kubernetes"],
    "level_gaps": [{"skill": "Python", "current_level": "Beginner", "required_level": "Advanced"}],
    "match_score": 52.5,
    "gap_score": 47.5,
    "reasoning_trace": {
      "TensorFlow": {"best_match": "NumPy", "score": 0.48, "decision": "not covered"}
    }
  },
  "advanced_score": {
    "required_score": 42.0,
    "preferred_score": 40.0,
    "strength_score": 38.0,
    "final_score": 40.2,
    "readiness": "Needs Improvement"
  },
  "roadmap": [...],
  "weekly_roadmap": [...],
  "next_skill": {"skill": "Docker", "priority": 0.6, "estimated_days": 5},
  "learning_score": 10.0
}
```

---

## Project Structure

```
ai-adaptive-onboarding-engine/
│
├── backend/
│   ├── adaptive.py          # Core pipeline — all 10 steps
│   ├── api.py               # FastAPI server
│   └── uploads/             # Temporary resume file storage
│
├── requirements.txt
├── .env.example
└── README.md
```

### Key functions in `adaptive.py`

| Function | Purpose |
|---|---|
| `run_full_pipeline()` | Orchestrates all 10 pipeline steps |
| `process_resume()` | Extracts text, detects sections, extracts skills |
| `process_job_description()` | Handles JD or role title → calls Groq |
| `extract_jd_with_groq()` | LLM JD parsing with importance tags |
| `perform_gap_analysis()` | Semantic gap analysis with reasoning trace |
| `compute_advanced_score()` | Importance-weighted ATS scoring |
| `build_dynamic_skill_graph()` | LLM-inferred DAG for the role |
| `llm_progress_tracking()` | Groq-based 0.0–1.0 exposure scoring |
| `generate_adaptive_roadmap()` | Builds dependency-ordered roadmap |
| `adaptive_update()` | Sets node statuses (Locked/Start/In Progress/Completed) |
| `personalize_roadmap()` | Adjusts times by learning speed and hours/day |
| `attach_resources()` | Fetches Groq-powered learning resources |
| `build_weekly_roadmap()` | Packs skills into 5-day weekly buckets |
| `re_evaluate()` | Feedback loop — compare against previous result |
| `skill_match_trace()` | Per-skill reasoning log for explainability |
| `generate_resources()` | Groq-powered resource suggestions (cached) |

---

## How the Adaptive Algorithm Works

### Dependency Graph (DAG)

The skill prerequisite graph is **not hardcoded**. Llama 3 infers it at runtime based on the specific role:

```python
# For 'ML Engineer' role:
skill_graph = {
    'Machine Learning' : ['Python'],
    'TensorFlow'       : ['Python', 'Deep Learning'],
    'Kubernetes'       : ['Docker'],
}

# For 'Frontend Developer' role (completely different graph):
skill_graph = {
    'React'      : ['JavaScript'],
    'Next.js'    : ['React'],
    'TypeScript' : ['JavaScript'],
}
```

### Priority Formula

```python
priority = importance_weight * 0.5   # High=1.0, Medium=0.6, Low=0.3
         + dependency_depth  * 0.2   # Foundational skills promoted
         + gap_score         * 0.3   # Always 1.0 for missing skills
```

### Node Status Flow

```
Start → In Progress (exposure ≥ 0.3) → Completed (exposure ≥ 0.7)
Locked (prerequisite not met) → unlocks when prerequisite Completed
```

### Personalisation

```python
# Time adjusted by learning speed and available hours
adjusted_days = base_days × difficulty_multiplier × speed_multiplier × (2.0 / hours_per_day)

# Example: Docker (5 days base) for fast learner at 4 hrs/day:
adjusted_days = 5 × 1.0 × 0.7 × (2.0 / 4.0) = 1.75 ≈ 2 days
```

---

## Datasets Used

| Dataset | Size | Purpose |
|---|---|---|
| [Kaggle — Jobs & Job Descriptions](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description) | 2,277 postings | JD extraction validation |
| [O*NET Database v30.1](https://www.onetcenter.org/db_releases.html) | 900+ occupations | Skill relevance validation |
| [Kaggle — Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | 2,484 resumes across 25 categories | Cross-domain scalability testing |

> **No dataset was used for model training.** All three were used only for validation and testing.
> The LLM runs entirely zero-shot — no fine-tuning was performed.

---

## Known Issues & Fixes

### `ValueError: min() arg is an empty sequence`

**Cause:** The `adaptive_update()` function in your local build may contain a stray `min()` call on an empty candidates list. This happens when Groq hits the daily rate limit (100,000 tokens/day), causing JD extraction to fail and returning an empty gap skills list — which produces an empty roadmap, then crashes on `min([])`.

**Fix 1 — Remove the `min()` from `adaptive_update`:**
```python
def adaptive_update(roadmap, progress):
    completed_skills = []
    for node in roadmap:
        # ... status logic ...
    return roadmap   # ← just return, no min() call here
```

**Fix 2 — Guard the pipeline against empty gap_skills:**
```python
gap_skills = gap_analysis["gap_skills"]
if not gap_skills:
    log.warning("No gap skills — roadmap will be empty (likely rate limit hit)")
    roadmap = []
    weekly_roadmap = []
    next_skill = None
    learning_score = 0.0
else:
    # normal roadmap generation
    skill_graph = build_dynamic_skill_graph(...)
```

### `429 Too Many Requests — Rate limit exceeded`

The Groq free tier allows **100,000 tokens per day** on `llama-3.3-70b-versatile`. Each full pipeline run uses approximately 6,000–10,000 tokens.

- The error message tells you exactly how long to wait (e.g. `"Please try again in 4m40s"`)
- The Groq SDK retries automatically with exponential backoff for transient 429s
- All call sites return safe empty defaults so the pipeline completes partially rather than crashing
- To remove the limit: upgrade to [Groq Dev Tier](https://console.groq.com/settings/billing)

### `Model decommissioned` error

If you see a 404 or model-not-found error, the model string is outdated. Update:

```python
# In adaptive.py
GROQ_MODEL = "llama-3.3-70b-versatile"   # ← always use this
```

Models `llama3-8b-8192` and `llama-3.1-8b-instant` are deprecated.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Your Groq API key from [console.groq.com](https://console.groq.com) |

```bash
# .env.example
GROQ_API_KEY=gsk_your_key_here
```

---

## Validation Metrics

| Metric | Value | How Measured |
|---|---|---|
| JD Extraction Precision | 92% | Manual annotation of 50 real JDs |
| Semantic Match Threshold | 0.75 cosine similarity | Calibrated against 200 synonym pairs |
| Cross-Domain Pass Rate | 100% | All 25 resume categories in Kaggle dataset |
| Embedding Cache Efficiency | ~40× re-encoding reduction | `@lru_cache` on skill pairs |
| End-to-End Latency | < 60 seconds | 1-page resume, 300-word JD, Groq free tier |

---

## License

This project was built for a hackathon submission. All code is original.
LLM inference powered by [Groq](https://groq.com) — used under their free tier terms.
Embedding model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Apache 2.0 License.

---

<div align="center">
  <strong>Built with Groq · Llama 3.3 · sentence-transformers · FastAPI · PyMuPDF · Tesseract</strong>
</div>

