# Resume Ranker

A LinkedIn-style resume ranking utility that scores candidates against a job description using multiple AI-powered signals:

1. **Keyword & Skill Matching** — Exact skill matching with AI-inferred skill graph
2. **Semantic AI Matching** — Meaning-based NLP matching using sentence-transformers
3. **Title Relevance** — Job title alignment using semantic similarity
4. **Experience** — Years of experience vs JD requirements
5. **Location Match** — Location alignment and relocation readiness
6. **Roles & Responsibilities** — TF-IDF cosine similarity for role fit
7. **Profile Completeness** — Resume detail level and section coverage

Each candidate receives a detailed justification explaining their ranking.

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Run the Web UI

```bash
uvicorn resume_ranker.app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Usage

1. Paste the job description in the text area
2. Upload one or more resumes (PDF, DOCX, or TXT)
3. Click **Rank Candidates** to see results

### API Endpoint

```bash
curl -X POST http://localhost:8000/api/rank \
  -F "job_description=Looking for a Python developer with 5+ years experience..." \
  -F "resumes=@candidate1.pdf" \
  -F "resumes=@candidate2.pdf"
```

Returns JSON with ranked candidates, scores, and justifications.

## Scoring Methodology

| Signal | Weight | Method | Importance |
|--------|--------|--------|------------|
| Skills Match | 25% | Keyword matching + skill inference graph | High |
| Semantic AI Match | 20% | Sentence-transformer embedding similarity | High |
| Title Relevance | 10% | Semantic comparison of job titles | High |
| Experience | 15% | Regex-based years extraction vs requirements | High |
| Location Match | 10% | Location/relocation readiness detection | High |
| Roles & Responsibilities | 10% | TF-IDF vectorization + cosine similarity | Medium |
| Profile Completeness | 10% | Resume section coverage + detail level | Medium |

### Key Features

- **JD Section Detection**: Distinguishes Required vs Preferred skills — only penalizes for missing required skills
- **Skill Inference Graph**: Infers related skills (e.g., Spring Boot → Distributed Systems, Microservices)
- **Semantic Matching**: Understands that "Backend Architect" matches "Senior Java Engineer"
- **Location Intelligence**: Detects relocation readiness and city aliases (NYC = New York)
- **Profile Scoring**: Evaluates resume completeness (summary, experience, skills, education sections)

### Justification

Each candidate receives a breakdown showing:
- Matched, missing, additional, and AI-inferred skills
- Semantic alignment score with the JD
- Title relevance and seniority alignment
- Experience fit relative to requirements
- Location and relocation match
- Role alignment score with interpretation
- Profile completeness assessment

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Format
ruff format .
```

## Project Structure

```
resume_ranker/
├── app.py          # FastAPI web application
├── parser.py       # PDF/DOCX/TXT text extraction
├── skills.py       # Technology skills dictionary, extraction & inference graph
├── experience.py   # Experience years extraction
├── ranker.py       # Core ranking engine with 7 scoring signals
├── semantic.py     # Semantic AI matching (sentence-transformers)
├── signals.py      # Title, location, and completeness signals
├── templates/      # Jinja2 HTML templates
│   ├── index.html
│   └── results.html
└── static/
    └── style.css
```
