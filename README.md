# Resume Ranker

A utility that ranks job candidates against a job description based on three key criteria:

1. **Technology Skills** — Matches candidate skills against required technologies
2. **Experience** — Compares years of experience to job requirements
3. **Roles & Responsibilities** — Measures alignment using TF-IDF cosine similarity

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

| Criterion | Weight | Method |
|-----------|--------|--------|
| Skills Match | 45% | Keyword extraction from a curated tech skills dictionary |
| Experience | 25% | Regex-based extraction of years, scored against JD requirements |
| Roles & Responsibilities | 30% | TF-IDF vectorization + cosine similarity |

### Justification

Each candidate receives a breakdown showing:
- Matched, missing, and additional skills
- Experience fit relative to requirements
- Role alignment score with interpretation

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
├── skills.py       # Technology skills dictionary & extraction
├── experience.py   # Experience years extraction
├── ranker.py       # Core ranking engine
├── templates/      # Jinja2 HTML templates
│   ├── index.html
│   └── results.html
└── static/
    └── style.css
```
