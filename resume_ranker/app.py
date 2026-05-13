"""FastAPI application for the Resume Ranker utility."""

from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from resume_ranker.parser import extract_text
from resume_ranker.ranker import CandidateResult, rank_candidates

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Resume Ranker", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Render the upload form."""
    return templates.TemplateResponse(request, "index.html")


def _result_to_dict(r: CandidateResult) -> dict:
    return {
        "candidate_name": r.candidate_name,
        "rank": r.rank,
        "overall_score": round(r.overall_score, 1),
        "skills": {
            "match_percentage": round(r.skill_analysis.match_percentage, 1),
            "matched": r.skill_analysis.matched_skills,
            "missing": r.skill_analysis.missing_skills,
            "extra": r.skill_analysis.extra_skills,
        },
        "experience": {
            "score": round(r.experience_analysis.score, 1),
            "candidate_years": r.experience_analysis.candidate_years,
            "required_min": r.experience_analysis.required_min,
            "required_max": r.experience_analysis.required_max,
            "summary": r.experience_analysis.summary,
        },
        "roles": {
            "score": round(r.roles_analysis.similarity_score, 1),
            "summary": r.roles_analysis.summary,
        },
        "justification": r.justification,
    }


@app.post("/rank", response_class=HTMLResponse)
async def rank_resumes(
    request: Request,
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...),
) -> HTMLResponse:
    """Process uploaded resumes and rank them against the job description."""
    candidates: list[tuple[str, str]] = []
    errors: list[str] = []

    for resume_file in resumes:
        if not resume_file.filename:
            continue
        try:
            content = await resume_file.read()
            text = extract_text(resume_file.filename, content)
            candidate_name = Path(resume_file.filename).stem.replace("_", " ").replace("-", " ")
            candidates.append((candidate_name, text))
        except ValueError as e:
            errors.append(f"{resume_file.filename}: {e}")
        except Exception as e:
            errors.append(f"{resume_file.filename}: Failed to parse — {e}")

    if not candidates:
        return templates.TemplateResponse(
            request,
            "index.html",
            context={
                "error": "No valid resumes were uploaded. " + " ".join(errors),
            },
        )

    results = rank_candidates(job_description, candidates)
    results_data = [_result_to_dict(r) for r in results]

    return templates.TemplateResponse(
        request,
        "results.html",
        context={
            "results": results_data,
            "job_description": job_description,
            "total_candidates": len(results),
            "errors": errors,
        },
    )


@app.post("/api/rank")
async def api_rank_resumes(
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...),
) -> dict:
    """API endpoint returning JSON ranking results."""
    candidates: list[tuple[str, str]] = []
    errors: list[str] = []

    for resume_file in resumes:
        if not resume_file.filename:
            continue
        try:
            content = await resume_file.read()
            text = extract_text(resume_file.filename, content)
            candidate_name = Path(resume_file.filename).stem.replace("_", " ").replace("-", " ")
            candidates.append((candidate_name, text))
        except ValueError as e:
            errors.append(f"{resume_file.filename}: {e}")
        except Exception as e:
            errors.append(f"{resume_file.filename}: Failed to parse — {e}")

    if not candidates:
        return {"error": "No valid resumes provided", "details": errors}

    results = rank_candidates(job_description, candidates)

    return {
        "total_candidates": len(results),
        "rankings": [_result_to_dict(r) for r in results],
        "errors": errors if errors else None,
    }
