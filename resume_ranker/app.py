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


def _build_jd_text(
    designation: str,
    skills: str,
    min_experience: str,
    max_experience: str,
    roles_responsibilities: str,
) -> str:
    """Build a structured JD string from form fields that existing parsers understand."""
    parts = [f"Job Title: {designation}"]
    if min_experience or max_experience:
        min_val = min_experience if min_experience else "0"
        max_val = max_experience if max_experience else ""
        exp_str = f"{min_val}-{max_val} years" if max_val else f"{min_val}+ years"
        parts.append(f"\nExperience: {exp_str}")
    if skills.strip():
        parts.append(f"\nRequired Skills\n{skills}")
    if roles_responsibilities.strip():
        parts.append(f"\nRoles and Responsibilities\n{roles_responsibilities}")
    return "\n".join(parts)


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
            "inferred": r.skill_analysis.inferred_skills,
        },
        "semantic": {
            "score": round(r.semantic_analysis.score, 1),
            "summary": r.semantic_analysis.summary,
        },
        "title": {
            "score": round(r.title_analysis.score, 1),
            "jd_title": r.title_analysis.jd_title,
            "candidate_titles": [t["title"] for t in r.title_analysis.candidate_titles],
            "candidate_titles_detail": r.title_analysis.candidate_titles,
            "summary": r.title_analysis.summary,
            "debug": r.title_analysis.debug,
        },
        "experience": {
            "score": round(r.experience_analysis.score, 1),
            "candidate_years": r.experience_analysis.candidate_years,
            "required_min": r.experience_analysis.required_min,
            "required_max": r.experience_analysis.required_max,
            "summary": r.experience_analysis.summary,
            "candidate_total_years": r.experience_analysis.candidate_total_years,
            "candidate_relevant_years": r.experience_analysis.candidate_relevant_years,
            "total_experience_score": round(r.experience_analysis.total_experience_score, 1),
            "relevant_experience_score": round(r.experience_analysis.relevant_experience_score, 1),
            "confidence_score": round(r.experience_analysis.confidence_score, 1),
            "source": r.experience_analysis.source,
            "warnings": r.experience_analysis.warnings,
        },
        "justification": r.justification,
    }


@app.post("/rank", response_class=HTMLResponse)
async def rank_resumes(
    request: Request,
    designation: str = Form(...),
    skills: str = Form(""),
    min_experience: str = Form(""),
    max_experience: str = Form(""),
    roles_responsibilities: str = Form(""),
    resumes: list[UploadFile] = File(...),
) -> HTMLResponse:
    """Process uploaded resumes and rank them against the structured job description."""
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

    jd_text = _build_jd_text(
        designation, skills, min_experience, max_experience, roles_responsibilities
    )
    results = rank_candidates(jd_text, candidates)
    results_data = [_result_to_dict(r) for r in results]

    return templates.TemplateResponse(
        request,
        "results.html",
        context={
            "results": results_data,
            "jd_designation": designation,
            "jd_skills": skills,
            "jd_min_exp": min_experience,
            "jd_max_exp": max_experience,
            "total_candidates": len(results),
            "errors": errors,
        },
    )


@app.post("/api/rank")
async def api_rank_resumes(
    designation: str = Form(...),
    skills: str = Form(""),
    min_experience: str = Form(""),
    max_experience: str = Form(""),
    roles_responsibilities: str = Form(""),
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

    jd_text = _build_jd_text(
        designation, skills, min_experience, max_experience, roles_responsibilities
    )
    results = rank_candidates(jd_text, candidates)

    return {
        "total_candidates": len(results),
        "rankings": [_result_to_dict(r) for r in results],
        "jd": {
            "designation": designation,
            "skills": skills,
            "min_experience": min_experience,
            "max_experience": max_experience,
            "roles_responsibilities": roles_responsibilities,
        },
        "errors": errors if errors else None,
    }
