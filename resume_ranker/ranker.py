"""Core ranking engine that scores and ranks candidates against a job description."""

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_ranker.experience import extract_required_experience, extract_years_of_experience
from resume_ranker.skills import extract_skills, flatten_skills


@dataclass
class SkillAnalysis:
    """Analysis of skill matching between a candidate and job description."""

    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    match_percentage: float = 0.0


@dataclass
class ExperienceAnalysis:
    """Analysis of experience matching."""

    candidate_years: float | None = None
    required_min: float | None = None
    required_max: float | None = None
    score: float = 0.0
    summary: str = ""


@dataclass
class RolesAnalysis:
    """Analysis of roles and responsibilities matching."""

    similarity_score: float = 0.0
    summary: str = ""


@dataclass
class CandidateResult:
    """Complete ranking result for a single candidate."""

    candidate_name: str
    overall_score: float = 0.0
    rank: int = 0
    skill_analysis: SkillAnalysis = field(default_factory=SkillAnalysis)
    experience_analysis: ExperienceAnalysis = field(default_factory=ExperienceAnalysis)
    roles_analysis: RolesAnalysis = field(default_factory=RolesAnalysis)
    justification: str = ""

    # Weight configuration
    SKILL_WEIGHT: float = 0.45
    EXPERIENCE_WEIGHT: float = 0.25
    ROLES_WEIGHT: float = 0.30


def _analyze_skills(jd_text: str, resume_text: str) -> SkillAnalysis:
    """Compare skills between job description and resume."""
    jd_skills_categorized = extract_skills(jd_text)
    resume_skills_categorized = extract_skills(resume_text)

    jd_skills = flatten_skills(jd_skills_categorized)
    resume_skills = flatten_skills(resume_skills_categorized)

    if not jd_skills:
        return SkillAnalysis(
            matched_skills=sorted(resume_skills),
            missing_skills=[],
            extra_skills=sorted(resume_skills),
            match_percentage=100.0 if resume_skills else 0.0,
        )

    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    match_pct = (len(matched) / len(jd_skills)) * 100.0

    return SkillAnalysis(
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        match_percentage=match_pct,
    )


def _analyze_experience(jd_text: str, resume_text: str) -> ExperienceAnalysis:
    """Compare experience requirements against candidate's experience."""
    candidate_years = extract_years_of_experience(resume_text)
    req_min, req_max = extract_required_experience(jd_text)

    if req_min is None and candidate_years is None:
        return ExperienceAnalysis(
            score=50.0,
            summary="Experience requirements and candidate experience could not be determined.",
        )

    if req_min is None:
        return ExperienceAnalysis(
            candidate_years=candidate_years,
            score=70.0,
            summary=(
                f"Candidate has {candidate_years:.0f} years of experience. "
                "No specific requirement found in job description."
            ),
        )

    if candidate_years is None:
        return ExperienceAnalysis(
            required_min=req_min,
            required_max=req_max,
            score=30.0,
            summary=(
                f"Job requires {req_min:.0f}"
                + (f"-{req_max:.0f}" if req_max else "+")
                + " years. Could not determine candidate's experience."
            ),
        )

    # Score based on how well candidate meets requirements
    if req_max is not None:
        if req_min <= candidate_years <= req_max:
            score = 100.0
            summary = (
                f"Candidate's {candidate_years:.0f} years perfectly fits "
                f"the {req_min:.0f}-{req_max:.0f} year requirement."
            )
        elif candidate_years > req_max:
            over = candidate_years - req_max
            score = max(60.0, 100.0 - over * 5)
            summary = (
                f"Candidate has {candidate_years:.0f} years, exceeding "
                f"the {req_min:.0f}-{req_max:.0f} year range (overqualified)."
            )
        else:
            deficit = req_min - candidate_years
            score = max(10.0, 100.0 - deficit * 15)
            summary = (
                f"Candidate has {candidate_years:.0f} years, below "
                f"the {req_min:.0f}-{req_max:.0f} year requirement."
            )
    else:
        if candidate_years >= req_min:
            excess = candidate_years - req_min
            score = min(100.0, 80.0 + excess * 4)
            summary = (
                f"Candidate's {candidate_years:.0f} years meets or exceeds "
                f"the {req_min:.0f}+ year requirement."
            )
        else:
            deficit = req_min - candidate_years
            score = max(10.0, 80.0 - deficit * 15)
            summary = (
                f"Candidate has {candidate_years:.0f} years, below "
                f"the {req_min:.0f}+ year requirement."
            )

    return ExperienceAnalysis(
        candidate_years=candidate_years,
        required_min=req_min,
        required_max=req_max,
        score=score,
        summary=summary,
    )


def _analyze_roles(jd_text: str, resume_text: str) -> RolesAnalysis:
    """Analyze similarity of roles and responsibilities using TF-IDF cosine similarity."""
    if not jd_text.strip() or not resume_text.strip():
        return RolesAnalysis(
            similarity_score=0.0,
            summary="Insufficient text for roles comparison.",
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        return RolesAnalysis(
            similarity_score=0.0,
            summary="Could not compute roles similarity due to insufficient content.",
        )

    score = similarity * 100.0

    if score >= 70:
        level = "Strong"
    elif score >= 40:
        level = "Moderate"
    elif score >= 20:
        level = "Partial"
    else:
        level = "Low"

    return RolesAnalysis(
        similarity_score=score,
        summary=(
            f"{level} alignment ({score:.1f}%) between candidate's experience "
            "and job responsibilities."
        ),
    )


def _generate_justification(result: CandidateResult) -> str:
    """Generate a human-readable justification for the candidate's ranking."""
    parts: list[str] = []

    parts.append(f"Overall Score: {result.overall_score:.1f}/100 (Rank #{result.rank})")
    parts.append("")

    # Skills
    sa = result.skill_analysis
    parts.append(f"SKILLS ({sa.match_percentage:.0f}% match, weight: 45%)")
    if sa.matched_skills:
        parts.append(f"  Matched: {', '.join(sa.matched_skills)}")
    if sa.missing_skills:
        parts.append(f"  Missing: {', '.join(sa.missing_skills)}")
    if sa.extra_skills:
        parts.append(f"  Additional: {', '.join(sa.extra_skills)}")
    parts.append("")

    # Experience
    parts.append(f"EXPERIENCE (score: {result.experience_analysis.score:.0f}/100, weight: 25%)")
    parts.append(f"  {result.experience_analysis.summary}")
    parts.append("")

    # Roles
    parts.append(
        f"ROLES & RESPONSIBILITIES "
        f"(score: {result.roles_analysis.similarity_score:.0f}/100, weight: 30%)"
    )
    parts.append(f"  {result.roles_analysis.summary}")

    return "\n".join(parts)


def rank_candidates(
    jd_text: str,
    candidates: list[tuple[str, str]],
) -> list[CandidateResult]:
    """Rank candidates against a job description.

    Args:
        jd_text: The job description text.
        candidates: List of (candidate_name, resume_text) tuples.

    Returns:
        List of CandidateResult sorted by overall_score descending.
    """
    results: list[CandidateResult] = []

    for name, resume_text in candidates:
        skill_analysis = _analyze_skills(jd_text, resume_text)
        exp_analysis = _analyze_experience(jd_text, resume_text)
        roles_analysis = _analyze_roles(jd_text, resume_text)

        overall = (
            skill_analysis.match_percentage * CandidateResult.SKILL_WEIGHT
            + exp_analysis.score * CandidateResult.EXPERIENCE_WEIGHT
            + roles_analysis.similarity_score * CandidateResult.ROLES_WEIGHT
        )

        result = CandidateResult(
            candidate_name=name,
            overall_score=overall,
            skill_analysis=skill_analysis,
            experience_analysis=exp_analysis,
            roles_analysis=roles_analysis,
        )
        results.append(result)

    # Sort by overall score descending
    results.sort(key=lambda r: r.overall_score, reverse=True)

    # Assign ranks
    for i, result in enumerate(results, start=1):
        result.rank = i
        result.justification = _generate_justification(result)

    return results
