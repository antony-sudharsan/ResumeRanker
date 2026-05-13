"""Core ranking engine that scores and ranks candidates against a job description."""

import re
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


_PREFERRED_HEADERS = re.compile(
    r"(?:preferred|nice\s+to\s+have|good\s+to\s+have|bonus|desirable|optional)\s+"
    r"(?:skills?|qualifications?|requirements?|experience)?",
    re.IGNORECASE,
)
_RESPONSIBILITIES_HEADERS = re.compile(
    r"(?:key\s+)?(?:responsibilities|duties|role\s+description|what\s+you.?ll\s+do)",
    re.IGNORECASE,
)
_REQUIRED_SKILLS_HEADERS = re.compile(
    r"(?:required|must\s+have|essential|key|core|minimum)?\s*"
    r"(?:skills?|qualifications?|requirements?|competencies|technical\s+skills?)",
    re.IGNORECASE,
)
_SECTION_HEADER = re.compile(r"^(?:#{1,4}\s+)?[A-Z][A-Za-z &/,\-–]+$", re.MULTILINE)


def _find_section_range(
    lines: list[str], header_re: re.Pattern[str], start_from: int = 0
) -> tuple[int, int] | None:
    """Find start and end line indices for a section matching header_re."""
    section_start = None
    for i in range(start_from, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        # Line must look like a section header (short, title-case, no trailing period)
        if not _SECTION_HEADER.match(stripped):
            continue
        if header_re.search(stripped):
            section_start = i
            break
    if section_start is None:
        return None
    section_end = len(lines)
    for i in range(section_start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped and _SECTION_HEADER.match(stripped):
            section_end = i
            break
    return section_start, section_end


def _split_jd_sections(jd_text: str) -> tuple[str, str, str]:
    """Split a JD into required-skills, preferred-skills, and responsibilities.

    Returns (required_skills_text, preferred_text, responsibilities_text).
    If a required-skills section is found, only that section is used for skill
    matching. Otherwise the full JD minus preferred/responsibilities is used.
    """
    lines = jd_text.split("\n")

    pref_range = _find_section_range(lines, _PREFERRED_HEADERS)
    resp_range = _find_section_range(lines, _RESPONSIBILITIES_HEADERS)
    req_range = _find_section_range(lines, _REQUIRED_SKILLS_HEADERS)

    pref_text = "\n".join(lines[pref_range[0] : pref_range[1]]) if pref_range else ""
    resp_text = "\n".join(lines[resp_range[0] : resp_range[1]]) if resp_range else ""

    # If a dedicated "Required Skills" section exists, use only that
    if req_range:
        req_text = "\n".join(lines[req_range[0] : req_range[1]])
    else:
        # Fallback: use the full JD minus preferred and responsibilities sections
        exclude = set()
        for r in (pref_range, resp_range):
            if r:
                exclude.update(range(r[0], r[1]))
        req_text = "\n".join(line for i, line in enumerate(lines) if i not in exclude)

    return req_text, pref_text, resp_text


def _analyze_skills(jd_text: str, resume_text: str) -> SkillAnalysis:
    """Compare skills between job description and resume.

    Only skills from the required-skills section of the JD count toward the
    match percentage. Skills found only in preferred or responsibilities
    sections are not penalised as missing.
    """
    required_jd, preferred_jd, _resp_jd = _split_jd_sections(jd_text)

    required_skills = flatten_skills(extract_skills(required_jd))
    preferred_skills = flatten_skills(extract_skills(preferred_jd)) - required_skills
    resume_skills = flatten_skills(extract_skills(resume_text))

    all_jd_skills = required_skills | preferred_skills

    if not all_jd_skills:
        return SkillAnalysis(
            matched_skills=sorted(resume_skills),
            missing_skills=[],
            extra_skills=sorted(resume_skills),
            match_percentage=100.0 if resume_skills else 0.0,
        )

    matched = sorted(all_jd_skills & resume_skills)
    missing = sorted(required_skills - resume_skills)
    extra = sorted(resume_skills - all_jd_skills)

    if required_skills:
        match_pct = (len(required_skills & resume_skills) / len(required_skills)) * 100.0
    else:
        match_pct = 100.0

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
            score = min(100.0, 95.0 + excess * 1)
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
        raw_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        return RolesAnalysis(
            similarity_score=0.0,
            summary="Could not compute roles similarity due to insufficient content.",
        )

    # Raw TF-IDF cosine similarity between a JD and a resume rarely exceeds 0.5
    # because they use different writing styles. Scale so that 0.5+ raw → 90-100.
    score = min(100.0, raw_similarity * 200.0)

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
