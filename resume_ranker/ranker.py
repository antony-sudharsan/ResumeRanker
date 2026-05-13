"""Core ranking engine — LinkedIn-style scoring with multiple ranking signals."""

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_ranker.experience import extract_required_experience, extract_years_of_experience
from resume_ranker.skills import extract_skills, flatten_skills, infer_skills


@dataclass
class SkillAnalysis:
    """Analysis of skill matching between a candidate and job description."""

    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    inferred_skills: list[str] = field(default_factory=list)
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
class SemanticAnalysis:
    """Analysis of semantic/meaning-based matching."""

    score: float = 0.0
    summary: str = ""


@dataclass
class TitleAnalysis:
    """Analysis of job title relevance."""

    jd_title: str = ""
    candidate_titles: list[str] = field(default_factory=list)
    score: float = 0.0
    summary: str = ""


@dataclass
class LocationAnalysis:
    """Analysis of location/relocation match."""

    jd_location: str = ""
    candidate_location: str = ""
    relocation_ready: bool = False
    score: float = 0.0
    summary: str = ""


@dataclass
class ProfileCompletenessAnalysis:
    """Analysis of resume/profile completeness."""

    sections_found: list[str] = field(default_factory=list)
    sections_missing: list[str] = field(default_factory=list)
    word_count: int = 0
    score: float = 0.0
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
    semantic_analysis: SemanticAnalysis = field(default_factory=SemanticAnalysis)
    title_analysis: TitleAnalysis = field(default_factory=TitleAnalysis)
    location_analysis: LocationAnalysis = field(default_factory=LocationAnalysis)
    completeness_analysis: ProfileCompletenessAnalysis = field(
        default_factory=ProfileCompletenessAnalysis
    )
    justification: str = ""

    # LinkedIn-style weight configuration
    SKILL_WEIGHT: float = 0.25
    SEMANTIC_WEIGHT: float = 0.20
    TITLE_WEIGHT: float = 0.10
    EXPERIENCE_WEIGHT: float = 0.15
    LOCATION_WEIGHT: float = 0.10
    ROLES_WEIGHT: float = 0.10
    COMPLETENESS_WEIGHT: float = 0.10


# ---------------------------------------------------------------------------
# JD section parsing
# ---------------------------------------------------------------------------

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
    """
    lines = jd_text.split("\n")

    pref_range = _find_section_range(lines, _PREFERRED_HEADERS)
    resp_range = _find_section_range(lines, _RESPONSIBILITIES_HEADERS)
    req_range = _find_section_range(lines, _REQUIRED_SKILLS_HEADERS)

    pref_text = "\n".join(lines[pref_range[0] : pref_range[1]]) if pref_range else ""
    resp_text = "\n".join(lines[resp_range[0] : resp_range[1]]) if resp_range else ""

    if req_range:
        req_text = "\n".join(lines[req_range[0] : req_range[1]])
    else:
        exclude = set()
        for r in (pref_range, resp_range):
            if r:
                exclude.update(range(r[0], r[1]))
        req_text = "\n".join(line for i, line in enumerate(lines) if i not in exclude)

    return req_text, pref_text, resp_text


# ---------------------------------------------------------------------------
# Individual analysis functions
# ---------------------------------------------------------------------------


def _analyze_skills(jd_text: str, resume_text: str) -> SkillAnalysis:
    """Compare skills between job description and resume.

    Scores only against required-section skills. Includes skill inference
    to detect implied skills from the candidate's profile.
    """
    required_jd, preferred_jd, _resp_jd = _split_jd_sections(jd_text)

    required_skills = flatten_skills(extract_skills(required_jd))
    preferred_skills = flatten_skills(extract_skills(preferred_jd)) - required_skills
    resume_skills = flatten_skills(extract_skills(resume_text))

    # Infer additional skills from what the candidate explicitly has
    inferred = infer_skills(resume_skills)

    all_jd_skills = required_skills | preferred_skills

    if not all_jd_skills:
        return SkillAnalysis(
            matched_skills=sorted(resume_skills),
            missing_skills=[],
            extra_skills=sorted(resume_skills),
            inferred_skills=sorted(inferred),
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
        inferred_skills=sorted(inferred),
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


def _analyze_semantic(jd_text: str, resume_text: str) -> SemanticAnalysis:
    """Semantic AI matching using sentence-transformer embeddings."""
    from resume_ranker.semantic import semantic_similarity_chunked

    sim = semantic_similarity_chunked(jd_text, resume_text)
    # Scale: 0.6+ raw semantic similarity is very strong for JD-resume
    score = min(100.0, sim * 150.0)

    if score >= 70:
        level = "Strong"
    elif score >= 40:
        level = "Moderate"
    elif score >= 20:
        level = "Partial"
    else:
        level = "Low"

    return SemanticAnalysis(
        score=score,
        summary=f"{level} semantic alignment ({score:.1f}%). AI-based meaning match.",
    )


def _analyze_title(jd_text: str, resume_text: str) -> TitleAnalysis:
    """Analyze job title relevance."""
    from resume_ranker.signals import analyze_title_relevance

    result = analyze_title_relevance(jd_text, resume_text)
    return TitleAnalysis(
        jd_title=result.jd_title,
        candidate_titles=result.candidate_titles,
        score=result.score,
        summary=result.summary,
    )


def _analyze_location(jd_text: str, resume_text: str) -> LocationAnalysis:
    """Analyze location alignment."""
    from resume_ranker.signals import analyze_location

    result = analyze_location(jd_text, resume_text)
    return LocationAnalysis(
        jd_location=result.jd_location,
        candidate_location=result.candidate_location,
        relocation_ready=result.relocation_ready,
        score=result.score,
        summary=result.summary,
    )


def _analyze_completeness(resume_text: str) -> ProfileCompletenessAnalysis:
    """Analyze profile/resume completeness."""
    from resume_ranker.signals import analyze_profile_completeness

    result = analyze_profile_completeness(resume_text)
    return ProfileCompletenessAnalysis(
        sections_found=result.sections_found,
        sections_missing=result.sections_missing,
        word_count=result.word_count,
        score=result.score,
        summary=result.summary,
    )


# ---------------------------------------------------------------------------
# Justification
# ---------------------------------------------------------------------------


def _generate_justification(result: CandidateResult) -> str:
    """Generate a human-readable justification for the candidate's ranking."""
    parts: list[str] = []

    parts.append(f"Overall Score: {result.overall_score:.1f}/100 (Rank #{result.rank})")
    parts.append("")

    # Skills
    sa = result.skill_analysis
    parts.append(
        f"SKILLS ({sa.match_percentage:.0f}% match, weight: {CandidateResult.SKILL_WEIGHT:.0%})"
    )
    if sa.matched_skills:
        parts.append(f"  Matched: {', '.join(sa.matched_skills)}")
    if sa.missing_skills:
        parts.append(f"  Missing: {', '.join(sa.missing_skills)}")
    if sa.extra_skills:
        parts.append(f"  Additional: {', '.join(sa.extra_skills)}")
    if sa.inferred_skills:
        parts.append(f"  Inferred: {', '.join(sa.inferred_skills)}")
    parts.append("")

    # Semantic
    sem = result.semantic_analysis
    parts.append(
        f"SEMANTIC AI MATCH (score: {sem.score:.0f}/100, weight: "
        f"{CandidateResult.SEMANTIC_WEIGHT:.0%})"
    )
    parts.append(f"  {sem.summary}")
    parts.append("")

    # Title
    ta = result.title_analysis
    parts.append(
        f"TITLE RELEVANCE (score: {ta.score:.0f}/100, weight: {CandidateResult.TITLE_WEIGHT:.0%})"
    )
    parts.append(f"  {ta.summary}")
    parts.append("")

    # Experience
    parts.append(
        f"EXPERIENCE (score: {result.experience_analysis.score:.0f}/100, weight: "
        f"{CandidateResult.EXPERIENCE_WEIGHT:.0%})"
    )
    parts.append(f"  {result.experience_analysis.summary}")
    parts.append("")

    # Location
    loc = result.location_analysis
    parts.append(
        f"LOCATION (score: {loc.score:.0f}/100, weight: {CandidateResult.LOCATION_WEIGHT:.0%})"
    )
    parts.append(f"  {loc.summary}")
    parts.append("")

    # Roles
    parts.append(
        f"ROLES & RESPONSIBILITIES "
        f"(score: {result.roles_analysis.similarity_score:.0f}/100, weight: "
        f"{CandidateResult.ROLES_WEIGHT:.0%})"
    )
    parts.append(f"  {result.roles_analysis.summary}")
    parts.append("")

    # Completeness
    comp = result.completeness_analysis
    parts.append(
        f"PROFILE COMPLETENESS (score: {comp.score:.0f}/100, weight: "
        f"{CandidateResult.COMPLETENESS_WEIGHT:.0%})"
    )
    parts.append(f"  {comp.summary}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------


def rank_candidates(
    jd_text: str,
    candidates: list[tuple[str, str]],
) -> list[CandidateResult]:
    """Rank candidates against a job description using LinkedIn-style signals.

    Scoring weights:
        Skills Match:         25%  (keyword matching with skill inference)
        Semantic AI Match:    20%  (meaning-based NLP matching)
        Title Relevance:      10%  (job title alignment)
        Experience:           15%  (years of experience)
        Location Match:       10%  (location/relocation readiness)
        Roles Alignment:      10%  (TF-IDF responsibilities matching)
        Profile Completeness: 10%  (resume detail level)
    """
    results: list[CandidateResult] = []

    for name, resume_text in candidates:
        skill_analysis = _analyze_skills(jd_text, resume_text)
        exp_analysis = _analyze_experience(jd_text, resume_text)
        roles_analysis = _analyze_roles(jd_text, resume_text)
        semantic_analysis = _analyze_semantic(jd_text, resume_text)
        title_analysis = _analyze_title(jd_text, resume_text)
        location_analysis = _analyze_location(jd_text, resume_text)
        completeness_analysis = _analyze_completeness(resume_text)

        overall = (
            skill_analysis.match_percentage * CandidateResult.SKILL_WEIGHT
            + semantic_analysis.score * CandidateResult.SEMANTIC_WEIGHT
            + title_analysis.score * CandidateResult.TITLE_WEIGHT
            + exp_analysis.score * CandidateResult.EXPERIENCE_WEIGHT
            + location_analysis.score * CandidateResult.LOCATION_WEIGHT
            + roles_analysis.similarity_score * CandidateResult.ROLES_WEIGHT
            + completeness_analysis.score * CandidateResult.COMPLETENESS_WEIGHT
        )

        result = CandidateResult(
            candidate_name=name,
            overall_score=overall,
            skill_analysis=skill_analysis,
            experience_analysis=exp_analysis,
            roles_analysis=roles_analysis,
            semantic_analysis=semantic_analysis,
            title_analysis=title_analysis,
            location_analysis=location_analysis,
            completeness_analysis=completeness_analysis,
        )
        results.append(result)

    results.sort(key=lambda r: r.overall_score, reverse=True)

    for i, result in enumerate(results, start=1):
        result.rank = i
        result.justification = _generate_justification(result)

    return results
