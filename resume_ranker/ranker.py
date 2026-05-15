"""Core ranking engine — LinkedIn-style scoring with multiple ranking signals."""

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_ranker.experience import (
    analyze_experience_full,
    extract_required_experience,
    extract_required_experience_detailed,
    extract_years_of_experience,
)
from resume_ranker.skills import (
    SemanticMatchResult,
    _get_overall_skill_context,
    extract_skills,
    extract_unknown_skills_semantic,
    flatten_skills,
    infer_skills,
)


@dataclass
class SkillAnalysis:
    """Analysis of skill matching between a candidate and job description."""

    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    inferred_skills: list[str] = field(default_factory=list)
    match_percentage: float = 0.0
    semantic_match_details: list[SemanticMatchResult] = field(default_factory=list)


@dataclass
class ExperienceAnalysis:
    """Analysis of experience matching."""

    candidate_years: float | None = None
    required_min: float | None = None
    required_max: float | None = None
    score: float = 0.0
    summary: str = ""

    # New fields for enhanced experience analysis
    candidate_total_years: float | None = None
    candidate_relevant_years: float | None = None
    total_experience_score: float = 0.0
    relevant_experience_score: float = 0.0
    confidence_score: float = 0.0
    source: str = ""
    matched_periods: list = field(default_factory=list)
    ignored_periods: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)


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
    candidate_titles: list[dict] = field(default_factory=list)
    score: float = 0.0
    summary: str = ""
    debug: dict = field(default_factory=dict)


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
    justification: str = ""

    # LinkedIn-style weight configuration
    SKILL_WEIGHT: float = 0.40
    SEMANTIC_WEIGHT: float = 0.20
    TITLE_WEIGHT: float = 0.10
    EXPERIENCE_WEIGHT: float = 0.20
    ROLES_WEIGHT: float = 0.10


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
_SECTION_HEADER = re.compile(r"^(?:#{1,2}\s+)?[A-Z][A-Za-z &/,\-–]+$", re.MULTILINE)


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
    to detect implied skills from the candidate's profile, plus a semantic
    fallback that catches JD skills not yet in the dictionary.
    """
    required_jd, preferred_jd, _resp_jd = _split_jd_sections(jd_text)

    # 1. Dictionary-based skill extraction
    required_skills = flatten_skills(extract_skills(required_jd))
    preferred_skills = flatten_skills(extract_skills(preferred_jd)) - required_skills
    resume_skills = flatten_skills(extract_skills(resume_text))

    # 2. Skill inference from explicit skills
    inferred = infer_skills(resume_skills)

    # 3. Semantic fallback for JD skills not in the dictionary
    #    Returns high-confidence matches, all unknown skills, and full match details
    #    including rejected (negated/weak) semantic matches with context labels.
    sem_matched, unknown_all, sem_details = extract_unknown_skills_semantic(
        required_jd, resume_text
    )
    sem_matched -= required_skills | preferred_skills
    unknown_all -= required_skills | preferred_skills

    if unknown_all:
        required_skills |= unknown_all
    if sem_matched:
        resume_skills |= sem_matched

    # 4. Context-aware filtering: remove skills that are ONLY mentioned in
    #    non-committal contexts. This catches false positives from Tier 1
    #    dictionary matching where a skill name appears literally but in a
    #    negated, learning-only, or weak-exposure context (e.g. "No experience
    #    with Docker", "Interested to learn machine learning").
    #    The context is still available in semantic_match_details for debugging.
    filtered_resume_skills: set[str] = set()
    for skill in resume_skills:
        ctx = _get_overall_skill_context(resume_text, skill)
        if ctx == "strong" or ctx == "unknown":
            filtered_resume_skills.add(skill)
    resume_skills = filtered_resume_skills

    all_jd_skills = required_skills | preferred_skills

    if not all_jd_skills:
        return SkillAnalysis(
            matched_skills=sorted(resume_skills),
            missing_skills=[],
            extra_skills=sorted(resume_skills),
            inferred_skills=sorted(inferred),
            match_percentage=100.0 if resume_skills else 0.0,
            semantic_match_details=sem_details,
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
        semantic_match_details=sem_details,
    )


def _analyze_experience(jd_text: str, resume_text: str) -> ExperienceAnalysis:
    """Compare experience requirements against candidate's experience."""
    # Use the old-school explicit years as the baseline candidate_years
    candidate_years = extract_years_of_experience(resume_text)
    req_min, req_max = extract_required_experience(jd_text)

    # Gather JD skills and title for relevance calculation
    try:
        from resume_ranker.skills import extract_skills, flatten_skills

        required_jd, _preferred_jd, _resp_jd = _split_jd_sections(jd_text)
        jd_skills = flatten_skills(extract_skills(required_jd))
    except Exception:
        jd_skills = set()

    jd_title = ""
    try:
        from resume_ranker.signals import extract_jd_title

        title_result = extract_jd_title(jd_text)
        if isinstance(title_result, str):
            jd_title = title_result
    except Exception:
        pass

    # Run the full experience analysis pipeline
    exp_debug = analyze_experience_full(resume_text, jd_text, jd_skills, jd_title)

    # Map results to ExperienceAnalysis fields
    total_years = exp_debug.get("candidate_total_years")
    relevant_years = exp_debug.get("candidate_relevant_years")
    final_score = exp_debug.get("final_experience_score", 0.0)
    total_exp_score = exp_debug.get("total_experience_score", 0.0)
    relevant_exp_score = exp_debug.get("relevant_experience_score", 0.0)
    conf_score = exp_debug.get("confidence_score", 0.0)
    source = exp_debug.get("source", "unknown")
    matched_periods = exp_debug.get("matched_periods", [])
    ignored_periods = exp_debug.get("ignored_periods", [])
    warnings = exp_debug.get("warnings", [])

    # Build a human-readable summary
    jd_req_str = (
        f"{req_min:.0f}" + (f"-{req_max:.0f}" if req_max else "+") if req_min is not None else "N/A"
    )

    summary_parts: list[str] = []
    if total_years is not None:
        summary_parts.append(f"Total: {total_years:.1f}y")
    if relevant_years is not None:
        summary_parts.append(f"Relevant: {relevant_years:.1f}y")
    if req_min is not None:
        summary_parts.append(f"Required: {jd_req_str}y")

    summary = " | ".join(summary_parts) if summary_parts else "No experience data"

    return ExperienceAnalysis(
        candidate_years=candidate_years,
        required_min=req_min,
        required_max=req_max,
        score=final_score,
        summary=summary,
        candidate_total_years=total_years,
        candidate_relevant_years=relevant_years,
        total_experience_score=total_exp_score,
        relevant_experience_score=relevant_exp_score,
        confidence_score=conf_score,
        source=source,
        matched_periods=matched_periods,
        ignored_periods=ignored_periods,
        warnings=warnings,
        debug=exp_debug,
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
    """Semantic AI matching using sentence-transformer embeddings.

    Improvements over basic chunked similarity:
      1. Requirements-focused blending — weights the "Requirements" section 3:2 over full JD
      2. Optional cross-encoder re-ranking (70/30 blend with bi-encoder if available)
      3. Calibrated scaling — maps 0.2-0.8 cosine similarity to 0-100 score
    """
    from resume_ranker.semantic import cross_encoder_score, semantic_similarity_chunked

    required_jd, _preferred_jd, _resp_jd = _split_jd_sections(jd_text)

    # Full JD semantic match (baseline)
    full_sim = semantic_similarity_chunked(jd_text, resume_text)

    # Requirements-focused match (penalizes candidates who match boilerplate but not reqs)
    if required_jd.strip():
        req_sim = semantic_similarity_chunked(required_jd, resume_text)
    else:
        req_sim = full_sim

    # Blend: 60% requirements-focused, 40% full JD
    # This ensures requirement alignment matters more than generic JD overlap
    sim = 0.6 * req_sim + 0.4 * full_sim

    # Cross-encoder re-ranking (optional — blends in if model is available)
    ce = cross_encoder_score(jd_text, resume_text)
    if ce is not None:
        sim = 0.7 * ce + 0.3 * sim

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
        debug=result.debug,
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
    if ta.debug:
        d = ta.debug
        parts.append(
            f"  sim={d.get('semantic_similarity', '?'):} "
            f"seniority={d.get('seniority_factor', '?'):} "
            f"family={d.get('role_family_factor', '?'):} "
            f"recency={d.get('recency_factor', '?'):} "
            f"→ {d.get('role_family_match', '?'):}"
        )
    parts.append("")

    # Experience
    ea = result.experience_analysis
    parts.append(
        f"EXPERIENCE (score: {ea.score:.0f}/100, weight: {CandidateResult.EXPERIENCE_WEIGHT:.0%})"
    )
    parts.append(f"  {ea.summary}")
    if ea.source:
        source_labels = {
            "max_of_both": "explicit + dates (max)",
            "date_based_preferred": "date-based (preferred)",
            "explicit_preferred": "explicit (preferred)",
            "explicit_only": "explicit years",
            "date_based_only": "date-based",
        }
        label = source_labels.get(ea.source, ea.source)
        conf = f" | Confidence: {ea.confidence_score:.0f}/100" if ea.confidence_score else ""
        parts.append(f"  Source: {label}{conf}")
    if ea.warnings:
        for w in ea.warnings:
            parts.append(f"  ⚠ {w}")
    parts.append("")

    # Roles
    parts.append(
        f"ROLES & RESPONSIBILITIES "
        f"(score: {result.roles_analysis.similarity_score:.0f}/100, weight: "
        f"{CandidateResult.ROLES_WEIGHT:.0%})"
    )
    parts.append(f"  {result.roles_analysis.summary}")
    parts.append("")

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
        Skills Match:         40%  (keyword matching with skill inference)
        Semantic AI Match:    20%  (meaning-based NLP matching)
        Title Relevance:      10%  (job title alignment)
        Experience:           20%  (years of experience)
        Roles Alignment:      10%  (TF-IDF responsibilities matching)
    """
    results: list[CandidateResult] = []

    for name, resume_text in candidates:
        skill_analysis = _analyze_skills(jd_text, resume_text)
        exp_analysis = _analyze_experience(jd_text, resume_text)
        roles_analysis = _analyze_roles(jd_text, resume_text)
        semantic_analysis = _analyze_semantic(jd_text, resume_text)
        title_analysis = _analyze_title(jd_text, resume_text)
        overall = (
            skill_analysis.match_percentage * CandidateResult.SKILL_WEIGHT
            + semantic_analysis.score * CandidateResult.SEMANTIC_WEIGHT
            + title_analysis.score * CandidateResult.TITLE_WEIGHT
            + exp_analysis.score * CandidateResult.EXPERIENCE_WEIGHT
            + roles_analysis.similarity_score * CandidateResult.ROLES_WEIGHT
        )

        result = CandidateResult(
            candidate_name=name,
            overall_score=overall,
            skill_analysis=skill_analysis,
            experience_analysis=exp_analysis,
            roles_analysis=roles_analysis,
            semantic_analysis=semantic_analysis,
            title_analysis=title_analysis,
        )
        results.append(result)

    results.sort(key=lambda r: r.overall_score, reverse=True)

    for i, result in enumerate(results, start=1):
        result.rank = i
        result.justification = _generate_justification(result)

    return results
