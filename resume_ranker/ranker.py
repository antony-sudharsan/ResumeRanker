"""Hybrid resume ranking engine — skills, experience, roles, and designation signals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from resume_ranker.experience import (
    analyze_experience_full,
    extract_required_experience,
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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
class TitleAnalysis:
    """Analysis of designation / job title relevance."""

    jd_title: str = ""
    candidate_titles: list[dict] = field(default_factory=list)
    score: float = 0.0
    summary: str = ""
    debug: dict = field(default_factory=dict)


@dataclass
class RolesAnalysis:
    """Analysis of roles & responsibilities matching."""

    score: float = 0.0
    summary: str = ""
    tfidf_score: float = 0.0
    semantic_score: float = 0.0
    action_verb_score: float = 0.0
    section_confidence: float = 0.0
    best_role_matches: list[dict] = field(default_factory=list)


@dataclass
class CandidateResult:
    """Complete ranking result for a single candidate."""

    candidate_name: str
    overall_score: float = 0.0
    rank: int = 0
    skill_analysis: SkillAnalysis = field(default_factory=SkillAnalysis)
    experience_analysis: ExperienceAnalysis = field(default_factory=ExperienceAnalysis)
    title_analysis: TitleAnalysis = field(default_factory=TitleAnalysis)
    roles_analysis: RolesAnalysis = field(default_factory=RolesAnalysis)
    justification: str = ""
    match_quality: str = ""

    # Weight configuration
    SKILL_WEIGHT: float = 0.40
    EXPERIENCE_WEIGHT: float = 0.25
    ROLES_WEIGHT: float = 0.25
    DESIGNATION_WEIGHT: float = 0.10


def _match_quality_label(score: float) -> str:
    if score >= 90:
        return "Excellent match"
    elif score >= 80:
        return "Strong match"
    elif score >= 70:
        return "Moderate match"
    elif score >= 50:
        return "Weak match"
    elif score >= 1:
        return "Poor match"
    return "No match"


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
            match_percentage=50.0,
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
    candidate_years = extract_years_of_experience(resume_text)
    req_min, req_max = extract_required_experience(jd_text)

    jd_responsibilities = ""
    jd_skills: set[str] = set()
    try:
        required_jd, _preferred_jd, _resp_jd = _split_jd_sections(jd_text)
        jd_skills = flatten_skills(extract_skills(required_jd))
        jd_responsibilities = _resp_jd
    except Exception:
        pass

    jd_title = ""
    try:
        from resume_ranker.signals import extract_jd_title

        title_result = extract_jd_title(jd_text)
        if isinstance(title_result, str):
            jd_title = title_result
    except Exception:
        pass

    exp_debug = analyze_experience_full(
        resume_text, jd_text, jd_skills, jd_title, jd_responsibilities
    )

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

    experience_score = exp_debug.get("experience_score")
    total_experience_fit: float | None = exp_debug.get("total_experience_fit")
    relevant_experience_fit: float | None = exp_debug.get("relevant_experience_fit")
    recency_score: float | None = exp_debug.get("recency_score")
    seniority_fit: float | None = exp_debug.get("seniority_fit")
    total_years_new = exp_debug.get("total_years_new")
    relevant_years_new = exp_debug.get("relevant_years_new")
    experience_ranking_periods = exp_debug.get("experience_ranking_periods", [])

    use_new_scoring = experience_score is not None
    effective_score = experience_score if use_new_scoring else final_score

    jd_req_str = (
        f"{req_min:.0f}" + (f"-{req_max:.0f}" if req_max else "+") if req_min is not None else "N/A"
    )

    summary_parts: list[str] = []
    if use_new_scoring:
        summary_parts.append(f"Score: {effective_score:.1f}")
        summary_parts.append(
            f"Total: {total_years_new:.1f}y"
            if total_years_new is not None
            else f"Total: {total_years:.1f}y"
            if total_years is not None
            else ""
        )
        summary_parts.append(
            f"Relevant: {relevant_years_new:.1f}y"
            if relevant_years_new is not None
            else f"Relevant: {relevant_years:.1f}y"
            if relevant_years is not None
            else ""
        )
    else:
        if total_years is not None:
            summary_parts.append(f"Total: {total_years:.1f}y")
        if relevant_years is not None:
            summary_parts.append(f"Relevant: {relevant_years:.1f}y")
    if req_min is not None:
        summary_parts.append(f"Required: {jd_req_str}y")

    summary = " | ".join(p for p in summary_parts if p) if summary_parts else "No experience data"

    debug_info = dict(exp_debug)
    if use_new_scoring:
        debug_info.update(
            {
                "experience_score": experience_score,
                "total_experience_fit": total_experience_fit,
                "relevant_experience_fit": relevant_experience_fit,
                "recency_score_component": recency_score,
                "seniority_fit_component": seniority_fit,
                "experience_ranking_periods": experience_ranking_periods,
            }
        )

    return ExperienceAnalysis(
        candidate_years=candidate_years,
        required_min=req_min,
        required_max=req_max,
        score=effective_score,
        summary=summary,
        candidate_total_years=total_years_new if use_new_scoring else total_years,
        candidate_relevant_years=relevant_years_new if use_new_scoring else relevant_years,
        total_experience_score=(total_experience_fit if use_new_scoring else total_exp_score)
        or 0.0,
        relevant_experience_score=(
            relevant_experience_fit if use_new_scoring else relevant_exp_score
        )
        or 0.0,
        confidence_score=conf_score,
        source=source,
        matched_periods=matched_periods,
        ignored_periods=ignored_periods,
        warnings=warnings,
        debug=debug_info,
    )


def _find_responsibility_section(jd_text: str) -> str:
    """Find the Roles & Responsibilities section in JD text.

    Handles headers with colons (e.g. 'Roles and Responsibilities:')
    that _split_jd_sections cannot parse."""
    lines = jd_text.split("\n")
    patterns = [
        re.compile(
            r"(?:roles?\s*(?:&|and)\s*responsibilities|responsibilities|duties|what.you.?ll\s*do)",
            re.IGNORECASE,
        ),
    ]
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip().rstrip(":")
        for pat in patterns:
            if pat.search(stripped):
                start = i + 1
                break
        if start is not None:
            break
    if start is None:
        return ""

    end = len(lines)
    for i in range(start, len(lines)):
        stripped = lines[i].strip().rstrip(":")
        if not stripped:
            continue
        if re.match(r"^[A-Z][A-Za-z &/,\-–]+$", stripped):
            end = i
            break

    return "\n".join(lines[start:end])


def _analyze_roles(jd_text: str, resume_text: str) -> RolesAnalysis:
    """Analyze roles & responsibilities match using hybrid scoring."""
    from resume_ranker.roles import calculate_roles_score

    resp_section = _find_responsibility_section(jd_text)
    if not resp_section.strip():
        resp_section = jd_text

    result = calculate_roles_score(resp_section, resume_text)
    score = result["roles_score"]

    if score >= 70:
        level = "Strong"
    elif score >= 40:
        level = "Moderate"
    elif score >= 20:
        level = "Weak"
    else:
        level = "Minimal"

    return RolesAnalysis(
        score=score,
        summary=f"{level} roles & responsibilities alignment ({score:.1f}%).",
        tfidf_score=result["tfidf_score"],
        semantic_score=result["semantic_score"],
        action_verb_score=result["action_verb_score"],
        section_confidence=result["section_confidence"],
        best_role_matches=result["best_role_matches"],
    )


def _analyze_designation(jd_text: str, resume_text: str) -> TitleAnalysis:
    """Analyze designation / job title relevance.

    Formula: semantic_similarity × role_family_factor × seniority_factor × recency_factor
    """
    from resume_ranker.signals import (
        analyze_title_relevance,
    )

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

    parts.append(
        f"Overall Score: {result.overall_score:.1f}/100 | {result.match_quality} | Rank #{result.rank}"
    )
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

    # Designation
    ta = result.title_analysis
    parts.append(
        f"DESIGNATION / TITLE (score: {ta.score:.0f}/100, weight: {CandidateResult.DESIGNATION_WEIGHT:.0%})"
    )
    parts.append(f"  {ta.summary}")
    if ta.debug:
        d = ta.debug
        parts.append(
            f"  semantic_sim={d.get('semantic_similarity', '?'):} "
            f"seniority={d.get('seniority_factor', '?'):} "
            f"family={d.get('role_family_factor', '?'):} "
            f"recency={d.get('recency_factor', '?'):} "
            f"→ {d.get('role_family_match', '?'):}"
        )
    parts.append("")

    # Roles & Responsibilities
    ra = result.roles_analysis
    parts.append(
        f"ROLES & RESPONSIBILITIES (score: {ra.score:.0f}/100, weight: {CandidateResult.ROLES_WEIGHT:.0%})"
    )
    parts.append(f"  {ra.summary}")
    if ra.best_role_matches:
        parts.append(f"  Top matches: {len(ra.best_role_matches)} role responsibilities matched")
    parts.append(
        f"  TF-IDF: {ra.tfidf_score:.1f} | Semantic: {ra.semantic_score:.1f} | "
        f"Action Verbs: {ra.action_verb_score:.1f} | Section: {ra.section_confidence:.1f}"
    )
    parts.append("")

    # Experience
    ea = result.experience_analysis
    parts.append(
        f"EXPERIENCE (score: {ea.score:.0f}/100, weight: {CandidateResult.EXPERIENCE_WEIGHT:.0%})"
    )
    parts.append(f"  {ea.summary}")
    if ea.debug.get("experience_score") is not None:
        d = ea.debug
        parts.append(
            f"  Total Fit: {d.get('total_experience_fit', '?'):} | "
            f"Relevant Fit: {d.get('relevant_experience_fit', '?'):} | "
            f"Recency: {d.get('recency_score_component', '?'):} | "
            f"Seniority: {d.get('seniority_fit_component', '?'):}"
        )
    elif ea.source:
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

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------


def rank_candidates(
    jd_text: str,
    candidates: list[tuple[str, str]],
) -> list[CandidateResult]:
    """Rank candidates against a job description using hybrid scoring signals.

    Scoring weights:
        Skills:                  40%  (keyword matching + semantic skill inference)
        Experience:              25%  (total + relevant + recency + seniority)
        Roles & Responsibilities: 25%  (TF-IDF + semantic + action verbs)
        Designation / Title:      10%  (semantic sim × family × seniority × recency)
    """
    results: list[CandidateResult] = []

    for name, resume_text in candidates:
        skill_analysis = _analyze_skills(jd_text, resume_text)
        exp_analysis = _analyze_experience(jd_text, resume_text)
        title_analysis = _analyze_designation(jd_text, resume_text)
        roles_analysis = _analyze_roles(jd_text, resume_text)

        overall = (
            skill_analysis.match_percentage * CandidateResult.SKILL_WEIGHT
            + exp_analysis.score * CandidateResult.EXPERIENCE_WEIGHT
            + roles_analysis.score * CandidateResult.ROLES_WEIGHT
            + title_analysis.score * CandidateResult.DESIGNATION_WEIGHT
        )

        match_quality = _match_quality_label(overall)

        result = CandidateResult(
            candidate_name=name,
            overall_score=overall,
            skill_analysis=skill_analysis,
            experience_analysis=exp_analysis,
            title_analysis=title_analysis,
            roles_analysis=roles_analysis,
            match_quality=match_quality,
        )
        results.append(result)

    results.sort(key=lambda r: r.overall_score, reverse=True)

    for i, result in enumerate(results, start=1):
        result.rank = i
        result.justification = _generate_justification(result)

    return results
