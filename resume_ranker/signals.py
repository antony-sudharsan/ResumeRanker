"""LinkedIn-style ranking signals: title relevance, location matching, profile completeness."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


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


# Common title-level keywords for semantic grouping
_TITLE_KEYWORDS = {
    "engineer",
    "developer",
    "architect",
    "lead",
    "manager",
    "director",
    "principal",
    "staff",
    "senior",
    "junior",
    "intern",
    "analyst",
    "consultant",
    "specialist",
    "administrator",
    "devops",
    "sre",
    "qa",
    "tester",
    "designer",
    "scientist",
    "researcher",
}

_SENIORITY_ORDER = [
    "intern",
    "junior",
    "mid",
    "senior",
    "staff",
    "principal",
    "lead",
    "architect",
    "manager",
    "director",
    "vp",
    "cto",
    "cio",
]

_TITLE_SECTION = re.compile(
    r"(?:professional\s+)?(?:experience|employment|work\s+history|career)",
    re.IGNORECASE,
)


def extract_jd_title(jd_text: str) -> str:
    """Extract the job title from the first non-empty line of the JD."""
    for line in jd_text.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) < 150:
            return stripped
    return ""


def extract_candidate_titles(resume_text: str) -> list[str]:
    """Extract job titles from a resume's experience section."""
    titles: list[str] = []
    lines = resume_text.split("\n")
    in_experience = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if _TITLE_SECTION.search(stripped):
            in_experience = True
            continue

        if in_experience:
            # Section headers that end the experience section
            if re.match(r"^(?:education|certifications?|skills|projects|awards)", stripped, re.I):
                break

            lower = stripped.lower()
            has_title_word = any(kw in lower for kw in _TITLE_KEYWORDS)
            is_short = len(stripped) < 100
            if has_title_word and is_short and not stripped.endswith("."):
                # Clean up: remove company/date info after dash or pipe
                title = re.split(r"\s*[|–—]\s*", stripped)[0].strip()
                if title and title not in titles:
                    titles.append(title)

    return titles


def _seniority_level(title: str) -> int:
    """Get the seniority level of a title (higher = more senior)."""
    title_lower = title.lower()
    for i, level in enumerate(_SENIORITY_ORDER):
        if level in title_lower:
            return i
    return 3  # default to mid-level


def analyze_title_relevance(jd_text: str, resume_text: str) -> TitleAnalysis:
    """Score how well candidate's job titles match the JD title."""
    from resume_ranker.semantic import semantic_similarity

    jd_title = extract_jd_title(jd_text)
    candidate_titles = extract_candidate_titles(resume_text)

    if not jd_title:
        return TitleAnalysis(score=50.0, summary="Could not extract job title from JD.")

    if not candidate_titles:
        return TitleAnalysis(
            jd_title=jd_title,
            score=30.0,
            summary="Could not extract job titles from resume.",
        )

    # Use semantic similarity for each candidate title vs JD title
    best_score = 0.0
    best_title = ""
    for title in candidate_titles:
        sim = semantic_similarity(jd_title, title)
        if sim > best_score:
            best_score = sim
            best_title = title

    # Seniority alignment bonus/penalty
    jd_seniority = _seniority_level(jd_title)
    candidate_seniority = _seniority_level(best_title)
    seniority_diff = abs(jd_seniority - candidate_seniority)
    seniority_factor = max(0.7, 1.0 - seniority_diff * 0.1)

    score = min(100.0, best_score * 120.0 * seniority_factor)

    if score >= 70:
        level = "Strong"
    elif score >= 40:
        level = "Moderate"
    else:
        level = "Low"

    return TitleAnalysis(
        jd_title=jd_title,
        candidate_titles=candidate_titles,
        score=score,
        summary=f'{level} title alignment. Best match: "{best_title}" → "{jd_title}".',
    )


# Location extraction patterns
_LOCATION_PATTERNS = [
    re.compile(
        r"(?:location|based\s+in|located\s+in|office)\s*[:–\-]?\s*(.+?)(?:\n|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2})\s*(?:\n|$)",
    ),
]

_RELOCATION_PATTERNS = re.compile(
    r"(?:willing|open|ready)\s+(?:to\s+)?relocat|relocation\s+(?:ready|available|open)",
    re.IGNORECASE,
)

_CITY_ALIASES: dict[str, list[str]] = {
    "new york": ["nyc", "new york city", "new york, ny", "manhattan", "brooklyn"],
    "san francisco": ["sf", "san francisco, ca", "bay area"],
    "los angeles": ["la", "los angeles, ca"],
    "chicago": ["chicago, il"],
    "seattle": ["seattle, wa"],
    "austin": ["austin, tx"],
    "boston": ["boston, ma"],
    "denver": ["denver, co"],
}


def _normalize_location(text: str) -> str:
    """Normalize a location string for comparison."""
    return re.sub(r"[^a-z\s]", "", text.lower()).strip()


def _locations_match(loc1: str, loc2: str) -> bool:
    """Check if two locations refer to the same place."""
    n1 = _normalize_location(loc1)
    n2 = _normalize_location(loc2)

    if not n1 or not n2:
        return False

    if n1 in n2 or n2 in n1:
        return True

    for canonical, aliases in _CITY_ALIASES.items():
        all_names = [canonical] + aliases
        all_normalized = [_normalize_location(a) for a in all_names]
        if any(n1 in an or an in n1 for an in all_normalized):
            if any(n2 in an or an in n2 for an in all_normalized):
                return True

    return False


def _clean_location(raw: str) -> str:
    """Strip parenthetical notes and trailing qualifiers from a location."""
    cleaned = re.sub(r"\(.*?\)", "", raw).strip().rstrip(",").strip()
    return cleaned


def analyze_location(jd_text: str, resume_text: str) -> LocationAnalysis:
    """Score location alignment between JD and candidate."""
    resume_lower = resume_text.lower()

    # Extract JD location
    jd_location = ""
    for pattern in _LOCATION_PATTERNS:
        match = pattern.search(jd_text)
        if match:
            jd_location = _clean_location(match.group(1))
            break

    # Check candidate relocation readiness
    relocation_ready = bool(_RELOCATION_PATTERNS.search(resume_text))

    # Extract candidate location
    candidate_location = ""
    for pattern in _LOCATION_PATTERNS:
        match = pattern.search(resume_text)
        if match:
            candidate_location = _clean_location(match.group(1))
            break

    if not jd_location:
        return LocationAnalysis(
            score=70.0,
            summary="No specific location requirement found in JD.",
        )

    # Scoring
    location_match = _locations_match(jd_location, candidate_location)

    # Check if resume mentions the JD city via aliases
    jd_loc_norm = _normalize_location(jd_location)
    resume_mentions_city = jd_loc_norm in resume_lower
    for canonical, aliases in _CITY_ALIASES.items():
        all_names = [canonical] + aliases
        all_normalized = [_normalize_location(a) for a in all_names]
        if any(jd_loc_norm in an or an in jd_loc_norm for an in all_normalized):
            if any(n in resume_lower for n in all_normalized):
                resume_mentions_city = True
                break

    if location_match:
        score = 100.0
        summary = f"Candidate is located in {candidate_location}, matching JD location."
    elif relocation_ready and resume_mentions_city:
        score = 95.0
        summary = f"Candidate is open to relocation to {jd_location}."
    elif relocation_ready:
        score = 75.0
        summary = f"Candidate is open to relocation (JD requires {jd_location})."
    elif resume_mentions_city:
        score = 80.0
        summary = f"Resume mentions {jd_location} but no explicit relocation readiness."
    else:
        score = 30.0
        summary = f"No location match or relocation readiness for {jd_location}."

    return LocationAnalysis(
        jd_location=jd_location,
        candidate_location=candidate_location,
        relocation_ready=relocation_ready,
        score=score,
        summary=summary,
    )


_RESUME_SECTIONS = {
    "summary": re.compile(r"(?:professional\s+)?(?:summary|profile|objective|about)", re.I),
    "experience": re.compile(r"(?:professional\s+)?(?:experience|employment|work\s+history)", re.I),
    "skills": re.compile(r"(?:technical\s+)?skills|competencies|technologies", re.I),
    "education": re.compile(r"education|academic|degree", re.I),
    "certifications": re.compile(r"certifications?|licenses?|credentials", re.I),
    "projects": re.compile(r"projects|portfolio", re.I),
}


def analyze_profile_completeness(resume_text: str) -> ProfileCompletenessAnalysis:
    """Score the completeness of a resume/profile."""
    sections_found: list[str] = []
    sections_missing: list[str] = []

    for section_name, pattern in _RESUME_SECTIONS.items():
        if pattern.search(resume_text):
            sections_found.append(section_name)
        else:
            sections_missing.append(section_name)

    word_count = len(resume_text.split())

    # Section score (60% of completeness)
    core_sections = {"summary", "experience", "skills", "education"}
    core_found = sum(1 for s in sections_found if s in core_sections)
    section_score = (core_found / len(core_sections)) * 60.0

    # Bonus for optional sections
    optional = {"certifications", "projects"}
    optional_found = sum(1 for s in sections_found if s in optional)
    section_score += optional_found * 5.0

    # Word count score (40% of completeness)
    if word_count >= 400:
        word_score = 40.0
    elif word_count >= 200:
        word_score = 30.0
    elif word_count >= 100:
        word_score = 20.0
    else:
        word_score = 10.0

    score = min(100.0, section_score + word_score)

    if score >= 80:
        level = "Complete"
    elif score >= 50:
        level = "Partial"
    else:
        level = "Incomplete"

    return ProfileCompletenessAnalysis(
        sections_found=sections_found,
        sections_missing=sections_missing,
        word_count=word_count,
        score=score,
        summary=(
            f"{level} profile ({len(sections_found)}/{len(_RESUME_SECTIONS)} sections, "
            f"{word_count} words)."
        ),
    )
