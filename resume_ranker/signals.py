"""LinkedIn-style ranking signals: title relevance, location matching, profile completeness."""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field


@dataclass
class TitleAnalysis:
    """Analysis of job title relevance with detailed debug info."""

    jd_title: str = ""
    candidate_titles: list[dict] = field(default_factory=list)
    score: float = 0.0
    summary: str = ""
    debug: dict = field(default_factory=dict)


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


# ==============================================================================
# Title Relevance — Role Family, Seniority, and Extraction Constants
# ==============================================================================

# Expanded seniority ladder (higher index = more senior)
_SENIORITY_ORDER = [
    "intern",
    "junior",
    "associate",
    "mid",
    "senior",
    "lead",
    "staff",
    "principal",
    "architect",
    "manager",
    "director",
]

# Role-family keyword mapping: each family has a set of keywords to detect.
# Order matters — more specific families are checked first so that
# e.g. "data engineer" matches "data" before "generic_engineering".
_ROLE_FAMILY_KEYWORDS: list[tuple[str, set[str]]] = [
    (
        "ai_ml",
        {
            "machine learning engineer",
            "ml engineer",
            "ai engineer",
            "deep learning",
            "nlp engineer",
            "computer vision",
            "llm engineer",
            "data scientist",
        },
    ),
    (
        "data",
        {
            "data engineer",
            "data pipeline",
            "etl engineer",
            "spark engineer",
            "big data engineer",
            "data infrastructure",
        },
    ),
    (
        "devops",
        {
            "devops engineer",
            "sre",
            "site reliability engineer",
            "infrastructure engineer",
            "platform engineer",
            "ci/cd engineer",
            "release engineer",
        },
    ),
    (
        "qa",
        {
            "qa engineer",
            "test engineer",
            "quality assurance",
            "sdet",
            "automation engineer",
            "software test engineer",
        },
    ),
    (
        "mobile",
        {
            "mobile engineer",
            "ios engineer",
            "ios developer",
            "android engineer",
            "android developer",
            "react native",
            "flutter engineer",
            "mobile developer",
        },
    ),
    (
        "analytics_bi",
        {
            "analytics engineer",
            "business intelligence",
            "bi engineer",
            "bi developer",
            "tableau",
            "power bi",
            "looker",
            "data analyst",
        },
    ),
    (
        "security",
        {
            "security engineer",
            "cyber security",
            "infosec",
            "penetration tester",
            "soc analyst",
            "cloud security",
        },
    ),
    (
        "frontend",
        {
            "frontend engineer",
            "frontend developer",
            "react engineer",
            "angular engineer",
            "vue engineer",
            "ui engineer",
            "ux engineer",
            "web developer",
            "javascript engineer",
        },
    ),
    (
        "backend",
        {
            "backend engineer",
            "backend developer",
            "server engineer",
            "java engineer",
            "java developer",
            "spring engineer",
            "python engineer",
            "python developer",
            "django engineer",
            "fastapi engineer",
            "node engineer",
            "node developer",
            "rust engineer",
            "golang engineer",
            "microservices engineer",
            "api engineer",
            "backend architect",
        },
    ),
    (
        "fullstack",
        {
            "fullstack engineer",
            "full stack engineer",
            "full-stack engineer",
            "fullstack developer",
        },
    ),
    (
        "cloud",
        {
            "cloud engineer",
            "aws engineer",
            "azure engineer",
            "gcp engineer",
            "cloud architect",
        },
    ),
    (
        "product",
        {
            "product manager",
            "product owner",
            "program manager",
            "technical program manager",
            "tpm",
        },
    ),
    (
        "generic_engineering",
        {
            "software engineer",
            "software developer",
            "swe",
            "sde",
            "engineer",
            "developer",
            "programmer",
        },
    ),
]

# Role family relatedness matrix.
# Families not listed here or not in the related set are considered "unrelated".
# unrelated families get a penalty factor of 0.60
# related families get 0.85, same family gets 1.0
_ROLE_FAMILY_RELATED: dict[str, set[str]] = {
    "frontend": {"fullstack", "mobile", "generic_engineering"},
    "backend": {"fullstack", "cloud", "devops", "generic_engineering"},
    "fullstack": {"frontend", "backend", "generic_engineering"},
    "data": {"analytics_bi", "ai_ml", "generic_engineering"},
    "devops": {"cloud", "backend", "generic_engineering"},
    "qa": {"generic_engineering"},
    "mobile": {"frontend", "generic_engineering"},
    "ai_ml": {"data", "analytics_bi", "generic_engineering"},
    "analytics_bi": {"data", "ai_ml", "generic_engineering"},
    "security": {"cloud", "devops", "generic_engineering"},
    "product": {"generic_engineering"},
    "cloud": {"devops", "backend", "generic_engineering"},
    "generic_engineering": set(),
}

# JD title label prefixes
_JD_TITLE_LABELS = [
    re.compile(r"job\s+title\s*:", re.IGNORECASE),
    re.compile(r"job\s+description\s*[-–—:|]\s*", re.IGNORECASE),
    re.compile(r"role\s*:", re.IGNORECASE),
    re.compile(r"position\s*:", re.IGNORECASE),
    re.compile(r"opening\s*:", re.IGNORECASE),
    re.compile(r"designation\s*:", re.IGNORECASE),
]

# Role keywords for JD/extraction phrase detection
_ROLE_KEYWORDS: set[str] = {
    "engineer",
    "developer",
    "analyst",
    "architect",
    "manager",
    "consultant",
    "scientist",
    "administrator",
    "designer",
    "lead",
    "specialist",
    "president",
    "vp",
    "director",
    "head",
    "chief",
    "sde",
    "swe",
}

# Experience section header pattern
_TITLE_SECTION = re.compile(
    r"(?:professional\s+)?(?:experience|employment|work\s+history|career)",
    re.IGNORECASE,
)

# Section boundary headers (end of experience section)
_SECTION_BOUNDARY = re.compile(
    r"^(?:education|certifications?|skills|projects|awards|summary|"
    r"publications|languages|interests|references)",
    re.IGNORECASE,
)

# Company-like line heuristic
_COMPANY_LINE = re.compile(
    r"(?:inc|llc|ltd|corp|corporation|company|technologies|tech|"
    r"systems|software|consulting|group|solutions|laboratories|lab)",
    re.IGNORECASE,
)

# Date pattern for extracting years near title lines
_DATE_PATTERN = re.compile(
    r"(20\d{2})\s*[-–]\s*(20\d{2}|present|current|now)",
    re.IGNORECASE,
)
_DATE_PATTERN_OPEN = re.compile(r"(20\d{2})\s*[-–]\s*(?:$|[^-–])")
_DATE_PATTERN_SINGLE = re.compile(r"(20\d{2})")


# ==============================================================================
# 1. JD Title Extraction
# ==============================================================================


def _clean_jd_title(raw_title: str) -> str:
    """Clean a JD title by stripping markdown markers and boilerplate prefixes."""
    title = raw_title.strip()

    # Strip leading markdown heading markers
    title = re.sub(r"^#+\s*", "", title)

    # Strip known boilerplate prefixes (case-insensitive)
    boilerplate_prefixes = [
        r"^job\s+description\s*[-–—:|]\s*",
        r"^role\s+description\s*[-–—:|]\s*",
        r"^job\s+posting\s*[-–—:|]\s*",
        r"^we\s+(?:are\s+)?hiring\s+(?:a|an|for\s+a)\s+",
        r"^we\s+(?:are\s+)?looking\s+for\s+(?:a|an)\s+",
        r"^hiring\s+(?:for|a|an)\s+",
        r"^looking\s+for\s+(?:a|an)\s+",
        r"^open(?:ing)?\s*[-–—:]\s*",
        r"^position\s*[-–—:]\s*",
    ]
    for prefix in boilerplate_prefixes:
        title = re.sub(prefix, "", title, flags=re.IGNORECASE)

    # Remove decorative leading/trailing separators and bullet markers
    title = title.strip().lstrip("-–—:|# *•").rstrip("-–—:|# *•").strip()

    # Normalize whitespace
    title = re.sub(r"\s+", " ", title).strip().rstrip(".:;,")

    return title


def extract_jd_title(jd_text: str) -> str:
    """Extract the job title from a job description.

    Strategy:
      1. Prefer lines prefixed with "Job Title:", "Role:", "Position:", "Opening:".
      2. Fall back to the first short line (≤10 words) containing a role keyword.
      3. Ignore company/about/benefits boilerplate lines.
      All extracted titles are cleaned of markdown markers and boilerplate phrases.
    """
    lines = jd_text.split("\n")

    # Phase 1: look for labelled title lines
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        for pattern in _JD_TITLE_LABELS:
            match = pattern.search(stripped)
            if match:
                title = stripped[match.end() :].strip().lstrip("-–: ").strip()
                title = _clean_jd_title(title)
                if title and len(title.split()) <= 15:
                    return title

    # Phase 2: look for first short line with role keywords
    company_indicators = {
        "about",
        "company",
        "benefits",
        "why join",
        "who we are",
        "our mission",
        "about us",
        "perks",
    }
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        lower = stripped.lower()

        # Skip very long lines (titles are typically ≤10 words)
        words = stripped.split()
        if len(words) > 10:
            continue

        # Skip bullet-pointed lines (responsibilities, not titles)
        if stripped.startswith(("*", "-", "•", "–")):
            continue

        # Skip company/about/benefits lines
        if any(indicator in lower for indicator in company_indicators):
            continue

        # Skip lines that look like company names
        if _COMPANY_LINE.search(stripped) and len(words) <= 3:
            continue

        # Skip lines that are numbered lists
        if re.match(r"^\d+[.)]\s+", stripped):
            continue

        # Check for role keywords
        if any(kw in lower for kw in _ROLE_KEYWORDS):
            title = _clean_jd_title(stripped)
            if title and len(title.split()) <= 10:
                # Re-check keywords on cleaned title to avoid false matches
                if any(kw in title.lower() for kw in _ROLE_KEYWORDS):
                    return title

    # Phase 3: extract title-like phrases from long lines using a regex pattern.
    # Looks for contiguous sequences of capitalized words followed by a role keyword,
    # which is the typical structure of job titles embedded in sentences.
    # e.g. "We are seeking a passionate Frontend-Focused Full Stack Developer"
    #                                                      ^--- extracted ---^
    role_kw_pattern = "|".join(
        re.escape(kw) for kw in sorted(_ROLE_KEYWORDS, key=len, reverse=True)
    )
    # Match sequences of 1-8 capitalized words (optionally hyphenated) ending with a role keyword
    # Also handles titles like "Full Stack Developer" or "Senior Java Engineer"
    title_extract_re = re.compile(
        r"(?:[A-Z][a-z]+(?:[-–][A-Z][a-z]+)?\s+){1,8}"
        r"(?i:(?:" + role_kw_pattern + r"))",
    )

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        # Skip bullet-pointed and numbered lines (not titles)
        if stripped.startswith(("*", "-", "•", "–")):
            continue
        if re.match(r"^\d+[.)]\s+", stripped):
            continue
        # Check if any keyword appears at all
        if not any(kw in lower for kw in _ROLE_KEYWORDS):
            continue

        # Extract all title-like phrases from this line
        for match in title_extract_re.finditer(stripped):
            candidate = match.group(0).strip()
            candidate_lower = candidate.lower()
            # Filter out boilerplate
            if any(indicator in candidate_lower for indicator in company_indicators):
                continue
            # Clean the candidate
            title = _clean_jd_title(candidate)
            if not title or len(title.split()) < 2 or len(title.split()) > 10:
                continue
            if not any(kw in title.lower() for kw in _ROLE_KEYWORDS):
                continue
            # Prefer titles with seniority markers (word-boundary match)
            seniority_hint = any(
                re.search(r"\b" + re.escape(s) + r"\b", title.lower()) for s in _SENIORITY_ORDER
            )
            # Prefer properly titled case
            first_word_capped = title.split()[0][0].isupper() if title.split() else False
            if seniority_hint or first_word_capped:
                return title

    # Last resort: among all short non-bullet lines with keywords, return the one
    # that looks most like a title (shortest, non-sentence)
    candidates: list[tuple[int, str]] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(("*", "-", "•", "–")):
            continue
        lower = stripped.lower()
        if not any(kw in lower for kw in _ROLE_KEYWORDS):
            continue
        words = stripped.split()
        if 2 <= len(words) <= 10:
            title = _clean_jd_title(stripped)
            if title and 2 <= len(title.split()) <= 10:
                if any(kw in title.lower() for kw in _ROLE_KEYWORDS):
                    candidates.append((len(words), title))

    if candidates:
        candidates.sort(key=lambda x: x[0])  # prefer shorter
        return candidates[0][1]

    return ""


# ==============================================================================
# 2. Resume Candidate Title Extraction
# ==============================================================================

# Lines to skip in header/top-of-resume when looking for a title
_HEADER_SKIP_PATTERNS = [
    re.compile(r"\w+@\w+"),  # email
    re.compile(r"\d{3}[-.]?\d{3}[-.]?\d{4}"),  # phone
    re.compile(r"(?:linkedin|github|twitter)", re.IGNORECASE),
    re.compile(r"^[|/\\\s]*$"),  # separators
]


def _looks_like_title_line(stripped: str) -> bool:
    """Check if a single line looks like a job title (not a section header,
    bullet point, contact detail, company name, or sentence)."""
    lower = stripped.lower()

    # Skip bullets, empty, too long, period-ended
    if stripped.startswith(("-", "•", "*", "–")):
        return False
    if not stripped:
        return False
    if len(stripped) >= 100:
        return False
    if stripped.endswith("."):
        return False

    # Skip section headers
    if _SECTION_BOUNDARY.match(stripped):
        return False
    if _TITLE_SECTION.search(stripped):
        return False

    # Skip contact / link lines
    for pat in _HEADER_SKIP_PATTERNS:
        if pat.search(stripped):
            return False

    # Skip lines that are all numbers / symbols
    if re.match(r"^[\d\s|/\\+\-()]+$", stripped):
        return False

    # Must contain at least one role keyword
    if not any(kw in lower for kw in _ROLE_KEYWORDS):
        return False

    # Ideally 2-6 words (short title)
    words = stripped.split()
    if not (2 <= len(words) <= 8):
        return False

    return True


def _extract_titles_from_header(resume_text: str) -> list[dict]:
    """Fallback: scan the top ~12 non-empty lines of a resume for a job title.

    Used when the Experience section yields no results — many resumes
    list the current/prominent title in a header block near the top.
    """
    lines = resume_text.split("\n")
    seen_titles: set[str] = set()
    results: list[dict] = []
    non_empty_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        non_empty_count += 1
        if non_empty_count > 12:
            break

        if not _looks_like_title_line(stripped):
            continue

        # Clean: remove trailing company/date after common separators
        title = re.split(r"\s*[|–—]\s*", stripped)[0].strip()
        title = re.sub(r"\s+", " ", title).strip().rstrip(".:;,")

        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())

        results.append(
            {
                "title": title,
                "start_year": None,
                "end_year": None,
                "is_current": True,
            }
        )

    return results


def _extract_years_from_line(line: str) -> dict | None:
    """Extract year range information from a single line of text."""
    line = line.strip()
    # Pattern: 2020-2024, 2020 - 2024, 2020 - Present
    m = _DATE_PATTERN.search(line)
    if m:
        start = int(m.group(1))
        end_str = m.group(2)
        is_current = end_str.lower() in ("present", "current", "now")
        return {
            "start_year": start,
            "end_year": None if is_current else int(end_str),
            "is_current": is_current,
        }
    # Pattern: 2020- (open ended)
    m = _DATE_PATTERN_OPEN.search(line)
    if m:
        return {
            "start_year": int(m.group(1)),
            "end_year": None,
            "is_current": True,
        }
    # Pattern: single year
    m = _DATE_PATTERN_SINGLE.search(line)
    if m:
        y = int(m.group(1))
        return {
            "start_year": y,
            "end_year": y,
            "is_current": False,
        }
    return None


def _extract_nearby_years(lines: list[str], idx: int) -> dict:
    """Look for date information in the surrounding lines of a title."""
    start = max(0, idx - 1)
    end = min(len(lines), idx + 2)
    for i in range(start, end):
        result = _extract_years_from_line(lines[i])
        if result:
            return result
    return {"start_year": None, "end_year": None, "is_current": False}


def extract_candidate_titles(resume_text: str) -> list[dict]:
    """Extract job titles with date context from a resume's experience section.

    Returns a list of dicts, each with:
        title (str)        — cleaned job title
        start_year (int|None)
        end_year (int|None)
        is_current (bool)
    """
    results: list[dict] = []
    lines = resume_text.split("\n")
    in_experience = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        if _TITLE_SECTION.search(stripped):
            in_experience = True
            continue

        if not in_experience:
            continue

        # Stop at next major section
        if _SECTION_BOUNDARY.match(stripped):
            break

        # Skip bullet points (responsibilities, not titles)
        if stripped.startswith(("-", "•", "*", "–")):
            continue

        lower = stripped.lower()

        # Must contain at least one role keyword as a whole word
        has_title_word = any(
            re.search(r"\b" + re.escape(kw) + r"\b", lower) for kw in _ROLE_KEYWORDS
        )
        if not has_title_word:
            continue

        # Title lines should be reasonably short
        if len(stripped) >= 100:
            continue

        # Skip lines ending with period (likely a sentence, not title)
        if stripped.endswith("."):
            continue

        # Clean up: remove company/date suffix after dash, pipe, or em-dash
        title = re.split(r"\s*[|–—]\s*", stripped)[0].strip()
        if not title:
            continue

        # Normalize whitespace
        title = re.sub(r"\s+", " ", title).strip().rstrip(".:;,")

        # Strip trailing date patterns from title (e.g. "Software Engineer Jun 2025" → "Software Engineer")
        title = re.sub(
            r"\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)?\s*\d{4}\s*$",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()
        if not title:
            continue

        word_count = len(title.split())
        if word_count < 2 or word_count > 10:
            continue

        # Deduplicate by title text
        if any(r["title"] == title for r in results):
            continue

        year_info = _extract_nearby_years(lines, idx)
        results.append(
            {
                "title": title,
                "start_year": year_info["start_year"],
                "end_year": year_info["end_year"],
                "is_current": year_info["is_current"],
            }
        )

    if not results:
        results = _extract_titles_from_header(resume_text)

    return results


# ==============================================================================
# 3. Role Family Classification
# ==============================================================================


def detect_role_family(title: str) -> str:
    """Detect the role family for a given job title.

    Returns one of: frontend, backend, fullstack, data, devops, qa,
    mobile, ai_ml, analytics_bi, security, product, cloud,
    generic_engineering.
    """
    title_lower = title.lower()
    for family, keywords in _ROLE_FAMILY_KEYWORDS:
        for kw in keywords:
            if kw in title_lower:
                return family
    return "generic_engineering"


def _get_role_family_factor(jd_family: str, candidate_family: str) -> float:
    """Compute the role-family alignment factor between JD and candidate.

    Returns:
        1.00  — same family
        0.85  — related family
        0.60  — unrelated family
    """
    if jd_family == candidate_family:
        return 1.0
    related = _ROLE_FAMILY_RELATED.get(jd_family, set())
    if candidate_family in related:
        return 0.85
    return 0.60


# ==============================================================================
# 4. Seniority Detection & Factor
# ==============================================================================


def detect_seniority(title: str) -> int:
    """Detect the seniority level index for a given job title.

    Returns an index into _SENIORITY_ORDER (higher = more senior).
    Defaults to "mid" (index 3) if no keyword found.
    """
    title_lower = title.lower()
    for i, level in enumerate(_SENIORITY_ORDER):
        if level in title_lower:
            return i
    return 3  # default to mid-level


def _get_seniority_factor(jd_seniority: int, candidate_seniority: int) -> float:
    """Compute seniority alignment factor.

    Penalizes mismatches by 0.1 per level of difference.
    Minimum factor is 0.65.
    """
    diff = abs(jd_seniority - candidate_seniority)
    return max(0.65, 1.0 - diff * 0.1)


# ==============================================================================
# 5. Recency Weighting
# ==============================================================================


def _get_recency_weight(recency_info: dict) -> float:
    """Weight a candidate title based on how recent it is.

    Current role                        → 1.0
    Ended 1-2 years ago                 → 0.9
    Ended 3-5 years ago                 → 0.75
    Ended 5+ years ago                  → 0.6
    """
    is_current = recency_info.get("is_current", False)
    end_year = recency_info.get("end_year")

    if is_current or end_year is None:
        return 1.0

    current_year = datetime.datetime.now().year
    years_ago = current_year - end_year

    if years_ago <= 0:
        return 1.0
    elif years_ago <= 2:
        return 0.9
    elif years_ago <= 5:
        return 0.75
    else:
        return 0.6


# ==============================================================================
# 6. Main Title Relevance Scoring
# ==============================================================================


def analyze_title_relevance(jd_text: str, resume_text: str) -> TitleAnalysis:
    """Score how well candidate's job titles match the JD title.

    For each candidate title, computes:
        final = semantic_similarity × seniority_factor × role_family_factor × recency_factor

    Takes the best candidate title, scales to 0-100, and returns
    a detailed TitleAnalysis with debug info.
    """
    from resume_ranker.semantic import semantic_similarity

    jd_title = extract_jd_title(jd_text)
    candidate_titles = extract_candidate_titles(resume_text)

    if not jd_title:
        return TitleAnalysis(
            score=50.0,
            summary="Could not extract job title from JD.",
        )

    if not candidate_titles:
        return TitleAnalysis(
            jd_title=jd_title,
            score=30.0,
            summary="Could not extract job titles from resume.",
        )

    jd_family = detect_role_family(jd_title)
    jd_seniority = detect_seniority(jd_title)

    best_raw_score = 0.0
    best_entry = candidate_titles[0]
    best_semantic = 0.0
    best_seniority_factor = 1.0
    best_family_factor = 1.0
    best_recency_factor = 1.0

    for entry in candidate_titles:
        title = entry["title"]

        # Semantic similarity between JD title and candidate title
        sim = semantic_similarity(jd_title, title)

        # Seniority alignment
        cand_seniority = detect_seniority(title)
        seniority_factor = _get_seniority_factor(jd_seniority, cand_seniority)

        # Role family alignment
        cand_family = detect_role_family(title)
        family_factor = _get_role_family_factor(jd_family, cand_family)

        # Recency weighting
        recency_factor = _get_recency_weight(entry)

        # Combined raw score (0-1 range)
        raw = sim * seniority_factor * family_factor * recency_factor

        if raw > best_raw_score:
            best_raw_score = raw
            best_entry = entry
            best_semantic = sim
            best_seniority_factor = seniority_factor
            best_family_factor = family_factor
            best_recency_factor = recency_factor

    # Scale to 0-100
    score = min(100.0, best_raw_score * 100.0)

    # Qualitative label
    if score >= 70:
        level = "Strong"
    elif score >= 40:
        level = "Moderate"
    else:
        level = "Low"

    # Debug explainability
    best_candidate_family = detect_role_family(best_entry["title"])
    debug = {
        "jd_title": jd_title,
        "candidate_title": best_entry["title"],
        "semantic_similarity": round(best_semantic, 4),
        "seniority_factor": round(best_seniority_factor, 4),
        "role_family_factor": round(best_family_factor, 4),
        "recency_factor": round(best_recency_factor, 4),
        "final_score": round(score, 2),
        "role_family_match": best_candidate_family,
        "jd_role_family": jd_family,
        "jd_seniority_index": jd_seniority,
        "candidate_seniority_index": detect_seniority(best_entry["title"]),
    }

    return TitleAnalysis(
        jd_title=jd_title,
        candidate_titles=candidate_titles,
        score=score,
        summary=(f'{level} title alignment. Best match: "{best_entry["title"]}" → "{jd_title}".'),
        debug=debug,
    )


# ==============================================================================
# Location extraction patterns
# ==============================================================================

_LOCATION_PATTERNS = [
    re.compile(
        r"(?:location|based\s+in|located\s+in|office)\s*[:–\-]?\s*(.+?)(?:\n|$)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|\n)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2})\s*(?:\n|$)"),
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
