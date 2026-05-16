"""Experience extraction from resume text."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from resume_ranker.semantic import semantic_similarity
from resume_ranker.signals import detect_seniority

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MONTH_NAMES: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)"
)

_DATE_RANGE_REGEX = re.compile(
    rf"""
    (?:
        # Start with month name (e.g., Jan 2021)
        (?:
            (?P<sm>{MONTH_PATTERN})
            \s*
            (?P<sy>\d{{4}})
        )
        |
        # Start with numeric month (e.g., 01/2021, 01.2021)
        (?:
            (?P<sm_num>\d{{1,2}})
            [/\.]
            (?P<sy_num>\d{{4}})
        )
        |
        # Start with year only (e.g., 2021)
        (?P<sy_only>\d{{4}})
    )
    \s*
    (?:-|\u2013|\u2014|to)
    \s*
    (?:
        # End with month name (e.g., Jan 2024)
        (?:
            (?P<em>{MONTH_PATTERN})
            \s*
            (?P<ey>\d{{4}})
        )
        |
        # End with numeric month (e.g., 01/2024, 01.2024)
        (?:
            (?P<em_num>\d{{1,2}})
            [/\.]
            (?P<ey_num>\d{{4}})
        )
        |
        # Present/Current/Now
        (?P<end_now>Present|Current|Now)
        |
        # End with year only (e.g., 2024)
        (?P<ey_only>\d{{4}})
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

EMPLOYMENT_WEIGHTS: dict[str, float] = {
    "full_time": 1.0,
    "internship": 0.5,
    "volunteer": 0.3,
    "unknown": 0.7,
}

# Section header detection
_SECTION_HEADER = re.compile(
    r"^(?:#{1,4}\s+)?[A-Z][A-Za-z &/,\-()|:\u2013\u2014]+:?$", re.MULTILINE
)

_EXPERIENCE_HEADERS = re.compile(
    r"(?:professional\s+)?(?:experience|work\s+experience|employment|work\s+history|"
    r"career\s+history|professional\s+(?:background|history)|technical\s+experience|"
    r"industry\s+experience|relevant\s+experience)",
    re.IGNORECASE,
)

_EDUCATION_HEADERS = re.compile(
    r"education|academic\s+background|academic\s+qualifications",
    re.IGNORECASE,
)

_EXCLUDED_HEADERS = re.compile(
    r"(?:certification|certifications|certificate|certificates|"
    r"publication|publications|project|projects|patent|patents|"
    r"award|awards|honors?|honours?)",
    re.IGNORECASE,
)

_VOLUNTEER_HEADER = re.compile(r"volunteer", re.IGNORECASE)

# Other section headers that should create boundaries (prevent content bleed)
_OTHER_HEADERS = re.compile(
    r"(?:skills|technical\s+skills|core\s+competencies|technologies|"
    r"summary|professional\s+summary|profile|objective|"
    r"languages?|interests|activities|references|"
    r"achievements|accomplishments|additional\s+experience)",
    re.IGNORECASE,
)

# Safety: rejection words near a number match
_REJECT_NEAR = re.compile(
    r"\$[\d]|%|"
    r"(?<!\w)(?:users?|dashboards?|projects?|tickets?|employees?|"
    r"clients?|gpa|cgpa|phone|awards?|publications?)(?!\w)",
    re.IGNORECASE,
)

# Company header line detection
_ROLE_TITLE_KEYWORDS = re.compile(
    r"(?:engineer|developer|intern|manager|architect|lead|analyst|"
    r"scientist|consultant|specialist|coordinator|administrator|"
    r"associate|director|head|chief|officer|supervisor|representative|"
    r"trainee|apprentice|freelancer|contractor|designer|technician|"
    r"assistant|researcher|writer|editor|clerk)",
    re.IGNORECASE,
)

_COMPANY_LINE_INDICATORS = re.compile(
    r"(?:onsite|remote|hybrid|in[- ]person|work[- ]from)",
    re.IGNORECASE,
)


def _is_company_header_line(line: str) -> bool:
    """Check if a line is a company header rather than a role title."""
    lower = line.lower()
    if _ROLE_TITLE_KEYWORDS.search(lower):
        return False
    # Lines with location indicators and no role title are likely company headers
    if _COMPANY_LINE_INDICATORS.search(lower):
        return True
    return False


# ---------------------------------------------------------------------------
# Role families for relevance scoring
# ---------------------------------------------------------------------------

ROLE_FAMILIES: dict[str, list[str]] = {
    "data_scientist": [
        "data scientist",
        "ml engineer",
        "machine learning",
        "ml ops",
        "ai engineer",
        "nlp engineer",
        "deep learning",
        "research scientist",
    ],
    "data_analyst": [
        "data analyst",
        "bi analyst",
        "bi developer",
        "business analyst",
        "reporting analyst",
        "analytics",
        "tableau developer",
    ],
    "data_engineer": [
        "data engineer",
        "etl engineer",
        "data pipeline",
        "big data",
        "spark developer",
        "data architect",
        "data warehouse",
    ],
    "backend": [
        "backend",
        "back-end",
        "back end",
        "server-side",
        "api developer",
        "python developer",
        "python engineer",
        "java developer",
        "node.js developer",
        "node developer",
    ],
    "frontend": [
        "frontend",
        "front-end",
        "front end",
        "ui developer",
        "react developer",
        "angular developer",
        "web developer",
        "javascript developer",
        "typescript developer",
    ],
    "fullstack": [
        "full stack",
        "fullstack",
        "full-stack",
    ],
    "devops": [
        "devops",
        "sre",
        "platform engineer",
        "infrastructure engineer",
        "cloud engineer",
        "site reliability",
    ],
    "qa": [
        "qa",
        "test engineer",
        "quality assurance",
        "automation engineer",
        "sdet",
        "software test",
    ],
    "product": [
        "product manager",
        "product owner",
        "program manager",
        "technical product manager",
    ],
    "data_engineering": [
        "data engineer",
        "etl developer",
        "data architect",
        "data pipeline engineer",
        "big data engineer",
    ],
}

_RELATED_FAMILIES: list[tuple[str, str]] = [
    ("data_scientist", "data_analyst"),
    ("data_scientist", "data_engineer"),
    ("data_analyst", "data_engineer"),
    ("backend", "fullstack"),
    ("frontend", "fullstack"),
    ("backend", "devops"),
    ("backend", "data_engineer"),
    ("data_engineer", "data_scientist"),
    ("data_engineer", "data_analyst"),
    ("devops", "backend"),
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _parse_month(month_str: str | None) -> int:
    if not month_str:
        return 1
    return MONTH_NAMES.get(month_str.strip().lower()[:3], 1)


def _month_name(month_num: int) -> str:
    names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return names[month_num] if 1 <= month_num <= 12 else "Jan"


def _months_between(sy: int, sm: int, ey: int, em: int) -> int:
    return (ey - sy) * 12 + (em - sm)


def _is_year_number(num: float) -> bool:
    return 1900 <= num <= 2099


# ---------------------------------------------------------------------------
# Employment type classification
# ---------------------------------------------------------------------------


def _determine_employment_type(title: str, company: str, section_type: str) -> str:
    if section_type == "volunteer":
        return "volunteer"
    context = (title + " " + company).lower()
    if "intern" in context:
        return "internship"
    if "volunteer" in context:
        return "volunteer"
    if "co-op" in context or "coop" in context:
        return "internship"
    return "full_time"


# ---------------------------------------------------------------------------
# Date normalization helpers
# ---------------------------------------------------------------------------


_MONTH_NUM_TO_NAME: dict[str, str] = {
    "1": "Jan",
    "2": "Feb",
    "3": "Mar",
    "4": "Apr",
    "5": "May",
    "6": "Jun",
    "7": "Jul",
    "8": "Aug",
    "9": "Sep",
    "10": "Oct",
    "11": "Nov",
    "12": "Dec",
    "01": "Jan",
    "02": "Feb",
    "03": "Mar",
    "04": "Apr",
    "05": "May",
    "06": "Jun",
    "07": "Jul",
    "08": "Aug",
    "09": "Sep",
}


def _normalize_dates(text: str) -> str:
    text = re.sub(
        rf"({MONTH_PATTERN})\s+(\d{{2}})(?!\d)",
        lambda m: (
            f"{m.group(1)} 20{m.group(2)}"
            if int(m.group(2)) <= 30
            else f"{m.group(1)} 19{m.group(2)}"
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"'(\d{2})\b", r"20\1", text)
    return text


def _parse_end_month(month_str: str | None) -> int:
    if not month_str:
        return 12
    return MONTH_NAMES.get(month_str.strip().lower()[:3], 12)


# ---------------------------------------------------------------------------
# 1. Date-based experience extraction
# ---------------------------------------------------------------------------


def _classify_all_sections(text: str) -> list[tuple[str, int, int]]:
    """Classify all sections in the resume text.

    Only recognizes lines with known section keywords as section boundaries.
    Lines that match _SECTION_HEADER but don't contain recognized keywords
    are treated as regular content within the current section.

    Returns list of (section_type, start_line, end_line).
    """
    lines = text.split("\n")
    sections: list[tuple[str, int, int]] = []
    current_type = "unknown"
    current_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        match = _SECTION_HEADER.match(stripped)
        if not match:
            continue

        lower = stripped.lower()

        if _EXPERIENCE_HEADERS.search(lower) and not _VOLUNTEER_HEADER.search(lower):
            new_type = "professional_experience"
        elif _EDUCATION_HEADERS.search(lower):
            new_type = "education"
        elif _EXCLUDED_HEADERS.search(lower):
            new_type = "excluded"
        elif _VOLUNTEER_HEADER.search(lower):
            new_type = "volunteer"
        elif _OTHER_HEADERS.search(lower):
            new_type = "other"
        else:
            continue  # Not a known section header; keep current section

        if new_type != current_type:
            sections.append((current_type, current_start, i))
            current_type = new_type
            current_start = i

    sections.append((current_type, current_start, len(lines)))
    return sections


def extract_experience_periods(resume_text: str) -> list[dict[str, Any]]:
    """Extract experience date periods from the resume text.

    Only extracts dates from Professional Experience / Work Experience sections.
    Education, Certification, Publication, Project, Patent, and Award sections
    are excluded.
    """
    lines = resume_text.split("\n")
    sections = _classify_all_sections(resume_text)
    periods: list[dict[str, Any]] = []
    seen_periods: set[tuple[int, int, str]] = set()

    for section_type, start, end in sections:
        if section_type in ("professional_experience", "volunteer"):
            section_lines = lines[start:end]
            block = "\n".join(section_lines)
            extracted = _extract_periods_from_block(block, section_type)
            for p in extracted:
                key = (p["start_year"], p["start_month"], p.get("company", ""))
                if key not in seen_periods:
                    seen_periods.add(key)
                    periods.append(p)

    # Fallback: no sections detected, scan the whole text but be conservative
    if not periods:
        fallback = _extract_periods_from_block(resume_text, "unknown")
        if fallback:
            periods = fallback

    return periods


def _extract_periods_from_block(block: str, section_type: str) -> list[dict[str, Any]]:
    periods: list[dict[str, Any]] = []
    lines = block.split("\n")
    today = date.today()

    for i, line in enumerate(lines):
        normalized_line = _normalize_dates(line)
        for match in _DATE_RANGE_REGEX.finditer(normalized_line):
            sm_str = match.group("sm")
            sy_str = match.group("sy")
            sm_num_str = match.group("sm_num")
            sy_num_str = match.group("sy_num")
            sy_only_str = match.group("sy_only")
            em_str = match.group("em")
            ey_str = match.group("ey")
            em_num_str = match.group("em_num")
            ey_num_str = match.group("ey_num")
            ey_only_str = match.group("ey_only")
            end_now_str = match.group("end_now")

            if sy_str is not None:
                start_year = int(sy_str)
                start_month = _parse_month(sm_str)
            elif sy_num_str is not None:
                start_year = int(sy_num_str)
                start_month = int(sm_num_str)
            else:
                start_year = int(sy_only_str)
                start_month = 1

            if end_now_str is not None:
                end_year = today.year
                end_month = today.month
            elif ey_str is not None:
                end_year = int(ey_str)
                end_month = _parse_end_month(em_str)
            elif ey_num_str is not None:
                end_year = int(ey_num_str)
                end_month = int(em_num_str)
            else:
                end_year = int(ey_only_str)
                end_month = 12

            months = _months_between(start_year, start_month, end_year, end_month)
            if months <= 0:
                continue

            date_line = line.strip()
            prefix = _DATE_RANGE_REGEX.sub("", date_line).strip()
            prefix = re.sub(r"\s*[|–—]\s*$", "", prefix).strip()

            if prefix and len(prefix) > 3:
                parts = re.split(r"\s*[|–—]\s*", prefix, maxsplit=1)
                title = parts[0].strip()
                company = parts[1].strip() if len(parts) >= 2 else ""
                if not company:
                    company = lines[i - 1].strip() if i > 0 else ""
            else:
                title = lines[i - 1].strip() if i > 0 else ""
                company = lines[i - 2].strip() if i > 1 else ""

            # Grab follow-on bullet points for context (up to 4 lines)
            bullet_lines: list[str] = []
            for j in range(i + 1, min(i + 5, len(lines))):
                bl = lines[j].strip()
                if not bl:
                    break
                if _DATE_RANGE_REGEX.search(bl):
                    break
                bullet_lines.append(bl)

            context = f"{title} {company} {' '.join(bullet_lines)}"
            description = "\n".join(bullet_lines)

            employment_type = _determine_employment_type(title, company, section_type)

            # Skip company header lines that have redundant date ranges
            if _is_company_header_line(line):
                has_role_date_nearby = False
                for j in range(i + 1, min(i + 4, len(lines))):
                    jl = lines[j].strip()
                    if _ROLE_TITLE_KEYWORDS.search(jl) and _DATE_RANGE_REGEX.search(jl):
                        has_role_date_nearby = True
                        break
                if has_role_date_nearby:
                    continue

            is_current = end_now_str is not None
            start_date_str = f"{start_year:04d}-{start_month:02d}"
            end_date_str = f"{end_year:04d}-{end_month:02d}"

            periods.append(
                {
                    "title": title,
                    "company": company,
                    "context": context,
                    "description": description,
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "start_year": start_year,
                    "start_month": start_month,
                    "end_year": end_year,
                    "end_month": end_month,
                    "months": months,
                    "is_current": is_current,
                    "section": section_type,
                    "employment_type": employment_type,
                }
            )

    return periods


# ---------------------------------------------------------------------------
# 2. Calculate total professional experience (weighted, merged)
# ---------------------------------------------------------------------------


def _month_key(year: int, month: int) -> int:
    return year * 12 + month


def calculate_total_experience_months(periods: list[dict[str, Any]]) -> float:
    """Calculate total weighted months of experience from periods.

    Merges overlapping date ranges. Applies employment type weighting.
    """
    if not periods:
        return 0.0

    # Build month-by-month coverage with weights
    monthly_weights: dict[int, float] = {}

    for p in periods:
        start_mk = _month_key(p["start_year"], p["start_month"])
        end_mk = _month_key(p["end_year"], p["end_month"])
        weight = EMPLOYMENT_WEIGHTS.get(p["employment_type"], 0.7)

        for mk in range(start_mk, end_mk):
            if mk not in monthly_weights or weight > monthly_weights[mk]:
                monthly_weights[mk] = weight

    return sum(monthly_weights.values())


# ---------------------------------------------------------------------------
# 3. Calculate relevant experience
# ---------------------------------------------------------------------------


def _get_role_family(text: str) -> str | None:
    text_lower = text.lower()
    for family, keywords in ROLE_FAMILIES.items():
        for kw in keywords:
            if kw in text_lower:
                return family
    return None


def _role_family_similarity(period_context: str, jd_title: str) -> float:
    jd_family = _get_role_family(jd_title)
    period_family = _get_role_family(period_context)

    if jd_family is None or period_family is None:
        return 0.3

    if jd_family == period_family:
        return 1.0

    if (jd_family, period_family) in _RELATED_FAMILIES or (
        period_family,
        jd_family,
    ) in _RELATED_FAMILIES:
        return 0.5

    return 0.1


def calculate_relevant_experience_months(
    periods: list[dict[str, Any]],
    jd_text: str,
    jd_skills: set[str],
    jd_title: str,
) -> float:
    """Calculate weighted months of relevant experience.

    Relevance is based on role family overlap with the JD title,
    and skill overlap with JD required skills.
    """
    if not periods:
        return 0.0

    relevant_weighted_months = 0.0

    for p in periods:
        context_lower = p["context"].lower()
        months = p["months"]
        emp_weight = EMPLOYMENT_WEIGHTS.get(p["employment_type"], 0.7)
        weighted_months = months * emp_weight

        # Role family similarity
        family_sim = _role_family_similarity(context_lower, jd_title)

        # Skill overlap
        matched_skills = [s for s in jd_skills if s.lower() in context_lower]
        skill_ratio = len(matched_skills) / len(jd_skills) if jd_skills else 0.0

        # Determine relevance weight
        if family_sim >= 0.8 and skill_ratio >= 0.1:
            relevance = 1.0
        elif len(matched_skills) >= 4 or (len(matched_skills) >= 3 and skill_ratio >= 0.5):
            relevance = 1.0  # Strong skill overlap trumps generic titles
        elif family_sim >= 0.5:
            relevance = 0.7
        elif skill_ratio >= 0.2:
            relevance = 0.7
        elif len(matched_skills) >= 1:
            relevance = 0.4
        else:
            relevance = 0.0

        relevant_weighted_months += weighted_months * relevance

    return relevant_weighted_months


# ---------------------------------------------------------------------------
# 4. Improved explicit-years extraction
# ---------------------------------------------------------------------------

_EXPLICIT_PATTERNS = [
    r"(\d+\.?\d*)\+?\s*(?:years?|yrs?)[\s\-]*(?:of\s+)?(?:experience|exp)",
    r"(?:experience|exp)[\s:]*(\d+\.?\d*)\+?\s*(?:years?|yrs?)",
    r"(?:over|more\s+than|approximately|approx|around|about)\s+(\d+\.?\d*)\s*(?:years?|yrs?)",
    r"(\d+\.?\d*)\+?\s*(?:years?|yrs?)\s+(?:in|of|working)",
    r"(\d+\.?\d*)\s+plus\s+(?:years?|yrs?)[\s\-]*(?:of\s+)?(?:experience|exp|programming)",
]

_END_OF_SENTENCE = re.compile(r"[.!?;]")
_NUMBER_WITH_CONTEXT = re.compile(r"(\d+\.?\d*)\+?\s*(?:years?|yrs?)", re.IGNORECASE)


def _is_near_rejection(text: str, match_start: int, match_end: int, window: int = 15) -> bool:
    before = text[max(0, match_start - window) : match_start]
    after = text[match_end : min(len(text), match_end + window)]
    context = before + after
    return bool(_REJECT_NEAR.search(context))


def extract_years_of_experience(text: str) -> float | None:
    """Extract total years of experience from resume text.

    Improved with safety guards:
    - Rejects numbers near $, %, users, dashboards, etc.
    - Rejects numbers that look like calendar years (1900-2099).
    """
    max_years = None
    text_lower = text.lower()

    for pattern in _EXPLICIT_PATTERNS:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            years = float(match.group(1))

            if years > 50:
                continue
            if _is_year_number(years):
                continue
            if _is_near_rejection(text_lower, match.start(), match.end()):
                continue

            if max_years is None or years > max_years:
                max_years = years

    return max_years


# ---------------------------------------------------------------------------
# 5. Combine explicit and date-based experience
# ---------------------------------------------------------------------------


def combine_experience(
    explicit_years: float | None,
    date_based_years: float | None,
) -> dict[str, Any]:
    """Combine explicit and date-based experience estimates.

    Returns dict with final_total_years, source, and warnings.
    """
    result: dict[str, Any] = {
        "explicit_years": explicit_years,
        "date_based_years": date_based_years,
        "final_total_years": None,
        "source": "unknown",
        "warnings": [],
    }

    if explicit_years is not None and date_based_years is not None:
        diff = abs(explicit_years - date_based_years)
        if diff <= 1.5:
            result["final_total_years"] = max(explicit_years, date_based_years)
            result["source"] = "max_of_both"
        elif explicit_years > date_based_years + 2.0:
            result["final_total_years"] = date_based_years
            result["source"] = "date_based_preferred"
            result["warnings"].append(
                f"Explicit years ({explicit_years:.1f}) much higher than "
                f"date-based years ({date_based_years:.1f})"
            )
        else:
            result["final_total_years"] = round(explicit_years, 1)
            result["source"] = "explicit_preferred"
    elif explicit_years is not None:
        result["final_total_years"] = explicit_years
        result["source"] = "explicit_only"
    elif date_based_years is not None:
        result["final_total_years"] = round(date_based_years, 1)
        result["source"] = "date_based_only"
    else:
        result["warnings"].append("No experience data found")

    return result


# ---------------------------------------------------------------------------
# 6. New scoring formula
# ---------------------------------------------------------------------------


def _score_years_against_requirement(
    years: float | None, req_min: float | None, req_max: float | None
) -> float:
    if req_min is None:
        return 70.0
    if years is None:
        return 45.0
    if req_max is not None:
        if req_min <= years <= req_max:
            return 100.0
        if years > req_max:
            over = years - req_max
            return max(60.0, 100.0 - over * 5)
        deficit = req_min - years
        return max(10.0, 100.0 - deficit * 15)
    else:
        if years >= req_min:
            excess = years - req_min
            return min(100.0, 95.0 + excess * 1)
        deficit = req_min - years
        return max(10.0, 80.0 - deficit * 15)


def compute_confidence(
    combine_result: dict[str, Any], has_periods: bool, has_explicit: bool
) -> float:
    """Compute confidence score based on source reliability."""
    source = combine_result.get("source", "unknown")
    warnings = combine_result.get("warnings", [])

    if source == "unknown":
        return 30.0
    if source in ("date_based_only", "date_based_preferred") and has_periods:
        return 90.0
    if source == "max_of_both":
        return 95.0
    if source == "explicit_only" and has_explicit:
        return 75.0
    if source == "explicit_preferred":
        return 80.0
    if warnings:
        return 50.0
    return 70.0


def score_experience_full(
    jd_min: float | None,
    jd_max: float | None,
    final_total_years: float | None,
    relevant_years: float | None,
    combine_result: dict[str, Any],
    periods: list[dict[str, Any]],
    is_domain_specific: bool = False,
) -> dict[str, Any]:
    """Compute experience score using the full formula.

    When is_domain_specific is True (JD asks for specific domain experience),
    primary = relevant_years (70%), secondary = total_years (20%).

    When is_domain_specific is False (general requirement),
    only total_years matters (90%).

    final_score = primary_experience_score * 0.7
                 + secondary_experience_score * 0.2
                 + confidence_score * 0.1
    """
    total_experience_score = _score_years_against_requirement(final_total_years, jd_min, jd_max)
    relevant_experience_score = _score_years_against_requirement(relevant_years, jd_min, jd_max)

    has_periods = bool(periods)
    has_explicit = combine_result.get("explicit_years") is not None
    confidence = compute_confidence(combine_result, has_periods, has_explicit)

    if is_domain_specific and relevant_years is not None:
        primary_score = relevant_experience_score
        secondary_score = total_experience_score
        final_score = primary_score * 0.7 + secondary_score * 0.2 + confidence * 0.1
    else:
        final_score = total_experience_score * 0.9 + confidence * 0.1

    return {
        "jd_requirement": _format_jd_requirement(jd_min, jd_max),
        "candidate_total_years": final_total_years,
        "candidate_relevant_years": relevant_years,
        "total_experience_score": round(total_experience_score, 1),
        "relevant_experience_score": round(relevant_experience_score, 1),
        "confidence_score": round(confidence, 1),
        "final_experience_score": round(final_score, 1),
        "source": combine_result.get("source", "unknown"),
        "warnings": combine_result.get("warnings", []),
        "is_domain_specific": is_domain_specific,
    }


def _format_jd_requirement(min_y: float | None, max_y: float | None) -> str:
    if min_y is None and max_y is None:
        return "Not specified"
    if max_y is not None:
        return f"{min_y:.0f}-{max_y:.0f} years"
    return f"{min_y:.0f}+ years"


# ---------------------------------------------------------------------------
# 7. Convenience: full experience analysis
# ---------------------------------------------------------------------------


def analyze_experience_full(
    resume_text: str,
    jd_text: str,
    jd_skills: set[str] | None = None,
    jd_title: str = "",
    jd_responsibilities: str = "",
) -> dict[str, Any]:
    """Run the full experience analysis pipeline.

    Returns a dict suitable for building ExperienceAnalysis.
    """
    # Extract explicit years
    explicit_years = extract_years_of_experience(resume_text)

    # Extract date-based periods
    periods = extract_experience_periods(resume_text)
    date_based_months = calculate_total_experience_months(periods)
    date_based_years = date_based_months / 12.0 if date_based_months > 0 else None

    # Combine explicit and date-based
    combine_result = combine_experience(explicit_years, date_based_years)

    # Extract JD requirement with domain-specific detection
    jd_min, jd_max, is_domain_specific = extract_required_experience_detailed(jd_text)

    # Calculate relevant experience (legacy)
    if jd_skills is None:
        jd_skills = set()
    relevant_months = calculate_relevant_experience_months(periods, jd_text, jd_skills, jd_title)
    relevant_years = relevant_months / 12.0 if relevant_months > 0 else None

    # Score (legacy)
    score_info = score_experience_full(
        jd_min,
        jd_max,
        combine_result.get("final_total_years"),
        relevant_years,
        combine_result,
        periods,
        is_domain_specific=is_domain_specific,
    )

    # New experience ranking
    exp_ranking = compute_experience_ranking(
        periods,
        jd_title,
        jd_skills,
        jd_responsibilities,
        jd_min,
        jd_max,
    )

    # Separate matched and ignored periods
    matched_periods = [p for p in periods if True]  # all periods are matched for now

    score_info["matched_periods"] = matched_periods
    score_info["ignored_periods"] = []
    score_info["explicit_years"] = explicit_years
    score_info["date_based_years"] = date_based_years

    # Add new experience ranking fields
    score_info["experience_score"] = exp_ranking.get("experience_score")
    score_info["total_experience_fit"] = exp_ranking.get("total_experience_fit")
    score_info["relevant_experience_fit"] = exp_ranking.get("relevant_experience_fit")
    score_info["recency_score"] = exp_ranking.get("recency_score")
    score_info["seniority_fit"] = exp_ranking.get("seniority_fit")
    score_info["total_years_new"] = exp_ranking.get("total_years")
    score_info["relevant_years_new"] = exp_ranking.get("relevant_years")
    score_info["experience_ranking_periods"] = exp_ranking.get("periods", [])

    return score_info


# ---------------------------------------------------------------------------
# Word-number parsing
# ---------------------------------------------------------------------------

_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_NUMBER_PATTERN = r"(?:\d+\.?\d*|" + "|".join(_WORD_NUMBERS.keys()) + r")"


def _parse_number(num_str: str) -> float | None:
    if not num_str:
        return None
    low = num_str.strip().lower()
    if low in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[low])
    try:
        return float(low)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Domain-specific requirement detection
# ---------------------------------------------------------------------------

_DOMAIN_PATTERN = re.compile(
    r"(?:in|of|with|for)\s+"
    r"(?:\w+\s+){0,4}"
    r"(?:development|engineering|analysis?|science|architecture|"
    r"design|testing|administration|management|programming|"
    r"coding|building|creating|implementing|maintaining|"
    r"supporting|operating|automation|integration|"
    r"frontend|backend|full.?stack|data|cloud|devops|qa|"
    r"mobile|web|api|software|application)",
    re.IGNORECASE,
)

_GENERIC_ENDINGS = re.compile(
    r"(?:required|preferred|desired|necessary|"
    r"or\s+equivalent|is\s+a\s+must|highly\s+desired)\s*$",
    re.IGNORECASE,
)


def _is_domain_specific_match(text: str, match_end: int) -> bool:
    """Check if the requirement mentions a specific domain after the year."""
    after = text[match_end : match_end + 120]
    # Strip generic endings
    after_cleaned = _GENERIC_ENDINGS.sub("", after).strip()
    if not after_cleaned or len(after_cleaned.split()) <= 2:
        return False  # Just "experience required" or similar boilerplate
    if _DOMAIN_PATTERN.search(after_cleaned):
        return True
    return False


# ---------------------------------------------------------------------------
# 7. Public employment type detection
# ---------------------------------------------------------------------------


def detect_employment_type(title: str, section_text: str) -> str:
    return _determine_employment_type(title, "", section_text)


# ---------------------------------------------------------------------------
# 8. Recency weight for relevance calculation
# ---------------------------------------------------------------------------


def _get_recency_weight_for_relevance(end_year: int, end_month: int, is_current: bool) -> float:
    if is_current:
        return 1.0
    today = date.today()
    months_ago = (today.year - end_year) * 12 + (today.month - end_month)
    years_ago = months_ago / 12.0
    if years_ago <= 2:
        return 0.85
    elif years_ago <= 5:
        return 0.65
    else:
        return 0.45


# ---------------------------------------------------------------------------
# 9. Period relevance scoring
# ---------------------------------------------------------------------------


def _calculate_period_relevance(
    period: dict[str, Any],
    jd_title: str,
    jd_required_skills: set[str],
    jd_responsibilities: str,
) -> float:
    title_sim = semantic_similarity(period.get("title", ""), jd_title)
    role_title_score = min(1.0, title_sim * 1.5)

    period_family = _get_role_family(period.get("title", ""))
    jd_role_family = _get_role_family(jd_title)
    family_bonus = 0.0
    if jd_role_family and period_family:
        if jd_role_family == period_family:
            family_bonus = 0.3
        elif (jd_role_family, period_family) in _RELATED_FAMILIES or (
            period_family,
            jd_role_family,
        ) in _RELATED_FAMILIES:
            family_bonus = 0.15
    role_title_score = min(1.0, role_title_score + family_bonus)

    description = period.get("description", "")
    context_lower = (description + " " + period.get("title", "")).lower()
    if jd_required_skills:
        matched = sum(1 for s in jd_required_skills if s.lower() in context_lower)
        skill_score = min(1.0, (matched / len(jd_required_skills)) * 1.5)
    else:
        skill_score = 0.3

    if description.strip() and jd_responsibilities.strip():
        resp_sim = semantic_similarity(description, jd_responsibilities)
        responsibility_score = min(1.0, resp_sim * 1.5)
    else:
        responsibility_score = 0.3

    period_relevance = role_title_score * 0.35 + skill_score * 0.40 + responsibility_score * 0.25
    return min(1.0, period_relevance)


# ---------------------------------------------------------------------------
# 10. Total Experience Fit scoring
# ---------------------------------------------------------------------------


def calculate_total_experience_fit(
    periods: list[dict[str, Any]],
    jd_required_years: float | None,
    jd_max: float | None = None,
) -> tuple[float, float]:
    if not periods:
        return (0.0, 0.0) if jd_required_years else (50.0, 0.0)

    total_weighted_months = calculate_total_experience_months(periods)
    total_years = total_weighted_months / 12.0

    if jd_required_years is None:
        return (70.0, total_years)

    if jd_max is not None and total_years >= jd_required_years and total_years <= jd_max:
        return (100.0, total_years)

    if jd_max is not None and total_years > jd_max:
        excess = total_years - jd_max
        score = max(75.0, 100.0 - excess * 5)
    elif total_years >= jd_required_years:
        extra = total_years - jd_required_years
        score = max(75.0, 100.0 - extra * 3)
    else:
        deficit = jd_required_years - total_years
        score = max(10.0, 100.0 - deficit * 18)

    return (score, total_years)


# ---------------------------------------------------------------------------
# 11. Relevant Experience Fit scoring
# ---------------------------------------------------------------------------


def calculate_relevant_experience_fit(
    periods: list[dict[str, Any]],
    jd_title: str,
    jd_required_skills: set[str],
    jd_responsibilities: str,
    jd_required_years: float | None,
) -> dict[str, Any]:
    if not periods:
        return {
            "relevant_experience_fit": 0.0 if jd_required_years else 50.0,
            "relevant_years": 0.0,
            "period_details": [],
        }

    relevant_months = 0.0
    period_details: list[dict[str, Any]] = []

    for p in periods:
        is_current = p.get("is_current", False)
        recency_weight = _get_recency_weight_for_relevance(
            p.get("end_year", date.today().year),
            p.get("end_month", date.today().month),
            is_current,
        )
        period_relevance = _calculate_period_relevance(
            p, jd_title, jd_required_skills, jd_responsibilities
        )
        emp_weight = EMPLOYMENT_WEIGHTS.get(p.get("employment_type", "full_time"), 1.0)
        counted = p.get("months", 0) * emp_weight * period_relevance * recency_weight
        relevant_months += counted

        period_details.append(
            {
                "title": p.get("title", ""),
                "company": p.get("company", ""),
                "start_date": p.get("start_date", ""),
                "end_date": p.get("end_date", ""),
                "months": p.get("months", 0),
                "employment_type": p.get("employment_type", "full_time"),
                "relevance_score": round(period_relevance, 4),
                "recency_weight": round(recency_weight, 4),
                "counted_relevant_months": round(counted, 2),
            }
        )

    relevant_years = relevant_months / 12.0

    if jd_required_years is None:
        score = 70.0
    elif relevant_years >= jd_required_years:
        extra = relevant_years - jd_required_years
        score = max(80.0, 100.0 - extra * 2)
    else:
        deficit = jd_required_years - relevant_years
        score = max(10.0, 100.0 - deficit * 22)

    return {
        "relevant_experience_fit": round(score, 1),
        "relevant_years": round(relevant_years, 2),
        "period_details": period_details,
    }


# ---------------------------------------------------------------------------
# 12. Recency Score
# ---------------------------------------------------------------------------


def calculate_recency_score(
    periods: list[dict[str, Any]],
    jd_title: str,
    jd_required_skills: set[str],
) -> float:
    if not periods:
        return 30.0

    today = date.today()
    jd_role_family = _get_role_family(jd_title)
    best_score = 30.0

    for p in periods:
        is_current = p.get("is_current", False)
        period_family = _get_role_family(p.get("title", ""))
        context_lower = (p.get("description", "") + " " + p.get("title", "")).lower()

        is_relevant = False
        if jd_role_family and period_family:
            if jd_role_family == period_family:
                is_relevant = True
            elif (jd_role_family, period_family) in _RELATED_FAMILIES or (
                period_family,
                jd_role_family,
            ) in _RELATED_FAMILIES:
                is_relevant = True

        if not is_relevant and jd_required_skills:
            matched = sum(1 for s in jd_required_skills if s.lower() in context_lower)
            if matched >= 2:
                is_relevant = True

        if not is_relevant:
            continue

        end_year = p.get("end_year", today.year)
        end_month = p.get("end_month", today.month)

        if is_current:
            score = 95.0
        else:
            years_ago = (today.year - end_year) + (today.month - end_month) / 12.0
            if years_ago <= 2:
                score = 85.0
            elif years_ago <= 5:
                score = 68.0
            else:
                score = 50.0

        if score > best_score:
            best_score = score

    return best_score


# ---------------------------------------------------------------------------
# 13. Seniority Fit
# ---------------------------------------------------------------------------


def calculate_seniority_fit(
    periods: list[dict[str, Any]],
    jd_title: str,
) -> float:
    if not jd_title or not periods:
        return 70.0

    jd_seniority = detect_seniority(jd_title)
    candidate_seniority = 3

    for p in periods:
        if p.get("is_current", False):
            candidate_seniority = detect_seniority(p.get("title", ""))
            break

    if candidate_seniority == 3 and periods:
        candidate_seniority = detect_seniority(periods[0].get("title", ""))

    diff = candidate_seniority - jd_seniority

    if diff == 0:
        return 100.0
    elif diff < 0:
        abs_diff = abs(diff)
        if abs_diff == 1:
            return 85.0
        elif abs_diff == 2:
            return 65.0
        else:
            return 45.0
    else:
        if diff <= 2:
            return 95.0
        elif diff <= 4:
            return 90.0
        else:
            return 85.0


# ---------------------------------------------------------------------------
# 14. Final Experience Score computation
# ---------------------------------------------------------------------------


def compute_experience_ranking(
    periods: list[dict[str, Any]],
    jd_title: str,
    jd_required_skills: set[str],
    jd_responsibilities: str,
    jd_required_years: float | None,
    jd_max: float | None = None,
) -> dict[str, Any]:
    total_fit_score, total_years = calculate_total_experience_fit(
        periods, jd_required_years, jd_max
    )
    relevant_result = calculate_relevant_experience_fit(
        periods, jd_title, jd_required_skills, jd_responsibilities, jd_required_years
    )
    relevant_fit_score = relevant_result.get("relevant_experience_fit", 0)
    relevant_years = relevant_result.get("relevant_years", 0.0)
    period_details = relevant_result.get("period_details", [])

    recency = calculate_recency_score(periods, jd_title, jd_required_skills)
    seniority = calculate_seniority_fit(periods, jd_title)

    experience_score = total_fit_score * 0.50 + relevant_fit_score * 0.50

    warnings: list[str] = []
    if len(period_details) != len(periods):
        warnings.append("Some periods excluded from relevance calculation")

    return {
        "experience_score": round(experience_score, 1),
        "total_years": round(total_years, 2),
        "relevant_years": round(relevant_years, 2),
        "total_experience_fit": round(total_fit_score, 1),
        "relevant_experience_fit": round(relevant_fit_score, 1),
        "recency_score": round(recency, 1),
        "seniority_fit": round(seniority, 1),
        "periods": period_details,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Original API (kept for backward compatibility)
# ---------------------------------------------------------------------------

# extract_years_of_experience is defined above (improved version)


def extract_required_experience(jd_text: str) -> tuple[float | None, float | None]:
    """Extract minimum and maximum required experience from a job description.

    Returns (min_years, max_years). Either can be None if not found.
    """
    min_y, max_y, _ = extract_required_experience_detailed(jd_text)
    return min_y, max_y


def extract_required_experience_detailed(
    jd_text: str,
) -> tuple[float | None, float | None, bool]:
    """Extract required experience with domain-specific detection.

    Returns (min_years, max_years, is_domain_specific).
    is_domain_specific is True when the requirement mentions a specific
    field (e.g. "3 years of data analysis experience").
    """
    text_lower = jd_text.lower()

    def _check_domain(match_end: int) -> bool:
        return _is_domain_specific_match(text_lower, match_end)

    # Pattern: "5-10 years of data analysis experience"
    range_pattern = rf"({_NUMBER_PATTERN})\s*[-–to]+\s*({_NUMBER_PATTERN})\s*(?:years?|yrs?)"
    range_match = re.search(range_pattern, text_lower)
    if range_match:
        min_y = _parse_number(range_match.group(1))
        max_y = _parse_number(range_match.group(2))
        is_domain = _check_domain(range_match.end())
        return min_y, max_y, is_domain

    # Pattern: "minimum 5 years", "at least 5 years", "5+ years", "two+ years"
    min_patterns = [
        rf"(?:minimum|min|at\s+least|atleast)\s+({_NUMBER_PATTERN})\s*(?:years?|yrs?)",
        rf"({_NUMBER_PATTERN})\+\s*(?:years?|yrs?)",
        rf"({_NUMBER_PATTERN})\s+plus\s+(?:years?|yrs?)",
        rf"({_NUMBER_PATTERN})\s*(?:years?|yrs?)\s+(?=of|in)",
    ]
    found_min = None
    found_domain = False
    found_end = 0
    for pattern in min_patterns:
        for match in re.finditer(pattern, text_lower):
            val = _parse_number(match.group(1))
            if val is None:
                continue
            if found_min is None or val < found_min:
                found_min = val
                found_end = match.end()
                found_domain = _check_domain(match.end())

    if found_min is not None:
        return found_min, None, found_domain

    # Fallback: general extraction
    general = extract_years_of_experience(jd_text)
    if general is not None:
        return general, None, False

    return None, None, False
