"""Experience extraction from resume text."""

import re


def extract_years_of_experience(text: str) -> float | None:
    """Extract total years of experience from resume text.

    Looks for common patterns like:
    - "10 years of experience"
    - "5+ years experience"
    - "over 8 years"
    - "experience: 3 years"
    """
    patterns = [
        r"(\d+\.?\d*)\+?\s*(?:years?|yrs?)[\s\-]*(?:of\s+)?(?:experience|exp)",
        r"(?:experience|exp)[\s:]*(\d+\.?\d*)\+?\s*(?:years?|yrs?)",
        r"(?:over|more\s+than|approximately|approx|around|about)\s+(\d+\.?\d*)\s*(?:years?|yrs?)",
        r"(\d+\.?\d*)\+?\s*(?:years?|yrs?)\s+(?:in|of|working)",
    ]

    max_years = None
    text_lower = text.lower()

    for pattern in patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            years = float(match.group(1))
            if years <= 50:  # sanity check
                if max_years is None or years > max_years:
                    max_years = years

    return max_years


def extract_required_experience(jd_text: str) -> tuple[float | None, float | None]:
    """Extract minimum and maximum required experience from a job description.

    Returns (min_years, max_years). Either can be None if not found.
    """
    text_lower = jd_text.lower()

    # Pattern: "5-10 years"
    range_pattern = r"(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)\s*(?:years?|yrs?)"
    range_match = re.search(range_pattern, text_lower)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))

    # Pattern: "minimum 5 years" or "at least 5 years" or "5+ years"
    # When multiple "X+ years" appear (e.g. "8+ years" and "5+ years"),
    # use the lowest as the minimum requirement.
    min_patterns = [
        r"(?:minimum|min|at\s+least|atleast)\s+(\d+\.?\d*)\s*(?:years?|yrs?)",
        r"(\d+\.?\d*)\+\s*(?:years?|yrs?)",
    ]
    found_min = None
    for pattern in min_patterns:
        for match in re.finditer(pattern, text_lower):
            val = float(match.group(1))
            if found_min is None or val < found_min:
                found_min = val
    if found_min is not None:
        return found_min, None

    # Fallback: just find any year mention
    general = extract_years_of_experience(jd_text)
    if general is not None:
        return general, None

    return None, None
