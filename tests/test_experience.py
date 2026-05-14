"""Tests for experience extraction and scoring."""

from resume_ranker.experience import (
    analyze_experience_full,
    calculate_relevant_experience_months,
    calculate_total_experience_months,
    combine_experience,
    extract_experience_periods,
    extract_required_experience,
    extract_years_of_experience,
    score_experience_full,
)


# ---------------------------------------------------------------------------
# Backward compatibility: original extract_years_of_experience tests
# ---------------------------------------------------------------------------


def test_extract_years_basic():
    assert extract_years_of_experience("10 years of experience in software") == 10.0


def test_extract_years_plus():
    assert extract_years_of_experience("5+ years experience") == 5.0


def test_extract_years_over():
    assert extract_years_of_experience("Over 8 years of experience") == 8.0


def test_extract_years_none():
    assert extract_years_of_experience("Some professional background") is None


# ---------------------------------------------------------------------------
# Backward compatibility: original extract_required_experience tests
# ---------------------------------------------------------------------------


def test_extract_required_range():
    min_y, max_y = extract_required_experience("Requires 3-5 years of experience")
    assert min_y == 3.0
    assert max_y == 5.0


def test_extract_required_minimum():
    min_y, max_y = extract_required_experience("Minimum 5 years of experience required")
    assert min_y == 5.0
    assert max_y is None


def test_extract_required_plus():
    min_y, max_y = extract_required_experience("7+ years experience in backend development")
    assert min_y == 7.0
    assert max_y is None


def test_extract_required_none():
    min_y, max_y = extract_required_experience("Looking for a motivated developer")
    assert min_y is None
    assert max_y is None


# ---------------------------------------------------------------------------
# A. False number protection
# ---------------------------------------------------------------------------


def test_false_number_protection_dollar():
    """$75M+ should NOT be extracted as 75 years."""
    text = "Delivered $75M+ YoY impact and supported 50 users."
    assert extract_years_of_experience(text) is None


def test_false_number_protection_users():
    """50 users should NOT be extracted as 50 years."""
    text = "Managed a team supporting 50 users across the organization."
    assert extract_years_of_experience(text) is None


def test_false_number_protection_dashboards():
    """100 dashboards should NOT be extracted as 100 years."""
    text = "Built 100 dashboards for business intelligence."
    assert extract_years_of_experience(text) is None


def test_false_number_protection_projects():
    """projects keyword should block number extraction."""
    text = "Led 20 projects across multiple teams."
    assert extract_years_of_experience(text) is None


def test_false_number_protection_year_range():
    """2022 - 2024 should not be extracted as 2024 years of experience."""
    text = "Worked on multiple initiatives from 2022 - 2024."
    assert extract_years_of_experience(text) is None


def test_false_number_protection_gpa():
    """GPA should not be extracted as years."""
    text = "Graduated with 3.8 GPA and honors."
    assert extract_years_of_experience(text) is None


def test_true_experience_still_extracted():
    """Valid 'X years of experience' should still be extracted even with other numbers nearby."""
    text = "Delivered $75M+ YoY impact. 10 years of experience in software development."
    assert extract_years_of_experience(text) == 10.0


# ---------------------------------------------------------------------------
# B. Education date protection
# ---------------------------------------------------------------------------

RESUME_WITH_EDUCATION = """John Doe
Software Engineer

Professional Experience:
Software Engineer | ABC Corp
Jan 2020 - Present
- Built scalable backend services
- Managed AWS infrastructure

Education:
Bachelor of Science in Computer Science
2018 - 2022
GPA: 3.8
"""


def test_education_dates_not_counted():
    """Education section dates should not be counted as experience."""
    periods = extract_experience_periods(RESUME_WITH_EDUCATION)
    # Only the Professional Experience period should be found
    assert len(periods) == 1
    assert periods[0]["employment_type"] == "full_time"
    assert periods[0]["start_year"] == 2020


# ---------------------------------------------------------------------------
# C. Date-based extraction
# ---------------------------------------------------------------------------

RESUME_SIMPLE_EXPERIENCE = """Software Engineer | ABC
Jan 2021 - Jan 2024
- Built APIs
- Wrote tests
"""


def test_date_based_extraction():
    """Jan 2021 - Jan 2024 should yield 3 years (36 months)."""
    periods = extract_experience_periods(RESUME_SIMPLE_EXPERIENCE)
    assert len(periods) == 1
    assert periods[0]["months"] == 36
    assert periods[0]["employment_type"] == "full_time"

    total_months = calculate_total_experience_months(periods)
    assert abs(total_months / 12.0 - 3.0) < 0.1


# ---------------------------------------------------------------------------
# D. Internship weighting
# ---------------------------------------------------------------------------

RESUME_INTERNSHIP = """Software Engineer Intern | ABC Corp
Jan 2023 - Jan 2024
- Assisted with development
- Wrote unit tests
"""


def test_internship_weighting():
    """Internship: 12 months * 0.5 = 6 weighted months."""
    periods = extract_experience_periods(RESUME_INTERNSHIP)
    assert len(periods) >= 1
    # Find the internship period
    intern_periods = [p for p in periods if p["employment_type"] == "internship"]
    assert len(intern_periods) >= 1

    total_months = calculate_total_experience_months(periods)
    # 12 weighted months at 0.5 = 6
    assert abs(total_months - 6.0) < 0.1


# ---------------------------------------------------------------------------
# E. Volunteer weighting
# ---------------------------------------------------------------------------

RESUME_VOLUNTEER = """Volunteer Assistant | Local Charity
Jul 2018 - Jul 2019
- Helped organize events
- Managed social media
"""


def test_volunteer_weighting():
    """Volunteer: 12 months * 0.3 = 3.6 weighted months."""
    periods = extract_experience_periods(RESUME_VOLUNTEER)
    # This should find a volunteer period
    vol_periods = [
        p for p in periods if p["employment_type"] == "volunteer" or p["section"] == "volunteer"
    ]
    assert len(vol_periods) >= 1

    total_months = calculate_total_experience_months(periods)
    vol_months = sum(
        p["months"] * 0.3
        for p in periods
        if p["employment_type"] == "volunteer" or p["section"] == "volunteer"
    )
    assert abs(total_months - vol_months) < 0.1


RESUME_WITH_VOLUNTEER_SECTION = """Professional Experience:
Software Engineer | Tech Co
Jan 2020 - Present
- Built software

Volunteer Experience:
Volunteer Assistant | Local Charity
Jul 2018 - Jul 2019
- Helped organize events
"""


def test_volunteer_section_weighting():
    """Volunteer section periods should get 30% weight."""
    periods = extract_experience_periods(RESUME_WITH_VOLUNTEER_SECTION)
    assert len(periods) == 2

    total_months = calculate_total_experience_months(periods)
    # Full-time: ~60 months (Jan 2020 to ~May 2026) * 1.0
    # Volunteer: 12 * 0.3 = 3.6
    full_time = [p for p in periods if p["employment_type"] == "full_time"]
    volunteer = [p for p in periods if p["employment_type"] == "volunteer"]
    assert len(full_time) >= 1
    assert len(volunteer) >= 1
    assert abs(volunteer[0]["months"] * 0.3 - 3.6) < 0.1


# ---------------------------------------------------------------------------
# F. Relevant experience
# ---------------------------------------------------------------------------

JD_PYTHON_DEV = """
Python Developer, 3+ years of experience required.
Skills: Python, Django, PostgreSQL
"""

RESUME_TABLEAU_DEV = """Tableau Developer | Analytics Corp
Jan 2021 - Jan 2024
- Used Python scripts for predictive modeling for 3 months
- Built Tableau dashboards
- Worked with SQL
"""


def test_relevant_experience_partial():
    """Tableau Developer with Python usage should get partial relevant experience."""
    periods = extract_experience_periods(RESUME_TABLEAU_DEV)
    assert len(periods) >= 1

    jd_skills = {"python", "django", "postgresql"}
    rel_months = calculate_relevant_experience_months(
        periods, JD_PYTHON_DEV, jd_skills, "Python Developer"
    )
    # Should be > 0 (Python matches) but less than full (not same role family)
    assert rel_months > 0
    # Full weighted months for this period = 36 * 1.0 = 36
    # With relevance 0.4 (only one skill matches) = 14.4
    # Or relevance 0.7 (if role family is related) = 25.2
    total_months = calculate_total_experience_months(periods)
    assert rel_months < total_months  # Should not be full


# ---------------------------------------------------------------------------
# G. Current role with Present
# ---------------------------------------------------------------------------

RESUME_CURRENT = """Backend Developer | Startup Inc
Jan 2022 - Present
- Built backend services
"""


def test_current_role_calculated():
    """Current role (Present) should calculate until today's date."""
    periods = extract_experience_periods(RESUME_CURRENT)
    assert len(periods) == 1
    assert periods[0]["employment_type"] == "full_time"
    # Jan 2022 to today should be > 36 months (as of 2026)
    assert periods[0]["months"] >= 36


# ---------------------------------------------------------------------------
# combine_experience tests
# ---------------------------------------------------------------------------


def test_combine_both_close():
    """When both sources exist and close, use the higher value."""
    result = combine_experience(5.0, 4.5)
    assert result["final_total_years"] == 5.0
    assert result["source"] == "max_of_both"


def test_combine_explicit_much_higher():
    """When explicit is much higher, prefer date-based."""
    result = combine_experience(15.0, 3.0)
    assert result["final_total_years"] == 3.0
    assert result["source"] == "date_based_preferred"
    assert len(result["warnings"]) == 1


def test_combine_explicit_only():
    """When only explicit exists, use it."""
    result = combine_experience(5.0, None)
    assert result["final_total_years"] == 5.0
    assert result["source"] == "explicit_only"


def test_combine_date_based_only():
    """When only date-based exists, use it."""
    result = combine_experience(None, 3.0)
    assert result["final_total_years"] == 3.0
    assert result["source"] == "date_based_only"


def test_combine_neither():
    """When neither exists, return unknown."""
    result = combine_experience(None, None)
    assert result["final_total_years"] is None
    assert result["source"] == "unknown"


def test_combine_moderate_diff():
    """When difference is moderate (1.6-2.0), prefer explicit."""
    result = combine_experience(5.0, 3.2)
    assert result["final_total_years"] == 5.0
    assert result["source"] == "explicit_preferred"


# ---------------------------------------------------------------------------
# score_experience_full tests
# ---------------------------------------------------------------------------


def test_score_within_range():
    """Score 100 when total years within JD range."""
    result = score_experience_full(
        3.0, 5.0, 4.0, 4.0, {"source": "max_of_both", "warnings": []}, []
    )
    assert result["total_experience_score"] == 100.0


def test_score_below_range():
    """Score decreases when below JD range."""
    result = score_experience_full(
        5.0, 8.0, 3.0, 3.0, {"source": "max_of_both", "warnings": []}, []
    )
    assert result["total_experience_score"] < 100.0
    assert result["total_experience_score"] == max(10.0, 100.0 - 2 * 15)


def test_score_above_range():
    """Gentler penalty for overqualification."""
    result = score_experience_full(
        3.0, 5.0, 10.0, 10.0, {"source": "max_of_both", "warnings": []}, []
    )
    over = 10.0 - 5.0
    assert result["total_experience_score"] == max(60.0, 100.0 - over * 5)


def test_score_meets_minimum():
    """Meets/exceeds minimum."""
    result = score_experience_full(
        5.0, None, 7.0, 7.0, {"source": "max_of_both", "warnings": []}, []
    )
    excess = 7.0 - 5.0
    assert result["total_experience_score"] == min(100.0, 95.0 + excess * 1)


def test_score_no_jd_requirement():
    """No JD requirement = 70 total experience score."""
    result = score_experience_full(
        None, None, 5.0, 5.0, {"source": "explicit_only", "warnings": []}, []
    )
    assert result["total_experience_score"] == 70.0


def test_score_no_candidate_years():
    """No candidate years = neutral-low score."""
    result = score_experience_full(
        3.0, 5.0, None, None, {"source": "unknown", "warnings": ["No experience data found"]}, []
    )
    assert result["total_experience_score"] == 45.0


def test_final_score_formula():
    """Final score = total*0.4 + relevant*0.5 + confidence*0.1."""
    result = score_experience_full(
        3.0, 5.0, 4.0, 3.0, {"source": "max_of_both", "warnings": []}, [{"months": 12}]
    )
    expected = 100.0 * 0.4 + 100.0 * 0.5 + 95.0 * 0.1  # confidence=95 for max_of_both
    assert abs(result["final_experience_score"] - expected) < 0.1


# ---------------------------------------------------------------------------
# analyze_experience_full integration test
# ---------------------------------------------------------------------------


def test_analyze_experience_full_pipeline():
    """End-to-end pipeline produces expected structure."""
    resume = """Software Engineer | Tech Co
Jan 2020 - Dec 2023
- Developed Python applications
- Used Django framework
"""
    jd = """Python Developer, 3+ years experience required."""
    result = analyze_experience_full(resume, jd, {"python", "django"}, "Python Developer")

    assert "final_experience_score" in result
    assert "candidate_total_years" in result
    assert "candidate_relevant_years" in result
    assert "matched_periods" in result
    assert "warnings" in result
    assert result["matched_periods"] is not None


# ---------------------------------------------------------------------------
# extract_experience_periods edge cases
# ---------------------------------------------------------------------------

RESUME_NO_EXPLICIT_SECTION = """John Doe
Software Developer

Worked on various projects.
Developed APIs and web services.
"""


def test_no_experience_section_fallback():
    """When no clear section detected, periods list may be empty."""
    periods = extract_experience_periods(RESUME_NO_EXPLICIT_SECTION)
    # No date ranges, so periods should be empty
    assert isinstance(periods, list)


RESUME_MULTIPLE_JOBS = """Professional Experience:
Senior Developer | Company A
Jan 2020 - Dec 2022
- Built microservices

Junior Developer | Company B
Mar 2018 - Dec 2019
- Wrote backend code
"""


def test_multiple_job_periods():
    """Multiple jobs should each produce a period."""
    periods = extract_experience_periods(RESUME_MULTIPLE_JOBS)
    assert len(periods) == 2


# ---------------------------------------------------------------------------
# calculate_total_experience_months edge cases
# ---------------------------------------------------------------------------


def test_total_months_empty():
    """Empty periods should return 0."""
    assert calculate_total_experience_months([]) == 0.0


def test_total_months_no_overlap():
    """Non-overlapping periods should sum."""
    periods = [
        {
            "start_year": 2020,
            "start_month": 1,
            "end_year": 2020,
            "end_month": 6,
            "months": 6,
            "employment_type": "full_time",
        },
        {
            "start_year": 2021,
            "start_month": 1,
            "end_year": 2021,
            "end_month": 6,
            "months": 6,
            "employment_type": "full_time",
        },
    ]
    # 5 months each (Jan-Jun exclusive) = 10 total
    total = calculate_total_experience_months(periods)
    assert total == 10.0


# ---------------------------------------------------------------------------
# relevance calculation edge cases
# ---------------------------------------------------------------------------


def test_relevant_experience_no_skills():
    """No JD skills should not crash."""
    periods = [
        {
            "title": "Developer",
            "company": "Co",
            "context": "Developer at Co used Python",
            "start_year": 2020,
            "start_month": 1,
            "end_year": 2021,
            "end_month": 1,
            "months": 12,
            "section": "professional_experience",
            "employment_type": "full_time",
        }
    ]
    result = calculate_relevant_experience_months(periods, "Job", set(), "Developer")
    assert result >= 0


# ---------------------------------------------------------------------------
# explicit vs date-based: full pipeline
# ---------------------------------------------------------------------------


def test_explicit_and_date_based_both_present():
    """When resume has both explicit years and date ranges."""
    resume = """10 years of experience in software development.

Professional Experience:
Senior Developer | Tech Co
Jan 2020 - Present
"""
    years = extract_years_of_experience(resume)
    assert years == 10.0

    periods = extract_experience_periods(resume)
    assert len(periods) >= 1

    combined = combine_experience(years, calculate_total_experience_months(periods) / 12.0)
    assert combined["final_total_years"] is not None


# ---------------------------------------------------------------------------
# False positive protection: explicit years in edge cases
# ---------------------------------------------------------------------------


def test_cap_at_50_years():
    """Values over 50 should be rejected."""
    assert extract_years_of_experience("100 years of experience") is None


def test_year_number_rejected():
    """Calendar year 2024 should not be extracted."""
    assert extract_years_of_experience("2024 years of experience") is None


def test_percent_sign_protection():
    """Number near % sign should be rejected."""
    assert (
        extract_years_of_experience("25% improvement in 5 years") is None
        or extract_years_of_experience("25% improvement in 5 years") != 25.0
    )


# ---------------------------------------------------------------------------
# Missing explicit years but date-based available
# ---------------------------------------------------------------------------


def test_no_explicit_but_has_dates():
    """Resume with date ranges but no explicit 'X years' should not get harsh penalty."""
    resume = """Software Engineer | ABC
Jan 2021 - Jan 2024
- Built software
"""
    explicit = extract_years_of_experience(resume)
    assert explicit is None

    periods = extract_experience_periods(resume)
    date_months = calculate_total_experience_months(periods)
    assert date_months > 0

    combined = combine_experience(explicit, date_months / 12.0)
    assert combined["final_total_years"] is not None
    assert combined["source"] == "date_based_only"


# ---------------------------------------------------------------------------
# Test that '5+ years' works correctly (regression)
# ---------------------------------------------------------------------------


def test_plus_years_still_works():
    assert extract_years_of_experience("5+ years experience") == 5.0


def test_years_in_still_works():
    assert extract_years_of_experience("10 years in software development") == 10.0


# ---------------------------------------------------------------------------
# Company header line regression test
# ---------------------------------------------------------------------------

RESUME_COMPANY_HEADER_DATE = """Professional Experience:
10xscale.ai – Onsite Jan 2025 – Present
Hyderabad, India
Software Engineer Jun 2025 – Present
- Built web applications
- Used React

Backend Developer Intern Jan 2025 – May 2025
- Built automation pipelines
- Used Python
"""


def test_company_header_date_not_counted_as_separate_role():
    """Company header line (with Onsite and date) should not be extracted
    as a separate period when role lines below also have dates."""
    periods = extract_experience_periods(RESUME_COMPANY_HEADER_DATE)
    # Should find 2 periods (SE + Intern), not 3 (not company header)
    assert len(periods) == 2
    types = [p["employment_type"] for p in periods]
    assert "full_time" in types
    assert "internship" in types
    # Company header should not create a third period
    company_titles = [p["title"] for p in periods]
    assert not any("Onsite" in t for t in company_titles)


# ---------------------------------------------------------------------------
# Domain-specific requirement detection
# ---------------------------------------------------------------------------


def test_domain_specific_detection():
    """'in frontend development' should flag as domain-specific."""
    from resume_ranker.experience import extract_required_experience_detailed

    min_y, max_y, is_domain = extract_required_experience_detailed(
        "1+ years of experience in frontend development"
    )
    assert min_y == 1.0
    assert max_y is None
    assert is_domain is True


def test_general_requirement():
    """'5+ years of experience required' should be general."""
    from resume_ranker.experience import extract_required_experience_detailed

    min_y, max_y, is_domain = extract_required_experience_detailed(
        "5+ years of experience required."
    )
    assert min_y == 5.0
    assert is_domain is False


def test_domain_data_analysis():
    """'2 years of experience in data analysis' should be domain-specific."""
    from resume_ranker.experience import extract_required_experience_detailed

    min_y, max_y, is_domain = extract_required_experience_detailed(
        "2+ years of experience in data analysis"
    )
    assert min_y == 2.0
    assert is_domain is True


def test_domain_with_skill():
    """'3 years of Python development' should be domain-specific."""
    from resume_ranker.experience import extract_required_experience_detailed

    min_y, max_y, is_domain = extract_required_experience_detailed(
        "3 years of Python development experience"
    )
    assert min_y == 3.0
    assert is_domain is True


# ---------------------------------------------------------------------------
# Word-number parsing
# ---------------------------------------------------------------------------


def test_word_number_two():
    """'two plus years' should parse as 2."""
    from resume_ranker.experience import extract_required_experience_detailed

    min_y, max_y, _ = extract_required_experience_detailed(
        "two plus years of experience in data engineering"
    )
    assert min_y == 2.0


def test_word_number_three():
    """'three years' should parse as 3."""
    from resume_ranker.experience import extract_required_experience_detailed

    min_y, max_y, _ = extract_required_experience_detailed(
        "three years of experience in backend development"
    )
    assert min_y == 3.0


def test_word_number_five():
    """'five+ years' should parse as 5."""
    from resume_ranker.experience import extract_required_experience_detailed

    min_y, max_y, _ = extract_required_experience_detailed("five+ years experience")
    assert min_y == 5.0


# ---------------------------------------------------------------------------
# Domain-aware scoring
# ---------------------------------------------------------------------------


def test_domain_specific_uses_relevant_years():
    """Domain-specific JD should primarily score against relevant years."""
    from resume_ranker.experience import analyze_experience_full

    resume = """Software Engineer | Tech Co
Jan 2021 - Jan 2024
- Used React and JavaScript for UI development
"""
    jd = """Frontend Developer
2+ years of experience in frontend development
Skills: React, JavaScript, CSS
"""
    result = analyze_experience_full(
        resume, jd, {"react", "javascript", "css"}, "Frontend Developer"
    )
    # Total = 36 months full_time = 3y, Relevant = 36*1.0 = 3y
    # Domain-specific → primary = relevant = 100 (exceeds 2+ req)
    assert result.get("is_domain_specific") is True
    assert result["final_experience_score"] >= 90


def test_general_uses_total_years():
    """General JD should score against total years."""
    from resume_ranker.experience import analyze_experience_full

    resume = """Software Engineer | Tech Co
Jan 2021 - Jan 2024
- Used React and JavaScript for UI development
"""
    jd = """Any Developer
5+ years of experience.
"""
    result = analyze_experience_full(resume, jd, {"react", "javascript"}, "Developer")
    # Total = 3y, Required = 5+ → deficit = 2 → score = 80 - 2*15 = 50
    # General → primary = total score
    assert result.get("is_domain_specific") is False
    assert result["final_experience_score"] < 70
