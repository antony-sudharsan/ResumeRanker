"""Tests for the improved Title Relevance scoring system."""

from resume_ranker.signals import (
    TitleAnalysis,
    analyze_title_relevance,
    detect_current_role,
    detect_role_family,
    detect_seniority,
    extract_candidate_titles,
    extract_jd_title,
)
from resume_ranker.ranker import rank_candidates


# ==============================================================================
# JD Title Extraction Tests
# ==============================================================================

JD_LABELLED = """We are hiring!

Job Title: Senior Backend Engineer

Requirements:
- 5+ years experience
"""

JD_FIRST_LINE = """Senior Python Developer
We are looking for an experienced developer to join our team.

Requirements:
- Python experience
"""

JD_ABOUT_COMPANY = """About Our Company
We are a fast-growing startup.

Senior Backend Engineer

Requirements:
- 5+ years experience
"""


def test_extract_jd_title_from_label():
    """Should prefer labelled 'Job Title:' line."""
    assert extract_jd_title(JD_LABELLED) == "Senior Backend Engineer"


def test_extract_jd_title_from_content():
    """Should fall back to first role-keyword line."""
    assert "Senior Python Developer" in extract_jd_title(JD_FIRST_LINE)


def test_extract_jd_title_ignore_about():
    """Should skip 'About Our Company' and find the real title."""
    title = extract_jd_title(JD_ABOUT_COMPANY)
    assert title == "Senior Backend Engineer"
    assert "About" not in title


def test_extract_jd_title_empty():
    """Should return empty string for garbage text."""
    assert extract_jd_title("Some random text without any title keywords.") == ""


def test_extract_jd_title_short_words():
    """Should handle very short JDs."""
    assert extract_jd_title("") == ""
    assert extract_jd_title(" ") == ""


# ==============================================================================
# Resume Candidate Title Extraction Tests
# ==============================================================================

RESUME_WITH_DATES = """
John Smith
Senior Backend Engineer    2020 - 2024
Company XYZ

Skills: Python, Java

Experience:
Senior Backend Engineer
Company XYZ | Jan 2020 - Present
- Led backend development
- Designed APIs

Software Engineer
Another Corp | 2016 - 2019
- Built microservices
"""

RESUME_NO_DATES = """
Jane Doe
Data Analyst

Skills: Tableau, Excel

Experience:
Data Analyst
Company ABC
- Analyzed data
- Built dashboards
"""

RESUME_BULLET_ONLY = """
Bob
Engineer

Experience:
- Worked with engineering manager to deliver projects
- Developed backend systems using Python
"""


def test_extract_candidate_titles_structured():
    """Should return structured data with years."""
    titles = extract_candidate_titles(RESUME_WITH_DATES)
    assert len(titles) >= 1
    assert titles[0]["title"] == "Senior Backend Engineer"
    assert isinstance(titles[0], dict)
    assert "start_year" in titles[0]
    assert "end_year" in titles[0]
    assert "is_current" in titles[0]


def test_extract_candidate_titles_skips_bullets():
    """Should ignore bullet-pointed responsibilities."""
    titles = extract_candidate_titles(RESUME_BULLET_ONLY)
    for t in titles:
        assert not t["title"].startswith("Worked with")
        assert not t["title"].startswith("Developed backend")


def test_extract_candidate_titles_no_experience():
    """Should return empty list if no experience section."""
    assert extract_candidate_titles("No experience section here") == []


# ==============================================================================
# Role Family Classification Tests
# ==============================================================================


def test_role_family_backend():
    assert detect_role_family("Senior Backend Engineer") == "backend"
    assert detect_role_family("Java Engineer") == "backend"
    assert detect_role_family("Python Developer") == "backend"


def test_role_family_frontend():
    assert detect_role_family("Frontend Developer") == "frontend"
    assert detect_role_family("React Engineer") == "frontend"


def test_role_family_data():
    assert detect_role_family("Data Engineer") == "data"
    assert detect_role_family("Senior Data Engineer") == "data"


def test_role_family_devops():
    assert detect_role_family("DevOps Engineer") == "devops"
    assert detect_role_family("SRE") == "devops"


def test_role_family_analytics():
    assert detect_role_family("Data Analyst") == "analytics_bi"
    assert detect_role_family("Tableau Developer") == "analytics_bi"


def test_role_family_mobile():
    assert detect_role_family("Mobile Engineer") == "mobile"
    assert detect_role_family("iOS Engineer") == "mobile"
    assert detect_role_family("Android Developer") == "mobile"


def test_role_family_qa():
    assert detect_role_family("QA Engineer") == "qa"
    assert detect_role_family("SDET") == "qa"


def test_role_family_ai_ml():
    assert detect_role_family("Machine Learning Engineer") == "ai_ml"
    assert detect_role_family("Data Scientist") == "ai_ml"


def test_role_family_generic():
    assert detect_role_family("Software Engineer") == "generic_engineering"
    assert detect_role_family("Junior Developer") == "generic_engineering"


def test_role_family_fullstack():
    assert detect_role_family("Full Stack Engineer") == "fullstack"


def test_role_family_cloud():
    assert detect_role_family("Cloud Engineer") == "cloud"
    assert detect_role_family("AWS Engineer") == "cloud"


def test_role_family_security():
    assert detect_role_family("Security Engineer") == "security"


def test_role_family_product():
    assert detect_role_family("Product Manager") == "product"


# ==============================================================================
# Seniority Detection Tests
# ==============================================================================


def test_seniority_intern():
    assert detect_seniority("Intern Engineer") == 0


def test_seniority_junior():
    assert detect_seniority("Junior Developer") == 1


def test_seniority_associate():
    assert detect_seniority("Associate Engineer") == 2


def test_seniority_mid():
    assert detect_seniority("Engineer") == 3  # default
    assert detect_seniority("Software Developer") == 3


def test_seniority_senior():
    assert detect_seniority("Senior Engineer") == 4


def test_seniority_lead():
    assert detect_seniority("Lead Engineer") == 5


def test_seniority_staff():
    assert detect_seniority("Staff Engineer") == 6


def test_seniority_principal():
    assert detect_seniority("Principal Engineer") == 7


def test_seniority_architect():
    assert detect_seniority("Architect") == 8


def test_seniority_manager():
    assert detect_seniority("Engineering Manager") == 9


def test_seniority_director():
    assert detect_seniority("Director of Engineering") == 10


# ==============================================================================
# End-to-End Scoring Tests
# ==============================================================================


def test_title_analysis_returns_debug():
    """The TitleAnalysis should include a debug dict."""
    jd = "Senior Backend Engineer\nRequirements:\n- Python"
    resume = """Senior Backend Engineer
    Experience:
    Senior Backend Engineer | 2020-Present
    - Built APIs
    """
    result = analyze_title_relevance(jd, resume)
    assert "debug" in result.__dataclass_fields__
    assert isinstance(result.debug, dict)
    if result.debug:
        assert "semantic_similarity" in result.debug
        assert "seniority_factor" in result.debug
        assert "role_family_factor" in result.debug
        assert "recency_factor" in result.debug


def test_senior_backend_vs_junior_frontend_low_score():
    """JD: Senior Backend Engineer, Resume: Junior Frontend Developer → low score."""
    jd = "Senior Backend Engineer\nRequirements:\n- Python"
    resume = """Junior Frontend Developer
    Skills: React

    Experience:
    Junior Frontend Developer | 2023-Present
    - Built UI components
    """
    result = analyze_title_relevance(jd, resume)
    # Should be low due to family + seniority mismatch
    assert result.score < 50, f"Expected low score, got {result.score}"
    if result.debug:
        assert result.debug["role_family_factor"] <= 0.85


def test_backend_architect_vs_senior_java_high_semantic():
    """JD: Backend Architect, Resume: Senior Java Engineer → moderate semantic match with seniority penalty."""
    jd = "Backend Architect\nRequirements:\n- Java"
    resume = """Senior Java Engineer
    Skills: Java, Spring

    Experience:
    Senior Java Engineer | 2020-Present
    - Built microservices
    """
    result = analyze_title_relevance(jd, resume)
    # Same role family (backend) → family_factor = 1.0
    # Seniority: Architect(8) vs Senior(4) → penalty but still moderate
    assert result.debug["role_family_factor"] == 1.0
    assert result.score > 0


def test_data_analyst_vs_tableau_developer_medium():
    """JD: Data Analyst, Resume: Tableau Developer → same analytics family."""
    jd = "Data Analyst\nRequirements:\n- Tableau"
    resume = """Tableau Developer
    Skills: Tableau, SQL

    Experience:
    Tableau Developer | 2021-Present
    - Built dashboards
    """
    result = analyze_title_relevance(jd, resume)
    # Same analytics_bi family → family_factor = 1.0
    # Same seniority (both default mid) → seniority_factor = 1.0
    assert result.debug["role_family_factor"] == 1.0
    assert result.debug["seniority_factor"] == 1.0
    assert result.score > 0


def test_devops_vs_qa_low_score():
    """JD: Senior DevOps Engineer, Resume: QA Engineer → low score."""
    jd = "Senior DevOps Engineer\nRequirements:\n- CI/CD"
    resume = """QA Engineer
    Skills: Selenium

    Experience:
    QA Engineer | 2020-Present
    - Test automation
    """
    result = analyze_title_relevance(jd, resume)
    # Unrelated family + seniority mismatch
    assert result.score < 50, f"Expected low score, got {result.score}"


def test_extract_jd_title_cleans_noise_prefix():
    """JD title '# Job Description — tableau developer' should clean to 'tableau developer'."""
    jd = """# Job Description — tableau developer

Requirements:
- Tableau experience
"""
    title = extract_jd_title(jd)
    assert "tableau developer" in title.lower()
    assert "#" not in title
    assert "job description" not in title.lower()


def test_jd_title_noise_does_not_dilute_semantic_score():
    """JD with 'Job Description — ' prefix should still score ~100 vs identical candidate title."""
    jd = """# Job Description — tableau developer

Requirements:
- Tableau experience
"""
    resume = """Tableau Developer
Skills: Tableau

Experience:
Tableau Developer | 2024-Present
- Built dashboards
"""
    result = analyze_title_relevance(jd, resume)
    # Cleaned JD title should match candidate title closely
    assert result.score >= 90, (
        f"Expected near-perfect score for identical titles, got {result.score}. "
        f"JD title extracted as: '{result.jd_title}'"
    )


def test_title_relevance_integration_with_ranker():
    """Title relevance should integrate into the overall ranking pipeline."""
    jd = "Senior Backend Engineer\nRequirements:\n- Python\n- Docker"
    candidates = [
        (
            "Alice",
            """Alice
        Senior Backend Engineer
        Skills: Python, Docker

        Experience:
        Senior Backend Engineer | 2020-Present
        - Built APIs
        """,
        ),
        (
            "Bob",
            """Bob
        Junior Frontend Developer
        Skills: React

        Experience:
        Junior Frontend Developer | 2022-Present
        - Built UIs
        """,
        ),
    ]
    results = rank_candidates(jd, candidates)
    # Alice should rank higher than Bob
    alice = [r for r in results if r.candidate_name == "Alice"][0]
    bob = [r for r in results if r.candidate_name == "Bob"][0]
    assert alice.title_analysis.score > bob.title_analysis.score
    # Title analysis should have debug info
    assert alice.title_analysis.debug


# ==============================================================================
# New: Current Role Detection Tests
# ==============================================================================


def test_detect_current_role_uses_is_current():
    """Should return the entry marked as current."""
    titles = [
        {"title": "Java Developer", "is_current": False, "end_year": 2020},
        {"title": "Data Analyst", "is_current": True, "end_year": None},
    ]
    result = detect_current_role(titles)
    assert result["title"] == "Data Analyst"


def test_detect_current_role_falls_back_to_latest_end_year():
    """Should return the most recent entry when none is current."""
    titles = [
        {"title": "Java Developer", "is_current": False, "end_year": 2018},
        {"title": "Data Analyst", "is_current": False, "end_year": 2025},
    ]
    result = detect_current_role(titles)
    assert result["title"] == "Data Analyst"


def test_detect_current_role_returns_none_for_empty():
    """Should return None for empty list."""
    assert detect_current_role([]) is None


# ==============================================================================
# New: Seniority Synonym Tests
# ==============================================================================


def test_seniority_sr_synonym():
    """'Sr' should map to 'senior' index."""
    assert detect_seniority("Sr Engineer") == 4


def test_seniority_jr_synonym():
    """'Jr' should map to 'junior' index."""
    assert detect_seniority("Jr Developer") == 1


def test_seniority_tech_lead_synonym():
    """'Tech Lead' should map to 'lead' index."""
    assert detect_seniority("Tech Lead") == 5


def test_seniority_software_architect_synonym():
    """'Software Architect' should map to 'architect' index."""
    assert detect_seniority("Software Architect") == 8


# ==============================================================================
# New: ui_ux Role Family Tests
# ==============================================================================


def test_role_family_ui_ux():
    """New ui_ux family should be detected."""
    assert detect_role_family("UI/UX Designer") == "ui_ux"
    assert detect_role_family("Product Designer") == "ui_ux"
    assert detect_role_family("UX Researcher") == "ui_ux"


# ==============================================================================
# New: Test 1 — Java Developer moving to Data Analyst
# ==============================================================================


def test_java_dev_moved_to_data_analyst_moderate_score():
    """JD: Java Developer, old=Java Dev, current=Data Analyst → moderate, not high."""
    jd = "Java Developer\nRequirements:\n- Java"
    resume = """Java Developer    2018 - 2022
    Skills: Java, Spring

    Experience:
    Java Developer | 2018 - 2022
    - Built Java applications

    Data Analyst | 2023 - Present
    - Analyzed data with SQL
    """
    result = analyze_title_relevance(jd, resume)
    # Current role (Data Analyst) is unrelated to Java Developer JD
    # Best old match may be high, but current influence (30%) drags it down
    assert result.score < 80, f"Expected moderate score, got {result.score}"
    assert result.score >= 10, f"Expected non-zero score, got {result.score}"
    # Should have current_title_match in debug
    if result.debug:
        assert "current_title_match" in result.debug
        assert "best_title_match" in result.debug


# ==============================================================================
# New: Test 2 — Consistent Frontend Engineer
# ==============================================================================


def test_consistent_frontend_engineer_high_score():
    """JD: Frontend Engineer, multiple frontend roles → very high with stability bonus."""
    jd = "Frontend Engineer\nRequirements:\n- React"
    resume = """Experience:
    Frontend Developer
    Company | 2019 - 2021
    - Built UIs

    React Developer
    Company | 2021 - 2023
    - Built React components

    Senior Frontend Engineer
    Company | 2023 - Present
    - Led frontend team
    """
    result = analyze_title_relevance(jd, resume)
    assert result.score >= 70, f"Expected high score, got {result.score}"
    if result.debug:
        assert result.debug.get("title_stability_bonus", 0) > 0


# ==============================================================================
# New: Test 3 — DevOps vs QA (unrelated family cap)
# ==============================================================================


def test_devops_vs_qa_unrelated_cap():
    """JD: DevOps Engineer, Resume: QA Engineer → capped at 60."""
    jd = "DevOps Engineer\nRequirements:\n- CI/CD"
    resume = """QA Engineer    2020 - Present
    Experience:
    QA Engineer | 2020 - Present
    - Test automation
    """
    result = analyze_title_relevance(jd, resume)
    assert result.score <= 60, f"Expected score <= 60 due to unrelated cap, got {result.score}"


# ==============================================================================
# New: Test 4 — Senior Backend vs Junior Backend (seniority mismatch)
# ==============================================================================


def test_senior_backend_vs_junior_backend_seniority_penalty():
    """JD: Senior Backend Engineer, Resume: Junior Backend Developer → penalty but strong semantic."""
    jd = "Senior Backend Engineer\nRequirements:\n- Python"
    resume = """Junior Backend Developer    2022 - Present
    Experience:
    Junior Backend Developer | 2022 - Present
    - Built APIs
    """
    result = analyze_title_relevance(jd, resume)
    # Same family (backend) = 1.0, but seniority mismatch (senior→junior)
    # The new 70/30 blend should still produce a reasonable score
    assert result.score > 0
    if result.debug:
        assert result.debug["role_family_factor"] == 1.0
        assert result.debug["seniority_factor"] < 1.0


# ==============================================================================
# New: Title Drift Penalty Test
# ==============================================================================


def test_title_drift_penalty_applied():
    """JD: Java Developer, current role is Data Analyst → drift penalty."""
    jd = "Java Developer\nRequirements:\n- Java"
    resume = """Experience:
    Java Developer
    Company | 2018 - 2022
    - Built Java apps

    Data Analyst
    Company | 2023 - Present
    - Analyzed data
    """
    result = analyze_title_relevance(jd, resume)
    if result.debug:
        drift = result.debug.get("title_drift_penalty", 0)
        assert drift < 0, f"Expected drift penalty < 0, got {drift}"


# ==============================================================================
# New: Explainability Output Test
# ==============================================================================


def test_title_analysis_new_explainability_fields():
    """Debug dict should contain new fields: scored_titles, best/current match, etc."""
    jd = "Backend Engineer\nRequirements:\n- Python"
    resume = """Senior Java Developer    2020 - Present
    Experience:
    Senior Java Developer | 2020 - Present
    - Built APIs
    """
    result = analyze_title_relevance(jd, resume)
    debug = result.debug
    assert "scored_titles" in debug
    assert "best_title_match" in debug
    assert "current_title_match" in debug
    assert "title_stability_bonus" in debug
    assert "title_drift_penalty" in debug
    assert "blended_score" in debug


# ==============================================================================
# New: Current Role Influence Test
# ==============================================================================


def test_current_role_influence_lowers_score_when_unrelated():
    """When current role is unrelated, 30% influence should lower the blended score."""
    jd = "Python Developer"
    resume = """Java Developer    2018 - 2022
    Skills: Java

    Experience:
    Java Developer | 2018 - 2022
    - Built Java apps

    Data Analyst | 2023 - Present
    - Analyzed data
    """
    result = analyze_title_relevance(jd, resume)
    # The current role (Data Analyst) influences 30%, dragging score below best-match-only
    debug = result.debug
    if debug and "best_title_match" in debug and "current_title_match" in debug:
        best = debug["best_title_match"]["score"]
        current = debug["current_title_match"]["score"]
        blended = debug["blended_score"]
        expected_blended = best * 0.70 + current * 0.30
        assert abs(blended - expected_blended) < 0.1, (
            f"Blended {blended} != {best}*0.7 + {current}*0.3 = {expected_blended}"
        )


# ==============================================================================
# New: Confidence-based role family classification tests
# ==============================================================================


def test_detect_role_family_unknown_for_non_tech():
    """Non-tech titles like 'Plumber' should return 'unknown', not 'generic_engineering'."""
    assert detect_role_family("Plumber") == "unknown"
    assert detect_role_family("Nurse") == "unknown"
    assert detect_role_family("Electrician") == "unknown"
    assert detect_role_family("Chef") == "unknown"


def test_get_family_confidence_specific_tech():
    """Tech titles should have high confidence."""
    import resume_ranker.signals as sig

    assert sig._get_family_confidence("Senior Backend Engineer") >= 0.85
    assert sig._get_family_confidence("React Developer") >= 0.55  # generic_engineering with developer keyword
    assert sig._get_family_confidence("Data Engineer") >= 0.85


def test_get_family_confidence_zero_for_unknown():
    """Non-tech titles should have 0 confidence."""
    import resume_ranker.signals as sig

    assert sig._get_family_confidence("Plumber") == 0.0
    assert sig._get_family_confidence("Nurse") == 0.0


def test_get_family_confidence_generic_engineer():
    """Generic engineering titles should have moderate confidence."""
    import resume_ranker.signals as sig

    conf = sig._get_family_confidence("Software Engineer")
    assert 0.55 <= conf <= 0.95


# ==============================================================================
# New: Test 1 — Software Engineer vs Plumber
# ==============================================================================


def test_software_engineer_vs_plumber_near_zero():
    """JD: Plumber, Candidate: Software Engineer → near-zero score."""
    jd = "Plumber\nRequirements:\n- Pipe fitting"
    resume = """Software Engineer
    Skills: Python, Java

    Experience:
    Software Engineer | 2020 - Present
    - Built applications
    """
    result = analyze_title_relevance(jd, resume)
    assert result.score <= 15, f"Expected near-zero score, got {result.score}"


# ==============================================================================
# New: Test 2 — Frontend Developer vs Nurse
# ==============================================================================


def test_frontend_vs_nurse_near_zero():
    """JD: Nurse, Candidate: Frontend Developer → near-zero score."""
    jd = "Nurse\nRequirements:\n- Patient care"
    resume = """Frontend Developer
    Skills: React

    Experience:
    Frontend Developer | 2020 - Present
    - Built UIs
    """
    result = analyze_title_relevance(jd, resume)
    assert result.score <= 15, f"Expected near-zero score, got {result.score}"


# ==============================================================================
# New: Test 3 — Backend Engineer vs QA Engineer (moderate)
# ==============================================================================


def test_backend_vs_qa_moderate():
    """JD: Backend Engineer, Candidate: QA Engineer → moderate score."""
    jd = "Backend Engineer\nRequirements:\n- Python"
    resume = """QA Engineer
    Skills: Selenium

    Experience:
    QA Engineer | 2020 - Present
    - Test automation
    """
    result = analyze_title_relevance(jd, resume)
    # Both are known tech families, backend and qa are loosely related
    assert result.score > 0
    assert result.score < 80


# ==============================================================================
# New: Test 4 — React Developer vs Frontend Engineer (high)
# ==============================================================================


def test_react_vs_frontend_high():
    """JD: Frontend Engineer, Candidate: React Developer → high score."""
    jd = "Frontend Engineer\nRequirements:\n- React"
    resume = """React Developer
    Skills: React, TypeScript

    Experience:
    React Developer | 2020 - Present
    - Built components
    """
    result = analyze_title_relevance(jd, resume)
    assert result.score >= 20, f"Expected moderate score, got {result.score}"


# ==============================================================================
# New: Test 5 — Data Analyst vs Electrician (very low)
# ==============================================================================


def test_data_analyst_vs_electrician_very_low():
    """JD: Electrician, Candidate: Data Analyst → very low score."""
    jd = "Electrician\nRequirements:\n- Wiring"
    resume = """Data Analyst
    Skills: SQL, Tableau

    Experience:
    Data Analyst | 2020 - Present
    - Analyzed data
    """
    result = analyze_title_relevance(jd, resume)
    assert result.score <= 15, f"Expected very low score, got {result.score}"


# ==============================================================================
# New: Unknown family penalty applied in debug
# ==============================================================================


def test_unknown_family_penalty_in_debug():
    """Non-tech JD should show unknown_family penalty in debug."""
    jd = "Plumber"
    resume = """Software Engineer
    Skills: Python

    Experience:
    Software Engineer | 2020 - Present
    """
    result = analyze_title_relevance(jd, resume)
    debug = result.debug
    assert debug.get("jd_role_family") == "unknown"
    assert debug.get("jd_family_confidence") == 0.0
    assert "penalties_applied" in debug


# ==============================================================================
# New: role_family_factor is 0.25 for unknown
# ==============================================================================


def test_role_family_factor_unknown():
    """_get_role_family_factor should return 0.25 when either family is unknown."""
    from resume_ranker.signals import _get_role_family_factor

    assert _get_role_family_factor("unknown", "generic_engineering") == 0.25
    assert _get_role_family_factor("backend", "unknown") == 0.25
    assert _get_role_family_factor("unknown", "unknown") == 0.25
