"""Tests for the improved Title Relevance scoring system."""

from resume_ranker.signals import (
    TitleAnalysis,
    analyze_title_relevance,
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
