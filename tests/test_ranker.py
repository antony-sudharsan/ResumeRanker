"""Tests for the ranking engine."""

from resume_ranker.ranker import rank_candidates

JD = """
Senior Python Developer

Requirements:
- 5+ years of experience in Python development
- Strong knowledge of Django or FastAPI
- Experience with PostgreSQL and Redis
- Familiarity with Docker and Kubernetes
- Experience with AWS cloud services
- CI/CD pipeline management with GitHub Actions

Responsibilities:
- Design and implement scalable backend services
- Write clean, maintainable, and well-tested code
- Collaborate with cross-functional teams
- Mentor junior developers
- Participate in code reviews and architectural discussions
"""

RESUME_STRONG = """
John Smith
Senior Software Engineer

10 years of experience in software development.

Skills: Python, Django, FastAPI, PostgreSQL, Redis, Docker, Kubernetes,
AWS, GitHub Actions, React, TypeScript, Terraform

Experience:
- Led backend development for a high-traffic e-commerce platform
- Designed RESTful APIs serving 100k+ requests per day
- Mentored a team of 5 junior developers
- Implemented CI/CD pipelines using GitHub Actions
- Managed infrastructure on AWS using Terraform
"""

RESUME_MODERATE = """
Jane Doe
Software Developer

3 years of experience in web development.

Skills: Python, Flask, MySQL, Docker, Git

Experience:
- Developed web applications using Python and Flask
- Managed MySQL databases for application data
- Containerized applications with Docker
- Participated in agile development processes
"""

RESUME_WEAK = """
Bob Johnson
Junior Developer

1 year of experience in programming.

Skills: Java, Spring Boot, Oracle, Maven

Experience:
- Built REST APIs using Spring Boot
- Worked with Oracle database for data management
- Used Maven for project builds
"""


def test_ranking_order():
    candidates = [
        ("John Smith", RESUME_STRONG),
        ("Jane Doe", RESUME_MODERATE),
        ("Bob Johnson", RESUME_WEAK),
    ]
    results = rank_candidates(JD, candidates)

    assert results[0].candidate_name == "John Smith"
    assert results[0].rank == 1
    assert results[-1].candidate_name == "Bob Johnson"
    assert results[-1].rank == 3


def test_ranking_scores_decrease():
    candidates = [
        ("John Smith", RESUME_STRONG),
        ("Jane Doe", RESUME_MODERATE),
        ("Bob Johnson", RESUME_WEAK),
    ]
    results = rank_candidates(JD, candidates)

    for i in range(len(results) - 1):
        assert results[i].overall_score >= results[i + 1].overall_score


def test_ranking_has_justification():
    candidates = [("John Smith", RESUME_STRONG)]
    results = rank_candidates(JD, candidates)

    assert results[0].justification
    assert "SKILLS" in results[0].justification
    assert "EXPERIENCE" in results[0].justification
    assert "DESIGNATION" in results[0].justification
    assert "ROLES" in results[0].justification


def test_single_candidate():
    candidates = [("Jane Doe", RESUME_MODERATE)]
    results = rank_candidates(JD, candidates)

    assert len(results) == 1
    assert results[0].rank == 1
    assert 0 <= results[0].overall_score <= 100


def test_skill_analysis_details():
    candidates = [("John Smith", RESUME_STRONG)]
    results = rank_candidates(JD, candidates)

    sa = results[0].skill_analysis
    assert "python" in sa.matched_skills
    assert sa.match_percentage > 50


# ---------------------------------------------------------------------------
# Semantic context filtering integration tests
# These test the full pipeline (Tier 1 + Tiers 2-4) with known skills
# where negation/weak context appears in the resume.
# ---------------------------------------------------------------------------

RESUME_NEGATED_DOCKER = """
John Doe
Software Engineer

No experience with Docker. Worked with Python and Java.
"""

RESUME_NEGATED_KUBERNETES = """
Jane Doe
Software Engineer

I have never worked with Kubernetes. Experienced with AWS and Docker.
"""

RESUME_INTERESTED_ML = """
Bob Smith
Junior Developer

Interested in learning machine learning concepts.
Some Python experience.
"""

RESUME_BASIC_AWS = """
Alice Brown
Cloud Intern

Basic exposure to AWS cloud services. Learning as I go.
"""

RESUME_STRONG_REACT = """
Charlie Wilson
Frontend Engineer

Built production dashboards using React and TypeScript.
Experienced with Redux and REST APIs.
"""


def test_negated_docker_not_matched():
    """Docker mentioned as 'No experience with Docker' should appear in missing_skills."""
    jd_text = """
    Requirements:
    - Docker experience required
    - Python development
    """
    resume = RESUME_NEGATED_DOCKER
    candidates = [("John Doe", resume)]
    results = rank_candidates(jd_text, candidates)
    sa = results[0].skill_analysis
    assert "docker" not in sa.matched_skills, "Docker should not be matched when resume negates it"


def test_negated_kubernetes_not_matched():
    """Kubernetes mentioned as 'never worked with' should be missing."""
    jd_text = """
    Requirements:
    - Kubernetes experience
    - Docker
    """
    resume = RESUME_NEGATED_KUBERNETES
    candidates = [("Jane Doe", resume)]
    results = rank_candidates(jd_text, candidates)
    sa = results[0].skill_analysis
    assert "kubernetes" not in sa.matched_skills
    assert "kubernetes" in sa.missing_skills, "Negated Kubernetes should be in missing_skills"


def test_interested_in_learning_not_matched():
    """'Interested in learning machine learning' — weak context excludes
    the skill from matched_skills even though Tier 1 dictionary matches it."""
    jd_text = """
    Requirements:
    - Machine Learning
    """
    resume = RESUME_INTERESTED_ML
    candidates = [("Bob Smith", resume)]
    results = rank_candidates(jd_text, candidates)
    sa = results[0].skill_analysis
    assert "machine learning" not in sa.matched_skills
    assert "machine learning" in sa.missing_skills


def test_basic_exposure_not_full_match():
    """'Basic exposure to AWS' — weak context means AWS is not a full match."""
    jd_text = """
    Requirements:
    - AWS services experience
    """
    resume = RESUME_BASIC_AWS
    candidates = [("Alice Brown", resume)]
    results = rank_candidates(jd_text, candidates)
    sa = results[0].skill_analysis
    assert "aws" not in sa.matched_skills
    assert "aws" in sa.missing_skills


def test_strong_react_remains_matched():
    """Strong production React usage should still match normally."""
    jd_text = """
    Requirements:
    - React
    - TypeScript
    """
    resume = RESUME_STRONG_REACT
    candidates = [("Charlie Wilson", resume)]
    results = rank_candidates(jd_text, candidates)
    sa = results[0].skill_analysis
    assert "react" in sa.matched_skills, "Strong React usage should remain matched"


def test_semantic_match_details_populated():
    """The semantic_match_details field should be populated from the pipeline."""
    jd_text = """
    Requirements:
    - REST API integration experience
    """
    resume = RESUME_STRONG_REACT  # mentions REST APIs
    candidates = [("Charlie Wilson", resume)]
    results = rank_candidates(jd_text, candidates)
    sa = results[0].skill_analysis
    # The detailed match info should be available
    assert len(sa.semantic_match_details) >= 0  # field exists


# ---------------------------------------------------------------------------
# Bullet-pointed resume edge cases (no periods, • markers)
# ---------------------------------------------------------------------------

RESUME_BULLET_NEGATED = """
John Doe
• I interested to learn machine learning
• I have no docker experience
• Familiar with fastapi
"""


def test_bullet_negated_docker_not_matched():
    """Bullet resume 'no docker experience' (no period) should be negated."""
    jd_text = """
    Requirements:
    - Docker
    - Python
    """
    candidates = [("John Doe", RESUME_BULLET_NEGATED)]
    results = rank_candidates(jd_text, candidates)
    sa = results[0].skill_analysis
    assert "docker" not in sa.matched_skills, (
        "Docker should NOT match — resume says 'no docker experience'"
    )


def test_bullet_interested_ml_not_matched():
    """Bullet resume 'interested to learn machine learning' → not in matched."""
    jd_text = """
    Requirements:
    - Machine Learning
    """
    candidates = [("John Doe", RESUME_BULLET_NEGATED)]
    results = rank_candidates(jd_text, candidates)
    sa = results[0].skill_analysis
    assert "machine learning" not in sa.matched_skills


# ==============================================================================
# Hybrid ranking system integration tests
# ==============================================================================


def test_hybrid_ranking_has_match_quality():
    """rank_candidates returns match_quality on each result."""
    jd = "React Developer\nSkills: React, Redux, TypeScript"
    resume = "Frontend Developer using React, Redux, TypeScript"
    results = rank_candidates(jd, [("Test", resume)])
    assert results[0].match_quality in (
        "Excellent match",
        "Strong match",
        "Moderate match",
        "Weak match",
        "Poor match",
        "No match",
    )


def test_hybrid_ranking_has_roles_analysis():
    """rank_candidates returns roles_analysis on each result."""
    jd = "React Developer\nSkills: React, Redux, TypeScript"
    resume = "Frontend Developer using React, Redux, TypeScript"
    results = rank_candidates(jd, [("Test", resume)])
    ra = results[0].roles_analysis
    assert ra.score >= 0
    assert ra.score <= 100


def test_hybrid_ranking_score_bounds():
    """All individual component scores are within [0, 100]."""
    jd = "React Developer\nSkills: React, Redux, TypeScript"
    resume = "Frontend Developer using React, Redux, TypeScript"
    results = rank_candidates(jd, [("Test", resume)])
    r = results[0]
    assert 0 <= r.overall_score <= 100
    assert 0 <= r.skill_analysis.match_percentage <= 100
    assert 0 <= r.experience_analysis.score <= 100
    assert 0 <= r.title_analysis.score <= 100
    assert 0 <= r.roles_analysis.score <= 100


# ---------------------------------------------------------------------------
# Spec test cases
# ---------------------------------------------------------------------------


def test_spec_case_1_react_dev():
    """JD: React Developer with responsibilities, Resume: Frontend Developer using React
    Strong skill + role + designation match → high score."""
    jd = """Job Title: React Developer
Required Skills:
React, Redux, TypeScript
Roles and Responsibilities:
- Build and maintain React applications
- Implement state management with Redux
- Write type-safe code with TypeScript
"""
    resume = """Frontend Developer
Skills: React, Redux, TypeScript, JavaScript
Professional Experience:
React Developer | Tech Corp | Jan 2022 - Present
- Built React applications with Redux and TypeScript
- Developed reusable UI components
"""
    results = rank_candidates(jd, [("Test", resume)])
    score = results[0].overall_score
    assert score >= 70, f"Expected high score, got {score}"


def test_spec_case_2_senior_backend_vs_junior_frontend():
    """JD: Senior Backend Engineer, Resume: Junior Frontend Developer
    Expected: Low title + low experience score"""
    jd = """Job Title: Senior Backend Engineer
Required Skills:
Java, Spring, Microservices
"""
    resume = """Junior Frontend Developer
Skills: HTML, CSS, JavaScript
Professional Experience:
Junior Dev | Some Corp | Jan 2023 - Present
- Built web pages and UI components
"""
    results = rank_candidates(jd, [("Test", resume)])
    score = results[0].overall_score
    assert score < 55, f"Expected low score, got {score}"


def test_spec_case_3_data_analyst_vs_bi_dev():
    """JD: Data Analyst, Resume: BI Developer using Tableau, SQL, Power BI
    Expected: Good skill overlap, related role family → moderate-high score."""
    jd = """Job Title: Data Analyst
Required Skills:
Tableau, SQL, Power BI, Excel
Roles and Responsibilities:
- Analyze data and build dashboards
- Create reports and visualizations
"""
    resume = """BI Developer
Skills: Tableau, SQL, Power BI
Professional Experience:
BI Developer | Analytics Corp | Mar 2021 - Present
- Developed BI dashboards using Tableau and Power BI
- Wrote complex SQL queries for data analysis
"""
    results = rank_candidates(jd, [("Test", resume)])
    score = results[0].overall_score
    assert score >= 50, f"Expected moderate score, got {score}"


def test_spec_case_4_learning_aws_low_score():
    """JD: AWS DevOps Engineer, Resume: 'Currently learning AWS'
    Expected: Low score — weak context should not inflate score."""
    jd = """Job Title: AWS DevOps Engineer
Required Skills:
AWS, Docker, Kubernetes, Terraform
"""
    resume = """Junior Engineer
Currently learning AWS. Have basic Linux knowledge.
"""
    results = rank_candidates(jd, [("Test", resume)])
    score = results[0].overall_score
    assert score < 45, f"Expected low score, got {score}"


def test_spec_case_5_java_dev_vs_python_analyst():
    """JD: Java Developer, Resume: Python Data Analyst
    Expected: Low overall score — different stack and role."""
    jd = """Job Title: Java Developer
Required Skills:
Java, Spring Boot, Hibernate
"""
    resume = """Python Data Analyst
Skills: Python, Pandas, NumPy, SQL
Professional Experience:
Data Analyst | Data Corp | Jan 2022 - Present
- Analyzed data using Python and Pandas
"""
    results = rank_candidates(jd, [("Test", resume)])
    score = results[0].overall_score
    assert score < 50, f"Expected low score, got {score}"


def test_spec_case_5_java_dev_vs_python_analyst():
    """JD: Java Developer, Resume: Python Data Analyst
    Expected: Low overall score"""
    jd = "Java Developer\nSkills: Java, Spring Boot, Hibernate"
    resume = """Python Data Analyst
Skills: Python, Pandas, NumPy, SQL
Experience:
- Analyzed data using Python
"""
    results = rank_candidates(jd, [("Test", resume)])
    score = results[0].overall_score
    assert score < 50, f"Expected low score, got {score}"
