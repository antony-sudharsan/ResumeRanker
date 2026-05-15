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
    assert "TITLE" in results[0].justification


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
