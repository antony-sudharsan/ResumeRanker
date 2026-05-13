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
