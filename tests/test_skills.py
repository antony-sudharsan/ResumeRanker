"""Tests for skills extraction."""

from resume_ranker.skills import extract_skills, flatten_skills


def test_extract_python_and_java():
    text = "Experienced in Python and Java development"
    result = extract_skills(text)
    flat = flatten_skills(result)
    assert "python" in flat
    assert "java" in flat


def test_extract_web_frameworks():
    text = "Built applications with React, Django, and FastAPI"
    result = extract_skills(text)
    flat = flatten_skills(result)
    assert "react" in flat
    assert "django" in flat
    assert "fastapi" in flat


def test_extract_cloud_platforms():
    text = "Deployed services on AWS and Google Cloud Platform"
    result = extract_skills(text)
    flat = flatten_skills(result)
    assert "aws" in flat
    assert "google cloud platform" in flat


def test_no_false_positives():
    text = "Managed a team of engineers and led daily standups."
    result = extract_skills(text)
    flat = flatten_skills(result)
    assert "java" not in flat
    assert "c" not in flat


def test_case_insensitive():
    text = "Strong experience with KUBERNETES and Docker"
    result = extract_skills(text)
    flat = flatten_skills(result)
    assert "kubernetes" in flat
    assert "docker" in flat


def test_empty_text():
    result = extract_skills("")
    assert result == {}
