"""Tests for experience extraction."""

from resume_ranker.experience import extract_required_experience, extract_years_of_experience


def test_extract_years_basic():
    assert extract_years_of_experience("10 years of experience in software") == 10.0


def test_extract_years_plus():
    assert extract_years_of_experience("5+ years experience") == 5.0


def test_extract_years_over():
    assert extract_years_of_experience("Over 8 years of experience") == 8.0


def test_extract_years_none():
    assert extract_years_of_experience("Some professional background") is None


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
