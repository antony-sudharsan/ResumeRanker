"""Tests for Roles & Responsibilities matching."""

from resume_ranker.roles import (
    calculate_roles_score,
    extract_responsibilities,
    _tfidf_similarity,
    _semantic_bullet_similarity,
    _action_verb_score,
    _section_confidence,
    _prevent_false_positives,
)


def test_extract_responsibilities_handles_bullets():
    bullets = extract_responsibilities("- Designed REST APIs\n- Built microservices")
    assert len(bullets) >= 2
    assert any("Designed" in b for b in bullets)
    assert any("Built" in b for b in bullets)


def test_extract_responsibilities_skips_short_lines():
    bullets = extract_responsibilities("- Hi\n- Designed REST APIs")
    assert len(bullets) >= 1
    assert all(len(b) > 15 for b in bullets)


def test_extract_responsibilities_empty():
    assert extract_responsibilities("") == []
    assert extract_responsibilities("   \n\n") == []


def test_tfidf_similarity_perfect_match():
    jd = ["Designed and implemented REST APIs"]
    resume = ["Designed and implemented REST APIs"]
    sim = _tfidf_similarity(resume, jd)
    assert 0.5 <= sim <= 1.0


def test_tfidf_similarity_no_match():
    jd = ["Requires deep knowledge of quantum physics"]
    resume = ["Built frontend UI components with React"]
    sim = _tfidf_similarity(resume, jd)
    assert sim < 0.5


def test_tfidf_similarity_empty_inputs():
    assert _tfidf_similarity([], ["test"]) == 0.0
    assert _tfidf_similarity(["test"], []) == 0.0
    assert _tfidf_similarity([], []) == 0.0


def test_semantic_bullet_similarity_good_match():
    jd = ["Design and implement REST APIs"]
    resume = ["Built REST APIs using Python and Flask"]
    sim = _semantic_bullet_similarity(resume, jd)
    assert sim > 0.3


def test_semantic_bullet_similarity_no_match():
    jd = ["Manage Kubernetes clusters"]
    resume = ["Designed UI mockups in Figma"]
    sim = _semantic_bullet_similarity(resume, jd)
    assert sim < 0.4


def test_semantic_bullet_similarity_empty():
    assert _semantic_bullet_similarity([], ["test"]) == 0.0
    assert _semantic_bullet_similarity(["test"], []) == 0.0


def test_action_verb_score_perfect():
    jd = ["Designed and implemented microservices"]
    resume = ["Implemented new features", "Designed the architecture"]
    score = _action_verb_score(resume, jd)
    assert score > 0.5


def test_action_verb_score_no_overlap():
    jd = ["Designed and built data pipelines"]
    resume = ["Attended team standup meetings"]
    score = _action_verb_score(resume, jd)
    assert score <= 0.5


def test_action_verb_score_empty_jd():
    assert _action_verb_score(["Built things"], []) == 1.0


def test_section_confidence_well_structured():
    text = "Professional Experience\n- Designed APIs using Python\n- Built dashboards\n- Led a team of engineers\n"
    score = _section_confidence(text)
    assert score >= 0.7


def test_section_confidence_poorly_structured():
    text = "just some text with no real structure"
    score = _section_confidence(text)
    assert score <= 0.7


def test_section_confidence_empty():
    assert _section_confidence("") >= 0.5


def test_prevent_false_positives_stuffing():
    resume = ["React React React Docker AWS Docker"]
    jd = ["React Docker AWS"]
    semantic = 0.9
    adjusted = _prevent_false_positives(resume, jd, semantic)
    assert adjusted < semantic


def test_prevent_false_positives_no_stuffing():
    resume = ["Built production React apps deployed on AWS ECS"]
    jd = ["React Docker AWS"]
    semantic = 0.9
    adjusted = _prevent_false_positives(resume, jd, semantic)
    assert adjusted <= semantic


def test_calculate_roles_score_full_match():
    jd_text = """Responsibilities:
    - Design and implement scalable backend services
    - Write clean, maintainable code
    """
    resume_text = """Experience:
    - Designed and implemented scalable backend services
    - Wrote clean, maintainable code in Python
    """
    result = calculate_roles_score(jd_text, resume_text)
    assert "roles_score" in result
    assert result["roles_score"] >= 0
    assert result["roles_score"] <= 100
    assert result["best_role_matches"]


def test_calculate_roles_score_no_match():
    jd_text = """Responsibilities:
    - Manage Kubernetes cluster operations
    """
    resume_text = """Experience:
    - Designed marketing brochures in Adobe Photoshop
    """
    result = calculate_roles_score(jd_text, resume_text)
    assert result is not None
    assert "roles_score" in result


def test_calculate_roles_score_empty():
    result = calculate_roles_score("", "")
    assert result["roles_score"] == 0.0


def test_calculate_roles_score_has_all_components():
    jd_text = """Responsibilities:
    - Build REST APIs
    """
    resume_text = """Experience:
    - Built REST APIs with Python
    """
    result = calculate_roles_score(jd_text, resume_text)
    assert "tfidf_score" in result
    assert "semantic_score" in result
    assert "action_verb_score" in result
    assert "section_confidence" in result
    assert "best_role_matches" in result
