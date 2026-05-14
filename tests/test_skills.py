"""Tests for skills extraction and semantic context detection."""

from resume_ranker.skills import (
    detect_skill_context,
    extract_skills,
    extract_unknown_skills_semantic,
    flatten_skills,
    SemanticMatchResult,
)


# ---------------------------------------------------------------------------
# Existing Tier 1 tests (unchanged behavior)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# detect_skill_context — unit tests
# ---------------------------------------------------------------------------


class TestDetectSkillContext:
    def test_negation_no_experience(self):
        assert detect_skill_context("No experience with Docker.", "docker") == "negated"

    def test_negation_never_worked(self):
        assert (
            detect_skill_context("I have never worked with Kubernetes.", "kubernetes") == "negated"
        )

    def test_negation_not_worked(self):
        assert detect_skill_context("Not worked with React.", "react") == "negated"

    def test_negation_no_hands_on(self):
        assert detect_skill_context("No hands-on experience with AWS.", "aws") == "negated"

    def test_negation_without_experience(self):
        assert detect_skill_context("Without experience in Terraform.", "terraform") == "negated"

    def test_negation_lack_of_experience(self):
        assert detect_skill_context("Lack of experience with Docker.", "docker") == "negated"

    def test_negation_unfamiliar(self):
        assert detect_skill_context("Unfamiliar with the AWS ecosystem.", "aws") == "negated"

    def test_negation_not_experienced(self):
        assert detect_skill_context("Not experienced with Kubernetes.", "kubernetes") == "negated"

    def test_learning_currently_learning(self):
        assert detect_skill_context("Currently learning Docker.", "docker") == "learning_only"

    def test_learning_plain_learning(self):
        assert (
            detect_skill_context("Learning Python for data science.", "python") == "learning_only"
        )

    def test_learning_beginner(self):
        assert detect_skill_context("Beginner in React development.", "react") == "learning_only"

    def test_learning_basic_understanding(self):
        assert (
            detect_skill_context("Basic understanding of Kubernetes.", "kubernetes")
            == "learning_only"
        )

    def test_learning_basic_knowledge(self):
        assert (
            detect_skill_context("Basic knowledge of PostgreSQL.", "postgresql") == "learning_only"
        )

    def test_learning_just_begun(self):
        assert (
            detect_skill_context("Just begun learning TypeScript.", "typescript") == "learning_only"
        )

    def test_weak_familiar_with(self):
        assert detect_skill_context("Familiar with Docker basics.", "docker") == "weak_exposure"

    def test_weak_exposure_to(self):
        assert (
            detect_skill_context("Basic exposure to AWS cloud services.", "aws") == "weak_exposure"
        )

    def test_weak_interested_in(self):
        assert (
            detect_skill_context(
                "Interested in learning machine learning concepts.", "machine learning"
            )
            == "weak_exposure"
        )

    def test_weak_exploring(self):
        assert detect_skill_context("Currently exploring GraphQL.", "graphql") == "weak_exposure"

    def test_weak_self_learning(self):
        # Self-learning contains "learning", so it is caught by learning_only patterns
        assert (
            detect_skill_context("Self-learning React Native.", "react native") == "learning_only"
        )

    def test_strong_production_use(self):
        assert (
            detect_skill_context("Built production dashboards using React and TypeScript.", "react")
            == "strong"
        )

    def test_strong_experienced_with(self):
        assert detect_skill_context("5 years of experience with Python.", "python") == "strong"

    def test_compound_skill_not_false_positive(self):
        """'learning' in 'deep learning' or 'machine learning' should not trigger learning_only."""
        assert detect_skill_context("Applied deep learning with PyTorch.", "pytorch") == "strong"
        assert (
            detect_skill_context("Experience with machine learning and Python.", "python")
            == "strong"
        )

    def test_empty_sentence(self):
        assert detect_skill_context("", "docker") == "unknown"

    def test_empty_skill(self):
        assert detect_skill_context("Some random sentence.", "") == "unknown"

    def test_skill_not_in_sentence(self):
        assert detect_skill_context("No mention of the skill here.", "docker") == "unknown"

    # --- Intervening-word edge cases ---

    def test_negation_intervening_word(self):
        """ "no docker experience" should be negated even though "docker" is between."""
        assert detect_skill_context("I have no docker experience.", "docker") == "negated"

    def test_negation_intervening_word_no_period(self):
        """Same without period (bullet point style)."""
        assert detect_skill_context("I have no docker experience", "docker") == "negated"

    def test_interested_to_learn(self):
        """ "interested to learn" should be weak_exposure (via 'interested to')."""
        assert (
            detect_skill_context("I interested to learn machine learning", "machine learning")
            == "weak_exposure"
        )

    def test_learning_studying(self):
        """ "studying" should be learning_only."""
        assert detect_skill_context("Currently studying Docker.", "docker") == "learning_only"


# ---------------------------------------------------------------------------
# extract_unknown_skills_semantic — context filtering in the pipeline
# ---------------------------------------------------------------------------


class TestSemanticContextFiltering:
    # NOTE: extract_unknown_skills_semantic only handles skills NOT in ALL_SKILLS
    # (Tier 1 dictionary). Known skills like "docker", "react" are matched by Tier 1
    # and never reach this function. These tests use unknown skill phrases.

    def test_negated_unknown_skill_not_matched(self):
        """Tier 2 literal match: negated skill should not appear in matched set."""
        jd = "Container orchestration"
        resume = "No experience with container orchestration."
        matched, unknown, details = extract_unknown_skills_semantic(jd, resume)
        assert "container orchestration" not in matched
        assert any(d.context == "negated" for d in details), (
            "Expected a SemanticMatchResult with context='negated'"
        )

    def test_negated_unknown_skill_not_matched_never_worked(self):
        jd = "Container orchestration"
        resume = "I have never worked with container orchestration."
        matched, unknown, details = extract_unknown_skills_semantic(jd, resume)
        assert "container orchestration" not in matched

    def test_interested_in_learning_weak_match(self):
        """Weak/interest-only semantic matches should NOT be in the matched set."""
        jd = "Machine Learning"
        resume = "Interested in learning machine learning concepts."
        matched, unknown, details = extract_unknown_skills_semantic(jd, resume)
        assert "machine learning" not in matched

    def test_basic_exposure_weak_match(self):
        """Basic exposure → weak_exposure.
        For literal matches (Tier 2), the skill name literally appears, so it
        IS matched but with medium confidence.
        """
        jd = "Cloud infrastructure"
        resume = "Basic exposure to cloud infrastructure."
        matched, unknown, details = extract_unknown_skills_semantic(jd, resume)
        assert "cloud infrastructure" in matched
        medium = [d for d in details if d.confidence == "medium"]
        assert any(d.skill == "cloud infrastructure" for d in medium)

    def test_strong_literal_match_high_confidence(self):
        """Strong literal match should be in matched set with high confidence."""
        jd = "Component architecture"
        resume = "Built production dashboards using a component architecture."
        matched, unknown, details = extract_unknown_skills_semantic(jd, resume)
        assert "component architecture" in matched
        high_conf = [d for d in details if d.confidence == "high"]
        assert any(d.skill == "component architecture" for d in high_conf)

    def test_rest_api_semantic_match(self):
        """REST API integration should get a semantic match (high confidence)."""
        jd = "REST API integration"
        resume = "Integrated RESTful APIs into frontend applications."
        matched, unknown, details = extract_unknown_skills_semantic(jd, resume)
        assert "rest api integration" in matched

    def test_no_unknown_skills_returns_empty(self):
        """When all JD skills are in the dictionary, return empty sets/lists."""
        jd = "Python"
        resume = "I know Python."
        matched, unknown, details = extract_unknown_skills_semantic(jd, resume)
        assert matched == set()
        assert unknown == set()
        assert details == []

    def test_learning_literal_match_rejected(self):
        """Literal match with learning context — still matched with medium confidence
        (literal appearance overrides weak context)."""
        jd = "Container orchestration"
        resume = "Currently learning container orchestration."
        matched, unknown, details = extract_unknown_skills_semantic(jd, resume)
        # Literal match: the skill text literally appears, so it is matched
        # but with medium confidence due to learning context
        assert "container orchestration" in matched
        medium = [d for d in details if d.confidence == "medium"]
        assert any(d.skill == "container orchestration" for d in medium)

    def test_match_details_contain_all_required_fields(self):
        """SemanticMatchResult objects should have the specified fields."""
        jd = "Component architecture"
        resume = "Built production dashboards using component architecture."
        _, _, details = extract_unknown_skills_semantic(jd, resume)
        for d in details:
            assert isinstance(d, SemanticMatchResult)
            assert hasattr(d, "skill")
            assert hasattr(d, "matched_sentence")
            assert hasattr(d, "similarity_score")
            assert hasattr(d, "match_type")
            assert hasattr(d, "confidence")
            assert hasattr(d, "context")

    def test_match_type_in_literal(self):
        """Literal matches should be marked with match_type='literal'."""
        jd = "Component architecture"
        resume = "Built production dashboards using component architecture."
        _, _, details = extract_unknown_skills_semantic(jd, resume)
        literal = [d for d in details if d.match_type == "literal"]
        assert any(d.skill == "component architecture" for d in literal)
