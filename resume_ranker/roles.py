"""Roles & Responsibilities matching — TF-IDF, semantic, action verb, section-aware."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from resume_ranker.semantic import semantic_similarity

_ACTION_VERBS: set[str] = {
    "designed",
    "implemented",
    "built",
    "developed",
    "created",
    "architected",
    "engineered",
    "deployed",
    "maintained",
    "optimized",
    "refactored",
    "migrated",
    "integrated",
    "configured",
    "automated",
    "orchestrated",
    "monitored",
    "managed",
    "led",
    "mentored",
    "coordinated",
    "delivered",
    "launched",
    "authored",
    "established",
    "improved",
    "reduced",
    "increased",
    "scaled",
    "analyzed",
    "evaluated",
    "defined",
    "documented",
    "tested",
    "validated",
    "troubleshot",
    "resolved",
    "supported",
    "trained",
    "presented",
    "reported",
    "drove",
    "championed",
    "initiated",
    "transformed",
    "modernized",
    "standardized",
    "streamlined",
    "consolidated",
    "centralized",
}


_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:job\s+title|designation|role|position|opening)", re.IGNORECASE),
    re.compile(
        r"^(?:experience|skills|qualifications|requirements|responsibilities|about|summary|education)",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:min|max|minimum)\s*(?:years|experience)", re.IGNORECASE),
    re.compile(r"^\d+\+?\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"^(?:required|preferred|nice.to.have)\s*(?:skills|qualifications)", re.IGNORECASE),
]

_SKILL_LIKE_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9+.#()]*(?:\s*,\s*[A-Z][A-Za-z0-9+.#()]*)+$")


def extract_responsibilities(text: str) -> list[str]:
    """Extract responsibility-like bullets from text.

    Looks for bullet points and numbered lines that describe work activities.
    Filters out section headers, metadata lines, and skill-only lines.
    """
    bullets: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[\s•\-*–—→⋅∙◦‣⁃▸▹►▪●○■□◆◇]+", "", line).strip()
        if not cleaned or len(cleaned) <= 15:
            continue
        if any(p.search(cleaned) for p in _NOISE_PATTERNS):
            continue
        if _SKILL_LIKE_PATTERN.match(cleaned):
            continue
        bullets.append(cleaned)
    if not bullets:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        bullets = [l for l in lines if len(l) > 15]
    return bullets


def _extract_experience_bullets(resume_text: str) -> list[str]:
    """Extract bullet points from Experience / Work History sections."""
    exp_section = _find_section_text(
        resume_text,
        r"(?:professional\s+)?(?:experience|work\s+history|employment|career)",
    )
    if exp_section:
        return extract_responsibilities(exp_section)
    return extract_responsibilities(resume_text)


def _find_section_text(text: str, header_pattern: str) -> str:
    """Find text belonging to a section matching header_pattern."""
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if re.search(header_pattern, line.strip(), re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped and re.match(r"^(?:#{1,2}\s+)?[A-Z][A-Za-z &/,\-–]+$", stripped):
            end = i
            break
    return "\n".join(lines[start:end])


def _tfidf_similarity(resume_bullets: list[str], jd_bullets: list[str]) -> float:
    """Compute TF-IDF cosine similarity between resume and JD bullets."""
    if not resume_bullets or not jd_bullets:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        corpus = jd_bullets + resume_bullets
        tfidf_matrix = vectorizer.fit_transform(corpus)
        jd_vecs = tfidf_matrix[: len(jd_bullets)]
        resume_vecs = tfidf_matrix[len(jd_bullets) :]
        sim_matrix = (jd_vecs @ resume_vecs.T).toarray()
        best_per_jd = sim_matrix.max(axis=1)
        return float(np.mean(best_per_jd))
    except Exception:
        return 0.0


def _semantic_bullet_similarity(resume_bullets: list[str], jd_bullets: list[str]) -> float:
    """Compute semantic similarity between resume and JD bullets."""
    if not resume_bullets or not jd_bullets:
        return 0.0
    scores: list[float] = []
    for jd_bullet in jd_bullets:
        best = 0.0
        for res_bullet in resume_bullets:
            sim = semantic_similarity(jd_bullet, res_bullet)
            if sim > best:
                best = sim
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def _action_verb_score(resume_bullets: list[str], jd_bullets: list[str]) -> float:
    """Score based on action verb overlap between resume and JD."""

    def _extract_verbs(texts: list[str]) -> set[str]:
        verbs: set[str] = set()
        for t in texts:
            for word in t.lower().split():
                if word in _ACTION_VERBS:
                    verbs.add(word)
        return verbs

    resume_verbs = _extract_verbs(resume_bullets)
    jd_verbs = _extract_verbs(jd_bullets)
    if not jd_verbs:
        return 1.0
    overlap = len(resume_verbs & jd_verbs)
    total = len(jd_verbs)
    return overlap / total if total > 0 else 0.0


def _section_confidence(resume_text: str) -> float:
    """Score confidence based on structure — well-structured resumes score higher."""
    score = 0.5
    has_exp_header = bool(
        re.search(r"(?:professional\s+)?(?:experience|work\s+history)", resume_text, re.IGNORECASE)
    )
    if has_exp_header:
        score += 0.2
    bullets = extract_responsibilities(resume_text)
    verb_count = sum(1 for b in bullets for w in b.lower().split() if w in _ACTION_VERBS)
    if len(bullets) >= 3:
        score += 0.15
    if verb_count >= 3:
        score += 0.15
    return min(1.0, score)


def _prevent_false_positives(
    resume_bullets: list[str], jd_bullets: list[str], semantic_score: float
) -> float:
    """Reduce semantic score when there's keyword-stuffing risk."""
    if not resume_bullets or not jd_bullets:
        return semantic_score
    jd_words = set(" ".join(jd_bullets).lower().split())
    keyword_density: list[float] = []
    for bullet in resume_bullets:
        words = bullet.lower().split()
        if not words:
            continue
        match_count = sum(1 for w in words if w in jd_words)
        keyword_density.append(match_count / len(words))
    avg_density = float(np.mean(keyword_density)) if keyword_density else 0.0
    stuffing_penalty = max(0.0, (avg_density - 0.6) * 2.0)
    return max(0.0, semantic_score * (1.0 - stuffing_penalty * 0.3))


def calculate_roles_score(jd_text: str, resume_text: str) -> dict[str, Any]:
    """Calculate Roles & Responsibilities match score.

    Uses a hybrid approach:
      - Semantic (75%)
      - Action verb overlap (15%)
      - Section confidence (10%)

    Returns a dict with score, sub-scores, and best role matches for explainability.
    """
    jd_bullets = extract_responsibilities(jd_text)
    resume_bullets = _extract_experience_bullets(resume_text)
    if not resume_bullets:
        resume_bullets = extract_responsibilities(resume_text)
    if not jd_bullets or not resume_bullets:
        return {
            "roles_score": 0.0,
            "tfidf_score": 0.0,
            "semantic_score": 0.0,
            "action_verb_score": 0.0,
            "section_confidence": _section_confidence(resume_text) * 100,
            "best_role_matches": [],
        }

    raw_semantic = _semantic_bullet_similarity(resume_bullets, jd_bullets)
    raw_semantic = _prevent_false_positives(resume_bullets, jd_bullets, raw_semantic)
    semantic_score = min(100.0, raw_semantic * 150.0)

    raw_verb = _action_verb_score(resume_bullets, jd_bullets)
    verb_score = min(100.0, raw_verb * 100.0)

    raw_confidence = _section_confidence(resume_text)
    confidence_score = min(100.0, raw_confidence * 100.0)

    roles_score = semantic_score * 0.75 + verb_score * 0.15 + confidence_score * 0.10
    roles_score = max(0.0, min(100.0, roles_score))

    best_role_matches: list[dict[str, Any]] = []
    for jb in jd_bullets[:5]:
        best_match = ""
        best_sim = 0.0
        for rb in resume_bullets:
            sim = semantic_similarity(jb, rb)
            if sim > best_sim:
                best_sim = sim
                best_match = rb
        if best_match:
            best_role_matches.append(
                {
                    "jd_responsibility": jb[:120],
                    "resume_match": best_match[:120],
                    "score": round(min(100.0, best_sim * 150.0), 1),
                }
            )

    return {
        "roles_score": round(roles_score, 1),
        "tfidf_score": 0.0,
        "semantic_score": round(semantic_score, 1),
        "action_verb_score": round(verb_score, 1),
        "section_confidence": round(confidence_score, 1),
        "best_role_matches": best_role_matches,
    }
