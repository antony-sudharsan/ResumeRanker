"""Semantic similarity matching using sentence-transformers.

Provides meaning-based matching between job descriptions and resumes,
going beyond exact keyword matching to understand related concepts.
"""

from __future__ import annotations

import functools
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@functools.lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the sentence-transformer model (cached)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


# Optional cross-encoder cache
_cross_encoder = None


def _get_cross_encoder():
    """Load the cross-encoder model (cached, optional). Returns None if unavailable."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder = CrossEncoder("cross-encoder/stsb-MiniLM-L-6-v2")
        except Exception:
            _cross_encoder = False
    return _cross_encoder if _cross_encoder is not False else None


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity between two texts.

    Returns a score between 0.0 and 1.0.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    model = _get_model()
    embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
    sim = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
    return float(max(0.0, min(1.0, sim)))


def semantic_similarity_chunked(
    jd_text: str,
    resume_text: str,
    jd_weights: list[float] | None = None,
) -> float:
    """Compute semantic similarity using section-level chunking.

    Splits both texts into meaningful chunks and computes an aggregate
    similarity score, which is more accurate for longer documents.

    The optional jd_weights parameter allows weighting certain JD chunks
    higher (e.g., requirements sections).
    """
    if not jd_text.strip() or not resume_text.strip():
        return 0.0

    jd_chunks = _split_into_chunks(jd_text)
    resume_chunks = _split_into_chunks(resume_text)

    if not jd_chunks or not resume_chunks:
        return semantic_similarity(jd_text, resume_text)

    model = _get_model()
    jd_embeddings = model.encode(jd_chunks, convert_to_numpy=True)
    resume_embeddings = model.encode(resume_chunks, convert_to_numpy=True)

    sim_matrix = cosine_similarity(jd_embeddings, resume_embeddings)

    # For each JD chunk, take the best matching resume chunk
    best_matches = np.max(sim_matrix, axis=1)

    if jd_weights and len(jd_weights) == len(best_matches):
        w = np.array(jd_weights, dtype=float)
        w = w / w.sum()
        score = float(np.average(best_matches, weights=w))
    else:
        score = float(np.mean(best_matches))

    return max(0.0, min(1.0, score))


def cross_encoder_score(jd_text: str, resume_text: str) -> float | None:
    """Score using cross-encoder for more accurate pairwise similarity.

    Uses a cross-encoder (stsb-MiniLM-L-6-v2) for pairwise re-ranking,
    which tends to be more accurate than bi-encoder cosine similarity.

    Returns None if the cross-encoder model is not available.
    """
    if not jd_text.strip() or not resume_text.strip():
        return None

    model = _get_cross_encoder()
    if model is None:
        return None

    pairs = [(jd_text, resume_text)]
    score = model.predict(pairs)[0]
    # STS cross-encoders output 0-5; normalize to 0-1
    normalized = float(score) / 5.0
    return max(0.0, min(1.0, normalized))


def _split_into_chunks(text: str, min_length: int = 30) -> list[str]:
    """Split text into meaningful chunks.

    Multi-level strategy:
      1. Split on blank lines (paragraphs)
      2. Split on bullet points within paragraphs
      3. Sliding window (3 sentences, 1 overlap) for chunks >800 chars
      4. Fallback: sentence-level grouping if no paragraph structure
    """
    lines = text.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    bullet_re = re.compile(r"^\s*[•\-*]\s*")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            _flush_chunk(current_chunk, chunks, min_length)
            continue

        # Bullet points act as chunk separators
        if bullet_re.match(line) and current_chunk:
            _flush_chunk(current_chunk, chunks, min_length)

        current_chunk.append(stripped)

    _flush_chunk(current_chunk, chunks, min_length)

    # Sliding window for long chunks (exceeds ~200 word / 256 token limit)
    windowed: list[str] = []
    for chunk in chunks:
        if len(chunk) <= 800:
            windowed.append(chunk)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            if len(sentences) <= 2:
                windowed.append(chunk)
            else:
                for i in range(0, len(sentences), 2):
                    window = sentences[i : i + 3]
                    if window:
                        windowed.append(" ".join(window))
    chunks = windowed

    # Fallback: sentence-level grouping
    if not chunks:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current: list[str] = []
        for sent in sentences:
            current.append(sent)
            combined = " ".join(current)
            if len(combined) >= min_length:
                chunks.append(combined)
                current = []
        if current:
            chunks.append(" ".join(current))

    return chunks if chunks else [text[:500]]


def _flush_chunk(
    current_chunk: list[str],
    chunks: list[str],
    min_length: int,
) -> None:
    """Flush the current chunk buffer into chunks if it meets min_length."""
    if not current_chunk:
        return
    chunk_text = " ".join(current_chunk)
    if len(chunk_text) >= min_length:
        chunks.append(chunk_text)
    current_chunk.clear()
