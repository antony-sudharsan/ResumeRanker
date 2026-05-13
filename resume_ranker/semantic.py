"""Semantic similarity matching using sentence-transformers.

Provides meaning-based matching between job descriptions and resumes,
going beyond exact keyword matching to understand related concepts.
"""

from __future__ import annotations

import functools

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@functools.lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the sentence-transformer model (cached)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


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


def semantic_similarity_chunked(jd_text: str, resume_text: str) -> float:
    """Compute semantic similarity using section-level chunking.

    Splits both texts into meaningful chunks and computes an aggregate
    similarity score, which is more accurate for longer documents.
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
    score = float(np.mean(best_matches))
    return max(0.0, min(1.0, score))


def _split_into_chunks(text: str, min_length: int = 30) -> list[str]:
    """Split text into meaningful chunks (sentences/bullet points)."""
    lines = text.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= min_length:
                    chunks.append(chunk_text)
                current_chunk = []
            continue
        current_chunk.append(stripped)

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) >= min_length:
            chunks.append(chunk_text)

    return chunks if chunks else [text[:500]]
