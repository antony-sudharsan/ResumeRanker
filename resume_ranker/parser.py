"""Resume and job description text extraction from various file formats."""

import io
from pathlib import Path

import docx
import pdfplumber


def _get_column_bboxes(page):
    """Detect multi-column layout on a page using word x-coordinate distribution.

    Returns a list of bounding boxes [(x0, top, x1, bottom), ...] for each
    column, or None if the page appears to be single-column.
    """
    words = page.extract_words()
    if not words or page.width <= 0:
        return None

    num_bins = 50
    bin_width = page.width / num_bins
    bins = [0] * num_bins

    for w in words:
        mid_x = (w["x0"] + w["x1"]) / 2
        idx = min(int(mid_x / bin_width), num_bins - 1)
        bins[idx] += 1

    threshold = max(1, len(words) * 0.005)
    search_start = int(num_bins * 0.15)
    search_end = int(num_bins * 0.85)

    best_gap_start = None
    best_gap_len = 0
    cur_start = None

    for i in range(search_start, search_end):
        if bins[i] < threshold:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                gap_len = i - cur_start
                if gap_len > best_gap_len:
                    best_gap_len = gap_len
                    best_gap_start = cur_start
                cur_start = None

    if best_gap_len >= 3 and best_gap_start is not None:
        split_x = (best_gap_start + best_gap_start + best_gap_len) / 2 * bin_width
        return [
            (0, 0, split_x, page.height),
            (split_x, 0, page.width, page.height),
        ]
    return None


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text content from a PDF file with multi-column support."""
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            col_bboxes = _get_column_bboxes(page)
            if col_bboxes:
                for bbox in col_bboxes:
                    cropped = page.within_bbox(bbox)
                    if cropped:
                        col_text = cropped.extract_text()
                        if col_text:
                            text_parts.append(col_text)
            else:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text content from a DOCX file."""
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text content from a plain text file."""
    return file_bytes.decode("utf-8", errors="replace")


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract text from a file based on its extension."""
    suffix = Path(filename).suffix.lower()
    extractors = {
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
        ".txt": extract_text_from_txt,
    }
    extractor = extractors.get(suffix)
    if extractor is None:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .pdf, .docx, .txt")
    return extractor(file_bytes)
