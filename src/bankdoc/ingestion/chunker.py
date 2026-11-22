"""Text chunking.

Why chunk? Embedding models and LLM prompts work best on focused passages.
A whole 80-page report in one vector would match everything vaguely and
nothing precisely.

Strategy: sentence-aware splitting with overlap.
  - Sentences are never cut in half (unless a single sentence is huge).
  - The end of each chunk is repeated at the start of the next one, so
    context that straddles a boundary is not lost.
  - Chunks never cross page boundaries, so every chunk has exactly one
    page number for citations.
"""

import re
from dataclasses import dataclass

from bankdoc.ingestion.loader import PageText

# Chunks shorter than this are usually page numbers, headers or noise.
MIN_CHUNK_CHARS = 50


@dataclass
class Chunk:
    """One piece of text ready to be embedded and stored."""

    chunk_index: int  # position within the whole document (0-based)
    page_number: int  # source page, used for citations
    content: str


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple punctuation heuristic.

    A boundary is whitespace that follows '.', '!' or '?'. It is not
    perfect (abbreviations like "U.S." can trigger false splits) but it is
    good enough for retrieval and has no extra dependencies.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Cut an over-long 'sentence' (e.g. a table dump) into fixed-size pieces."""
    return [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]


def _overlap_tail(sentences: list[str], overlap_chars: int) -> list[str]:
    """Pick whole sentences from the END of a finished chunk to reuse as the
    start of the next chunk, up to overlap_chars characters in total."""
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > overlap_chars:
            break
        tail.insert(0, sentence)
        total += len(sentence) + 1
    return tail


def chunk_text(text: str, max_chars: int = 1000, overlap_chars: int = 150) -> list[str]:
    """Split one text (normally one page) into overlapping chunks.

    Args:
        text: The text to split.
        max_chars: Target maximum chunk length in characters.
        overlap_chars: How much of the previous chunk to repeat.

    Returns:
        A list of chunk strings. Empty input gives an empty list.
    """
    # Flatten into a list of sentences, hard-splitting any giant ones.
    sentences: list[str] = []
    for sentence in _split_sentences(text):
        if len(sentence) > max_chars:
            sentences.extend(_hard_split(sentence, max_chars))
        else:
            sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate_len = len(" ".join(current + [sentence]))
        if current and candidate_len > max_chars:
            # Current chunk is full: emit it and start the next one.
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap_chars)
            # If overlap + the new sentence is still too long, drop the overlap.
            if len(" ".join(current + [sentence])) > max_chars:
                current = []
        current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(
    pages: list[PageText], max_chars: int = 1000, overlap_chars: int = 150
) -> list[Chunk]:
    """Chunk every page of a document and number the chunks sequentially.

    Args:
        pages: Cleaned pages from loader.load_pdf().
        max_chars: Target maximum chunk length.
        overlap_chars: Overlap between consecutive chunks on the same page.

    Returns:
        Chunks in document order. Tiny noise chunks are discarded.
    """
    result: list[Chunk] = []
    for page in pages:
        for content in chunk_text(page.text, max_chars, overlap_chars):
            if len(content) < MIN_CHUNK_CHARS:
                continue
            result.append(
                Chunk(chunk_index=len(result), page_number=page.page_number, content=content)
            )
    return result
