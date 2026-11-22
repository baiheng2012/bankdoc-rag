"""Checks applied to the model's answer before it is returned.

1. Numeric grounding: every number in the answer must also appear in the
   chunks the answer cites. This catches wrong years and invented figures.
   It does NOT catch semantic errors (e.g. attributing a growth statement to
   inflation); that needs faithfulness evaluation (Step 6).
2. System-prompt leakage: the answer must not repeat our instructions.
"""

import re

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CITATION_MARKER = re.compile(r"\[\d+\]")
_FRACTIONS = {"½": ".5", "¼": ".25", "¾": ".75"}


def _normalise(text: str) -> str:
    """Turn '2½' into '2.5' so a source written with a fraction glyph can
    match an answer written with decimals. Bank of Canada reports use these."""
    for glyph, decimal in _FRACTIONS.items():
        text = re.sub(rf"(\d){glyph}", rf"\g<1>{decimal}", text)
    return text


def extract_numbers(text: str) -> set[str]:
    """Return the distinct numbers in text as strings ('1,000' -> '1000').

    Citation markers like [2] are removed first, because they are
    references, not claims.
    """
    cleaned = _CITATION_MARKER.sub(" ", _normalise(text))
    return {token.replace(",", "") for token in _NUMBER.findall(cleaned)}


def find_ungrounded_numbers(answer: str, cited_chunks: list) -> list[str]:
    """List numbers in the answer that appear in none of the cited chunks.

    The allowed set per chunk is its text PLUS its file name and page number.
    The file name matters because the model is told each report's date (taken
    from the file name) and may legitimately write "the October 2026 report".

    Args:
        answer: The model's answer.
        cited_chunks: RetrievedChunk objects the answer actually cites.

    Returns:
        Sorted list of ungrounded numbers; empty means the answer is grounded.
    """
    allowed: set[str] = set()
    for chunk in cited_chunks:
        allowed |= extract_numbers(f"{chunk.content} {chunk.source_uri} {chunk.page_number}")
    return sorted(extract_numbers(answer) - allowed)


def leaks_system_prompt(answer: str, system_prompt: str, shingle_words: int = 8) -> bool:
    """Detect verbatim copying of the system prompt into the answer.

    A "shingle" is a run of N consecutive words. If any N-word run of the
    answer also occurs in the system prompt, the answer is quoting our
    instructions. N=8 is long enough that ordinary prose rarely collides.
    """
    prompt_words = system_prompt.lower().split()
    shingles = {
        " ".join(prompt_words[i : i + shingle_words])
        for i in range(len(prompt_words) - shingle_words + 1)
    }
    answer_words = answer.lower().split()
    return any(
        " ".join(answer_words[i : i + shingle_words]) in shingles
        for i in range(len(answer_words) - shingle_words + 1)
    )
