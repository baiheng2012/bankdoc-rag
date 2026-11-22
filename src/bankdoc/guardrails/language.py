"""Script check (not a language detector).

The embedding model (bge-small-en) and the documents are English. Questions
in a non-Latin script (Chinese, Arabic, Cyrillic, ...) cannot be searched
reliably, and the English-only injection rules cannot inspect them. We
refuse them explicitly instead of letting them reach the LLM by accident.

LIMIT: this tests the SCRIPT, not the language. French or Spanish text uses
the Latin alphabet and passes. It is a cheap filter, not a security control.
"""

# Code points up to U+024F cover Basic Latin, Latin-1 Supplement (e, e, c, a
# with accents ...) and Latin Extended-A/B, so French accents count as Latin.
_LATIN_MAX_CODEPOINT = 0x024F

# Refuse when more than this share of the letters is non-Latin.
NON_LATIN_RATIO_LIMIT = 0.3


def non_latin_ratio(text: str) -> float:
    """Return the fraction of alphabetic characters outside the Latin range.

    Digits, spaces and punctuation are ignored, so "Inflation was 2.5%" is
    judged only by its letters. Text with no letters returns 0.0.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    non_latin = sum(1 for ch in letters if ord(ch) > _LATIN_MAX_CODEPOINT)
    return non_latin / len(letters)


def is_supported_script(text: str) -> bool:
    """True if the text is (mostly) written in the Latin alphabet."""
    return non_latin_ratio(text) <= NON_LATIN_RATIO_LIMIT
