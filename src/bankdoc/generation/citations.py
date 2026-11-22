"""Parsing and validating citation markers such as [1] or [2][3]."""

import re

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citation_numbers(answer: str) -> list[int]:
    """Return the distinct citation numbers in an answer, in order of appearance."""
    seen: list[int] = []
    for match in _CITATION_PATTERN.finditer(answer):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def split_valid_invalid(numbers: list[int], source_count: int) -> tuple[list[int], list[int]]:
    """Separate real citations from hallucinated ones.

    A citation is valid only if it points at one of the sources we actually
    provided (1..source_count). Models sometimes invent numbers like [9].

    Returns:
        (valid_numbers, invalid_numbers)
    """
    valid = [n for n in numbers if 1 <= n <= source_count]
    invalid = [n for n in numbers if not 1 <= n <= source_count]
    return valid, invalid
