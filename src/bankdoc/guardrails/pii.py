"""PII (personally identifiable information) redaction.

Detects emails, phone numbers, Canadian SINs (Social Insurance Numbers) and
credit card numbers, and replaces each with a placeholder such as [EMAIL].

Numbers with a check digit (cards, SINs) are validated with the Luhn
algorithm before redaction, so an arbitrary 9- or 16-digit number in a
question is not mistaken for PII.

Limits: regex detection finds well-formatted PII only. It does not detect
names or addresses (that needs an NER model, e.g. Microsoft Presidio).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class RedactionResult:
    """Redacted text plus a count of what was found, per PII type."""

    text: str
    findings: dict[str, int] = field(default_factory=dict)


# \d(?:[ -]?\d){12,18} = 13 to 19 digits, optionally separated by single
# spaces or hyphens. (?<!\d) and (?!\d) stop us matching inside longer numbers.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
_SIN = re.compile(r"(?<!\d)\d{3}[ -]?\d{3}[ -]?\d{3}(?!\d)")
_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\d)"
)


def luhn_valid(digits: str) -> bool:
    """Check a digit string with the Luhn algorithm.

    Walking from the rightmost digit, every second digit is doubled (and 9 is
    subtracted if the result exceeds 9). The number is valid if the total is
    divisible by 10. Credit cards and Canadian SINs both use this checksum.
    """
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _replace(
    pattern: re.Pattern,
    label: str,
    text: str,
    findings: dict[str, int],
    validator: Callable[[str], bool] | None = None,
) -> str:
    """Replace every match of pattern with [LABEL], counting replacements.

    If a validator is given, a match is replaced only when the validator
    accepts its digits; otherwise it is left untouched.
    """

    def substitute(match: re.Match) -> str:
        value = match.group(0)
        if validator is not None and not validator(re.sub(r"\D", "", value)):
            return value
        findings[label] = findings.get(label, 0) + 1
        return f"[{label}]"

    return pattern.sub(substitute, text)


def redact_pii(text: str) -> RedactionResult:
    """Redact PII from text.

    Order matters: cards (13-19 digits) before SINs (9 digits) before phone
    numbers (10 digits), so that each number is classified by its most
    specific pattern.
    """
    findings: dict[str, int] = {}
    text = _replace(_EMAIL, "EMAIL", text, findings)
    text = _replace(_CARD, "CREDIT_CARD", text, findings,
                    lambda d: 13 <= len(d) <= 19 and luhn_valid(d))
    text = _replace(_SIN, "SIN", text, findings,
                    lambda d: len(d) == 9 and luhn_valid(d))
    text = _replace(_PHONE, "PHONE", text, findings)
    return RedactionResult(text=text, findings=findings)


# Matches the placeholders produced by redact_pii().
_PLACEHOLDER = re.compile(r"\[(?:EMAIL|CREDIT_CARD|SIN|PHONE)\]")


def strip_placeholders(text: str) -> str:
    """Remove redaction placeholders and tidy the whitespace.

    Used to build the SEARCH query. A token like [EMAIL] has no meaning for
    semantic search and measurably lowers similarity scores, so the search
    sees the question without it. The LLM still receives the full redacted
    question, and the audit log stores the redacted version.
    """
    without_tokens = _PLACEHOLDER.sub(" ", text)
    return re.sub(r"\s+", " ", without_tokens).strip()