"""Prompt-injection detection (heuristic, rule based).

Prompt injection = text that tries to make the model follow the attacker's
instructions instead of ours. Two variants:
  - DIRECT: the user types it into the question box.
  - INDIRECT: it hides inside a document that retrieval later pulls in.
We run the same detector on both the question and every retrieved chunk.

IMPORTANT LIMITATION: this is pattern matching. A paraphrase, a typo trick
or another language can slip past it. Treat it as one cheap layer, not as
the defence. The real protection is architectural: the model has no tools
and no write access, document permissions are enforced in SQL, and the
output is validated before it is returned.
"""

import re

_FLAGS = re.IGNORECASE | re.MULTILINE

# rule name -> compiled pattern
_RULES: dict[str, re.Pattern] = {
    # "ignore all previous instructions", "disregard your rules"
    "ignore_previous": re.compile(
        r"\b(ignore|disregard|forget)\b\W+(?:\w+\W+){0,2}?"
        r"(previous|prior|above|earlier|preceding|your|system)\W+(?:\w+\W+){0,2}?"
        r"(instructions?|rules?|prompts?|guidelines?|context)\b",
        _FLAGS,
    ),
    "ignore_above": re.compile(
        r"\b(ignore|disregard)\b\W+(?:everything|all)\W+(?:above|before|previous)\b", _FLAGS
    ),
    # attempts to extract our system prompt
    "system_prompt_probe": re.compile(
        r"\b(system|hidden|initial|original)\s+(prompt|instructions)\b"
        r"|\byour\s+(instructions|prompt|programming)\b",
        _FLAGS,
    ),
    # attempts to give the model a new identity
    "role_hijack": re.compile(
        r"\byou\s+are\s+now\b|\bfrom\s+now\s+on\s+you\b|\bpretend\s+(to\s+be|you\s+are)\b"
        r"|\bact\s+as\s+(if\s+you\s+(have|had)\s+no|an?\s+unrestricted)\b",
        _FLAGS,
    ),
    "jailbreak_terms": re.compile(
        r"\bjailbreak(?:ed)?\b|\bDAN\s+mode\b|\bdeveloper\s+mode\b|\bdo\s+anything\s+now\b",
        _FLAGS,
    ),
    # fake chat-template markers, e.g. a line starting with "SYSTEM:"
    "fake_role_markers": re.compile(
        r"^\s*(system|assistant)\s*:|</?\s*(system|assistant)\s*>|\[\s*/?INST\s*\]|<\|im_(start|end)\|>",
        _FLAGS,
    ),
    "new_instructions": re.compile(r"\bnew\s+instructions?\s*:", _FLAGS),
}


def detect_injection(text: str) -> list[str]:
    """Return the names of all rules that match the text (empty list = clean)."""
    return [name for name, pattern in _RULES.items() if pattern.search(text)]
