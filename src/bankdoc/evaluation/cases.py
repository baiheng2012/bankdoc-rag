"""Evaluation cases: data structure and YAML loader.

An evaluation case is a question plus machine-checkable expectations. The
cases live in eval/cases.yaml so they can be reviewed and extended without
touching Python code.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

CATEGORIES = ("answerable", "unanswerable", "attack", "benign", "pii")

# Every key allowed inside a case's "expect" block. Unknown keys raise an
# error: a typo such as "fact:" instead of "facts:" would otherwise be
# silently ignored, the check would never run, and the case would "pass".
EXPECT_KEYS = {
    "refused",                # bool: must (not) be refused
    "reason_in",              # list: refusal_reason must be one of these
    "not_guardrail_blocked",  # bool: must not be refused by a SECURITY guardrail
    "cites_any",              # list: at least one citation must match one of these strings
    "forbids_cites",          # list: no citation may match any of these strings
    "facts",                  # list of lists: each inner list = alternatives, one must appear
    "forbidden_text",         # list: none of these may appear in the answer
    "no_pii_in_output",       # list: none of these raw values may appear in answer or stored question
}


@dataclass
class EvalCase:
    """One evaluation case."""

    id: str
    category: str
    question: str
    expect: dict
    clearance: int = 0
    # A non-empty string marks a documented, known failure (like pytest xfail).
    # The case still runs; its failure is reported as KNOWN and does not
    # break the quality gate. If it starts passing it is reported as FIXED.
    known_issue: str | None = None


def load_cases(path: Path) -> list[EvalCase]:
    """Load and validate evaluation cases from a YAML file.

    Raises:
        ValueError: on a missing field, duplicate id, unknown category or
            unknown expectation key. Failing loudly here is deliberate: a
            malformed dataset must never produce a silently weaker evaluation.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        #raise ValueError(f"{path}: the top level must be a list of cases")
        raise TypeError(f"{path}: the top level must be a list of cases")
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(data, start=1):
        label = f"case #{position} ({item.get('id', '?')})"
        missing = [key for key in ("id", "category", "question", "expect") if key not in item]
        if missing:
            raise ValueError(f"{label}: missing fields {missing}")
        if item["id"] in seen_ids:
            raise ValueError(f"{label}: duplicate id")
        seen_ids.add(item["id"])
        if item["category"] not in CATEGORIES:
            raise ValueError(f"{label}: unknown category {item['category']!r}")
        if not isinstance(item["expect"], dict) or not item["expect"]:
            raise ValueError(f"{label}: 'expect' must be a non-empty mapping")
        unknown = set(item["expect"]) - EXPECT_KEYS
        if unknown:
            raise ValueError(f"{label}: unknown expect keys {sorted(unknown)}")

        cases.append(
            EvalCase(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                expect=item["expect"],
                clearance=int(item.get("clearance", 0)),
                known_issue=item.get("known_issue"),
            )
        )
    return cases
