"""Evaluation runner: executes cases, derives a status, aggregates results."""

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bankdoc.core.config import get_settings
from bankdoc.evaluation.cases import EvalCase
from bankdoc.evaluation.checks import CheckResult, evaluate_case
from bankdoc.generation.answerer import answer_question

STATUSES = ("PASS", "FAIL", "KNOWN", "FIXED")


@dataclass
class CaseOutcome:
    """Everything recorded about one executed case."""

    case: EvalCase
    status: str  # PASS | FAIL | KNOWN (known issue still failing) | FIXED (known issue now passing)
    checks: list[CheckResult]
    answer: str = ""
    refused: bool = False
    refusal_reason: str | None = None
    cited: list[str] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None
    withheld: str = ""  # raw answer a guardrail blocked (debug only)


def run_case(case: EvalCase, answer_fn=answer_question) -> CaseOutcome:
    """Run one case and judge it.

    Args:
        case: The case to run.
        answer_fn: Function (question, clearance) -> AnswerResult. Injected so
            tests can run the runner without a model or database.

    An exception inside the system under test is recorded as a failed
    "no_exception" check instead of aborting the whole evaluation run.
    """
    started = time.perf_counter()
    result = None
    error = None
    try:
        result = answer_fn(case.question, clearance=case.clearance)
        checks = evaluate_case(case, result)
    except Exception as exc:  # noqa: BLE001 - a crash is itself an evaluation finding
        error = f"{type(exc).__name__}: {exc}"
        checks = [CheckResult("no_exception", False, error)]

    passed = all(check.passed for check in checks)
    if case.known_issue:
        status = "FIXED" if passed else "KNOWN"
    else:
        status = "PASS" if passed else "FAIL"

    return CaseOutcome(
        case=case,
        status=status,
        checks=checks,
        answer=result.answer if result else "",
        refused=result.refused if result else False,
        refusal_reason=result.refusal_reason if result else None,
        cited=[f"{c.source_uri} p{c.page_number}" for c in result.citations] if result else [],
        seconds=time.perf_counter() - started,
        error=error,
        withheld=result.withheld_answer if result else "",        
    )


def summarize(outcomes: list[CaseOutcome]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Count statuses per category and overall.

    Returns:
        (by_category, totals), each mapping status name -> count.
    """
    by_category: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        row = by_category.setdefault(outcome.case.category, {s: 0 for s in STATUSES})
        row[outcome.status] += 1
    totals = {s: sum(row[s] for row in by_category.values()) for s in STATUSES}
    return by_category, totals


def pass_rate(row: dict[str, int]) -> float | None:
    """Pass rate of a status-count row. KNOWN cases are excluded (they are
    documented, not counted against the rate). Returns None if nothing counts."""
    passed = row["PASS"] + row["FIXED"]
    counted = passed + row["FAIL"]
    return passed / counted if counted else None


def run_metadata() -> dict:
    """Conditions of this run, stored in the report so results are comparable."""
    settings = get_settings()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - not a git checkout, or git missing
        commit = "unknown"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "min_score": settings.min_score,
    }


def build_report(outcomes: list[CaseOutcome], meta: dict) -> dict:
    """Assemble the JSON-serialisable report."""
    by_category, totals = summarize(outcomes)
    return {
        "meta": meta,
        "totals": totals,
        "by_category": by_category,
        "cases": [
            {
                "id": o.case.id,
                "category": o.case.category,
                "question": o.case.question,
                "status": o.status,
                "known_issue": o.case.known_issue,
                "answer": o.answer,
                "refused": o.refused,
                "refusal_reason": o.refusal_reason,
                "cited": o.cited,
                "seconds": round(o.seconds, 2),
                "withheld_answer": o.withheld,                
                "failed_checks": [
                    {"name": c.name, "detail": c.detail} for c in o.checks if not c.passed
                ],
            }
            for o in outcomes
        ],
    }
