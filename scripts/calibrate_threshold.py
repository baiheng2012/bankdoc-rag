"""Calibrate MIN_SCORE from the evaluation dataset (retrieval only, no LLM).

For every case  take the TOP retrieval score. A question reaches the LLM
only if its top score >= MIN_SCORE, so for each candidate threshold we can
count:
  - how many questions that SHOULD be answered still reach the LLM (want 100%)
  - how many questions that should NOT be answered reach the LLM   (want low)

Usage:
    python scripts/calibrate_threshold.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankdoc.core.db import close_pool  # noqa: E402
from bankdoc.evaluation.cases import load_cases  # noqa: E402
from bankdoc.guardrails.injection import detect_injection  # noqa: E402
from bankdoc.guardrails.language import is_supported_script  # noqa: E402
from bankdoc.guardrails.pii import redact_pii, strip_placeholders  # noqa: E402
from bankdoc.retrieval.retriever import retrieve  # noqa: E402

THRESHOLDS = [round(0.50 + 0.025 * i, 3) for i in range(11)]  # 0.50 ... 0.75


def blocked_before_retrieval(question: str) -> bool:
    """True if a guardrail stops this question before retrieval ever runs.
    Such questions never reach the threshold, so they say nothing about it."""
    safe = redact_pii(question).text
    return bool(detect_injection(safe)) or not is_supported_script(safe)


def top_score(question: str, clearance: int) -> float:
    """Highest retrieval score, using the same query the real pipeline uses."""
    query = strip_placeholders(redact_pii(question).text)
    if len(query) < 3:
        return 0.0
    return max((c.score for c in retrieve(query, clearance)), default=0.0)


def main() -> None:
    cases = load_cases(Path("eval/cases.yaml"))
    positives, negatives = [], []  # (case id, top score)
    try:
        for case in cases:
            wanted = case.expect.get("refused")
            if wanted is None:
                continue  # no clear "should answer / should refuse" expectation
            if wanted is True and blocked_before_retrieval(case.question):
                continue
            (negatives if wanted else positives).append((case.id, top_score(case.question, case.clearance)))
    finally:
        close_pool()

    print("=== Should ANSWER (top retrieval score) ===")
    for case_id, score in sorted(positives, key=lambda x: x[1]):
        print(f"  {score:.3f}  {case_id}")
    print("\n=== Should REFUSE (top retrieval score) ===")
    for case_id, score in sorted(negatives, key=lambda x: -x[1]):
        print(f"  {score:.3f}  {case_id}")

    print(f"\n{'threshold':>10} {'answerable kept':>17} {'refusable leaked':>18}")
    for threshold in THRESHOLDS:
        kept = sum(1 for _, s in positives if s >= threshold)
        leaked = sum(1 for _, s in negatives if s >= threshold)
        print(f"{threshold:>10.3f} {kept:>10}/{len(positives):<6} {leaked:>11}/{len(negatives):<6}")
    print("\nPick the highest threshold that still keeps ALL answerable questions, "
          "then leave a safety margin. With few cases this is a rough guide, not a proof.")


if __name__ == "__main__":
    main()
