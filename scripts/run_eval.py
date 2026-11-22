"""Run the evaluation suite against the real pipeline.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --category attack benign
    python scripts/run_eval.py --id infl-latest
    LLM_MODEL=qwen2.5:3b python scripts/run_eval.py     # compare another model

Exit code: 0 if no case FAILED (KNOWN issues are allowed), 1 otherwise.
That exit code is what a CI quality gate will use in Step 7.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankdoc.core.db import close_pool  # noqa: E402
from bankdoc.evaluation.cases import load_cases  # noqa: E402
from bankdoc.evaluation.runner import (  # noqa: E402
    build_report, pass_rate, run_case, run_metadata, summarize,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation suite.")
    parser.add_argument("--cases", default="eval/cases.yaml")
    parser.add_argument("--category", nargs="+", help="Only run these categories")
    parser.add_argument("--id", nargs="+", help="Only run these case ids")
    parser.add_argument("--out", default="reports", help="Folder for the JSON report")
    args = parser.parse_args()

    # Guardrails log warnings when they fire; that is noise during an evaluation.
    logging.basicConfig(level=logging.ERROR)

    cases = load_cases(Path(args.cases))
    if args.category:
        cases = [c for c in cases if c.category in args.category]
    if args.id:
        cases = [c for c in cases if c.id in args.id]

    meta = run_metadata()
    print(f"Running {len(cases)} cases | model={meta['llm_model']} "
          f"min_score={meta['min_score']} top_k={meta['top_k']} commit={meta['git_commit']}\n")

    outcomes = []
    try:
        for case in cases:
            outcome = run_case(case)
            outcomes.append(outcome)
            print(f"[{outcome.status:5}] {case.id:24} {outcome.seconds:5.1f}s")
            if outcome.status in ("FAIL", "KNOWN"):
                for check in outcome.checks:
                    if not check.passed:
                        print(f"          - {check.name}: {check.detail}")
                print(f"          answer: {outcome.answer[:160]!r}")
    finally:
        close_pool()

    by_category, totals = summarize(outcomes)
    print("\n=== Summary by category ===")
    print(f"{'category':14}{'pass':>6}{'fail':>6}{'known':>7}{'fixed':>7}{'rate':>8}")
    for category, row in sorted(by_category.items()):
        rate = pass_rate(row)
        rate_text = f"{rate:.0%}" if rate is not None else "n/a"
        print(f"{category:14}{row['PASS']:>6}{row['FAIL']:>6}{row['KNOWN']:>7}{row['FIXED']:>7}{rate_text:>8}")
    print(f"\nTOTAL  PASS={totals['PASS']} FAIL={totals['FAIL']} "
          f"KNOWN={totals['KNOWN']} FIXED={totals['FIXED']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = meta["llm_model"].replace(":", "-").replace("/", "-")
    stamp = meta["timestamp"].replace(":", "").replace("-", "")[:15]
    report_path = out_dir / f"eval-{stamp}-{safe_model}.json"
    report_path.write_text(json.dumps(build_report(outcomes, meta), indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"Report written to {report_path}")

    return 1 if totals["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
