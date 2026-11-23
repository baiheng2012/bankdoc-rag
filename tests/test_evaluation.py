"""Tests for the evaluation framework itself (checks, runner, loader).

If the yardstick is wrong, every measurement is wrong, so the yardstick is
tested too. No model, database or network is used.
"""

from pathlib import Path

import pytest

from bankdoc.evaluation.cases import CATEGORIES, EvalCase, load_cases
from bankdoc.evaluation.checks import evaluate_case
from bankdoc.evaluation.runner import run_case, summarize
from bankdoc.generation.answerer import AnswerResult
from bankdoc.retrieval.retriever import RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]


def make_chunk(uri="mpr-2026-10-28.pdf", title="Monetary Policy Report") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=1, title=title, source_uri=uri, page_number=23,
                          content="c", score=0.8)


def make_result(answer="Inflation is 3% [1].", refused=False, reason=None,
                citations=None, redacted="") -> AnswerResult:
    return AnswerResult(
        answer=answer, refused=refused, refusal_reason=reason,
        citations=[make_chunk()] if citations is None else citations,
        redacted_question=redacted,
    )


def check(expect: dict, result: AnswerResult) -> dict[str, bool]:
    """Evaluate one expectation block and return {check name: passed}."""
    case = EvalCase("c", "answerable", "q", expect)
    return {c.name: c.passed for c in evaluate_case(case, result)}


# ---------------------------------------------------------------- checks

def test_refused_check():
    assert check({"refused": False}, make_result()) == {"refused": True}
    assert check({"refused": True}, make_result()) == {"refused": False}


def test_reason_in_check():
    result = make_result(refused=True, reason="prompt_injection")
    assert check({"reason_in": ["prompt_injection"]}, result)["reason_in"]
    assert not check({"reason_in": ["unsupported_language"]}, result)["reason_in"]


def test_not_guardrail_blocked_only_counts_security_reasons():
    security = make_result(refused=True, reason="prompt_injection")
    no_data = make_result(refused=True, reason="no_relevant_sources")
    assert not check({"not_guardrail_blocked": True}, security)["not_guardrail_blocked"]
    assert check({"not_guardrail_blocked": True}, no_data)["not_guardrail_blocked"]
    assert check({"not_guardrail_blocked": True}, make_result())["not_guardrail_blocked"]


def test_cites_any_matches_file_name_or_title():
    result = make_result()
    assert check({"cites_any": ["mpr-2026-10-28"]}, result)["cites_any"]
    assert check({"cites_any": ["monetary policy report"]}, result)["cites_any"]
    assert not check({"cites_any": ["mpr-2025-10-29"]}, result)["cites_any"]


def test_forbids_cites():
    result = make_result()
    assert not check({"forbids_cites": ["mpr-2026-10-28"]}, result)["forbids_cites"]
    assert check({"forbids_cites": ["mpr-2025-10-29"]}, result)["forbids_cites"]


def test_facts_need_every_group_but_any_alternative():
    answer = make_result(answer="Inflation is 3% in 2028 [1].")
    assert check({"facts": [["3%"], ["2027", "2028"]]}, answer)["facts"]
    assert not check({"facts": [["3%"], ["2030"]]}, answer)["facts"]


def test_forbidden_text_is_case_insensitive():
    result = make_result(answer="It is Higher than anticipated in the July Report [1].")
    assert not check({"forbidden_text": ["than anticipated in the july"]}, result)["forbidden_text"]
    assert check({"forbidden_text": ["something else"]}, result)["forbidden_text"]


def test_no_pii_checks_both_stored_question_and_answer():
    clean = make_result(answer="ok [1].", redacted="My email is [EMAIL]")
    leak_answer = make_result(answer="Mail a@b.com [1].", redacted="[EMAIL]")
    leak_question = make_result(answer="ok [1].", redacted="My email is a@b.com")
    expect = {"no_pii_in_output": ["a@b.com"]}
    assert check(expect, clean)["no_pii_in_output"]
    assert not check(expect, leak_answer)["no_pii_in_output"]
    assert not check(expect, leak_question)["no_pii_in_output"]


# ---------------------------------------------------------------- runner

def test_run_case_statuses():
    result = make_result()
    fake = lambda question, clearance: result  # noqa: E731

    passing = EvalCase("c", "answerable", "q", {"refused": False})
    failing = EvalCase("c", "answerable", "q", {"refused": True})
    known_failing = EvalCase("c", "answerable", "q", {"refused": True}, known_issue="documented")
    known_passing = EvalCase("c", "answerable", "q", {"refused": False}, known_issue="documented")

    assert run_case(passing, fake).status == "PASS"
    assert run_case(failing, fake).status == "FAIL"
    assert run_case(known_failing, fake).status == "KNOWN"
    assert run_case(known_passing, fake).status == "FIXED"


def test_run_case_records_exception_as_failure():
    def boom(question, clearance):
        raise RuntimeError("model crashed")

    outcome = run_case(EvalCase("c", "answerable", "q", {"refused": False}), boom)
    assert outcome.status == "FAIL"
    assert outcome.checks[0].name == "no_exception"
    assert "model crashed" in outcome.error


def test_summarize_counts_by_category_and_status():
    fake = lambda question, clearance: make_result()  # noqa: E731
    outcomes = [
        run_case(EvalCase("a", "answerable", "q", {"refused": False}), fake),                      # PASS
        run_case(EvalCase("b", "answerable", "q", {"refused": True}, known_issue="x"), fake),      # KNOWN
        run_case(EvalCase("c", "attack", "q", {"refused": True}), fake),                           # FAIL
    ]
    by_category, totals = summarize(outcomes)
    assert by_category["answerable"]["PASS"] == 1 and by_category["answerable"]["KNOWN"] == 1
    assert by_category["attack"]["FAIL"] == 1
    assert totals["FAIL"] == 1  # a FAIL anywhere means the quality gate fails


# ---------------------------------------------------------------- loader

def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cases.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loader_rejects_unknown_expect_key(tmp_path):
    path = write(tmp_path, '- {id: a, category: answerable, question: "q?", expect: {fact: [["x"]]}}')
    with pytest.raises(ValueError, match="unknown expect keys"):
        load_cases(path)


def test_loader_rejects_duplicate_ids(tmp_path):
    case = '- {id: a, category: answerable, question: "q?", expect: {refused: false}}\n'
    with pytest.raises(ValueError, match="duplicate id"):
        load_cases(write(tmp_path, case * 2))


def test_loader_rejects_unknown_category(tmp_path):
    path = write(tmp_path, '- {id: a, category: nonsense, question: "q?", expect: {refused: false}}')
    with pytest.raises(ValueError, match="unknown category"):
        load_cases(path)


def test_shipped_dataset_is_valid_and_covers_every_category():
    cases = load_cases(ROOT / "eval" / "cases.yaml")
    assert len(cases) >= 25
    assert {c.category for c in cases} == set(CATEGORIES)

def test_refused_detail_mentions_ungrounded_numbers():
    result = make_result(refused=True, reason="ungrounded_numbers")
    result.ungrounded_numbers = ["9.9"]
    case = EvalCase("c", "answerable", "q", {"refused": False})
    assert "9.9" in evaluate_case(case, result)[0].detail