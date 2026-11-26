# ADR 0004: Deterministic guardrails instead of prompt rules

Status: accepted

## Context
After the numeric-grounding guardrail blocked the self-computed 1.5, we tried to prevent the behaviour at the source with a new prompt rule: never calculate, add or estimate numbers.

## Evidence
Evaluated against the whole dataset with qwen2.5:7b:
- The targeted case produced a byte-identical withheld answer: the rule did not change the behaviour.
- `cpi-feb-apr` regressed: the model stopped reporting a figure that was present in the source.
- Removing the rule restored `cpi-feb-apr`.

## Decision
Remove the rule. Behavioural constraints live in code (guardrails and validators), and every prompt change must pass the full evaluation before it is kept.

## Consequences
- Attack, benign, PII and unanswerable categories pass identically on a 7B and a 3B model, because they depend on deterministic code, not model obedience.
- Only answer quality varies by model: 35 pass / 0 fail on 7B versus 31 pass / 4 fail on 3B (same dataset and settings).
