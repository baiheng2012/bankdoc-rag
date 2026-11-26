# ADR 0001: Retrieval threshold MIN_SCORE = 0.65

Status: accepted

## Context
Chunks below a similarity threshold are not sent to the LLM; if none pass, the system refuses without calling the model. The first value (0.55) was chosen from two examples (one relevant question, one cookie recipe).

## Decision
Calibrate on the evaluation dataset with `scripts/calibrate_threshold.py`: for each candidate threshold, count answerable questions that still reach the LLM and refusable questions that still leak through (19 answerable, 9 refusable).

| Threshold | Answerable kept | Refusable leaked |
|---|---|---|
| 0.55 | 19/19 | 5/9 |
| 0.65 | 19/19 | 1/9 |
| 0.70 | 17/19 | 1/9 |
| 0.725 | 15/19 | 0/9 |

Chosen: 0.65. The lowest-scoring answerable question was 0.693, so the margin is about 0.04.

## Consequences
- Off-topic questions (cookies, hockey, gibberish, a private address) are now stopped deterministically by the threshold instead of relying on the model to decline.
- A gibberish question scored 0.577, above the old threshold, so 0.55 was demonstrably too low.
- "What will the policy rate be in 2030?" scores 0.709: on-topic but beyond every projection horizon. No threshold separates it; only the LLM and the output guardrails handle it.
- Vague questions had to be added to the dataset, otherwise the threshold would have been tuned on questions that name the report explicitly and therefore score high.
- The dataset is small and was used for tuning, so 0.65 is a reasonable starting point, not a proven optimum. Re-run the calibration whenever documents or the embedding model change.
