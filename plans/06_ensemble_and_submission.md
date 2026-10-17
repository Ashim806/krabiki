# Stage 06 — Ensemble & submission

Notebook: [`06_ensemble_and_submission.ipynb`](../06_ensemble_and_submission.ipynb)

## What it does
Fuses the two score streams produced in 04 and 05, picks the fusion weight on sub-01..04, and writes the two required submission files for sub-05 and sub-06.

## Fusion
`S_fused = w · rank(S_enc) + (1 − w) · rank(S_dec)` where `rank(S)` is per-row rank-normalisation into `[0, 1]` (higher = more likely). Rank fusion is scale-free and matches what the metric itself computes.

`w` is a grid search over `[0, 1]` in 21 steps on sub-01..04 mean top-1. The best `W_ENC` is applied unchanged to sub-05/06.

## Format and round-trip
`save_submission` asserts `(102, 102)` int64 and that each row is a permutation of `TEST_IDS = 1001..1102`. The round-trip cell writes a sub-01 ranking, reloads it from disk and asserts the on-disk score equals the in-memory score — this is the exact code path we rely on for 05/06.

## Decisions
- Rank fusion over score-concatenation because the encoding stream's `S` is a weighted correlation (noisy magnitude), while the decoder's `S` is a cosine on L2-normalised vectors (bounded). Scales are not comparable; ranks are.
- `W_ENC` is tuned on sub-01..04 only. No model selection ever uses sub-05/06.
- `distinct top-1 ids = X/102` is logged for sub-05/06 as a quick degenerate-prediction check (if the model collapsed to one id, X would be ~1).

## Outputs
- `submissions/sub-05_ranked_candidate_list.npy` — `(102, 102)` int64, each row a permutation of 1001..1102 in `load_test(5)` order.
- `submissions/sub-06_ranked_candidate_list.npy` — same spec.

## Verification
- Round-trip assertion passes for sub-01.
- `distinct top-1 ids` for sub-05/06 is comfortably > 1 (chance diversity ≈ 102 unique).
- The printed `fused` row of the stream table beats both `encoding` and `decoder` on mean top-1/top-5 over sub-01..04. If not, drop fusion and submit the better single stream.

## Next
- 07 produces the plots and tables that go into the report.
