# Stage 04 — Encoding model

Notebook: [`04_encoding_model.ipynb`](../04_encoding_model.ipynb)

## What it does
Per-subject video→EEG encoder, then identification by weighted correlation between predicted EEG for the 102 candidate videos and measured test EEG. Produces `scores/encoding_sub-XX.npz` with `S` of shape `(102, 102)` in `load_test(sub)` order and column `j` = `TEST_IDS[j]`.

## Pipeline
1. **Preprocess EEG** (`preprocess_subject`): band-pass 0.1–40 Hz (MNE zero-phase), decimate ×5 → 100 Hz (300 samples / trial), train-only grand-ERP subtraction, per-split z-score, MVNN ZCA whitening on channels using train residual covariance. Cached to `cache_04/pp_sub-XX.npz`.
2. **Feature banks**: load each `features/<name>.npz` → z-score → PCA(0.95) → z-score. Default stack = low-level + 5 AlexNet layers + CLIP ViT-L/14 + VideoMAE + CLAP.
3. **Banded ridge** (himalaya `GroupRidgeCV`, torch-cuda backend when GPU). One ridge penalty per bank; 5-fold outer CV on train provides out-of-fold predictions `Y_pred_cv` used as the predictability weight; a final refit on full train produces `Y_pred_test` for all 102 candidate videos. Cached to `cache_04/br_sub-XX.npz`.
4. **Identification**: `weighted_sim` returns `(n_test, 102)` from z-scored dot product over channels × time with weight `max(r, 0)` where `r` = per-(ch,t) Pearson correlation of `Y_pred_cv` vs `Y_trn`. Window defaults to samples 5..60 (50–600 ms post-onset, i.e. the P1/N1/N2 range that the 01/02 analysis flagged).

## Results table
Ablation rows accumulate in `cache_04/results.csv` (one row per `tag × sub`), and `summary_table(results)` prints mean ± sd top-1/top-5 across sub-01..04. The expected rows:
| tag | note |
| --- | --- |
| `baseline_alexnet3` | matches 01/02 (~0.039 / 0.118 on sub-01) — sanity check |
| `alexnet_bands_unweighted` | banded ridge over 5 AlexNet layers only |
| `full_banks_unweighted` | all banks (AlexNet × 5 + CLIP + VideoMAE + CLAP + low-level) |
| `full_banks_weighted` | adds the CV-predictability weight to the ident |

## Decisions
- No hyperparameter is ever chosen using sub-01..04 tst labels — all tuning is CV over the train set.
- The 5–60 sample window is a reproduction choice from the baseline (`eeg[:, :, 5:60]`). Future ablation: scan the window.
- MVNN is train-only; the whitening transform is also applied to test. This is standard for identification and does not leak labels.

## Outputs for 06
`scores/encoding_sub-01..06.npz` with:
- `S` — `(102, 102)` similarity in `load_test(sub)` order
- `w` — `(17, 300)` CV-predictability weight (saved for diagnostics / 06 to re-weight if desired)
- `test_ids` — `[1001..1102]`

## Verification before moving on
- `baseline_alexnet3` row on sub-01 is within ±0.005 of 0.039 top-1 (any big drift means something is off in loading/decimation/windowing).
- `full_banks_weighted` row on sub-01..04 shows strictly higher top-5 than `baseline_alexnet3` — otherwise one of the new banks is poisoning the stack and we need to drop it.
- `S.shape == (102, 102)` and rows match `load_test(sub)` order (no sorting by label).

## Next
- Stage 05 (contrastive decoder) consumes the same features; 06 fuses `scores/encoding_sub-XX.npz` with `scores/decoder_sub-XX.npz`.
