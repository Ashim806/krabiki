# Stage 05 — Contrastive EEG decoder

Notebook: [`05_contrastive_decoder.ipynb`](../05_contrastive_decoder.ipynb)

## What it does
A single cross-subject EEG → video-embedding model trained with symmetric InfoNCE. Target space = concat(CLIP ViT-L/14, DINOv2 ViT-L/14, CLAP), each L2-normalised then concatenated and re-normalised. At inference, cosine similarity between the encoded EEG and each candidate video's target produces `S[sub]`.

## Architecture
- `EEGEncoder`:
  - 6 subject-specific `Conv1d(17, 17, kernel=1)` adapters — a learnable channel remix per subject.
  - Shared ShallowConvNet-style trunk: temporal `Conv2d(1→40, 1×25)` → spatial `Conv2d(40→40, 17×1)` → BN → square → `AvgPool1d(15)` → log → dropout.
  - Projection to the target dimension; output is L2-normalised.
- Learnable temperature `τ = exp(log_tau)`, initialised at CLIP's `1/0.07`.

## Training
- Pool all 6 subjects' train trials → `(sub_idx, eeg, vid)` triples.
- 5-fold CV **on video ids** (not trials) so cross-subject leakage of the same video cannot happen.
- Augmentation: per-trial z-score, time-shift ±5 samples, Gaussian noise σ=0.05.
- AdamW, cosine LR schedule, 60 epochs, early stop (patience 10) on held-out video top-1.
- Best fold's weights saved to `weights/decoder.pt`.

## Scoring
For each subject, every test EEG recording is encoded; cosine similarity vs the 102 candidate target embeddings → `S` of shape `(102, 102)` in `load_test(sub)` order with column `j` = `TEST_IDS[j]`. Saved to `scores/decoder_sub-01..06.npz`.

## Decisions / deltas vs plan
- Target = CLIP + DINOv2 + CLAP (not CLIP + VideoMAE + BEATs). DINOv2 added after review; CLAP replaces BEATs; VideoMAE is reserved for the encoding model (04) because its 1408-dim per-video features add noise to a contrastive target space.
- MVNN **not** used in 05 — the per-subject input adapter already learns a channel remix, and running MVNN on 05's data would require 04's cache.
- CV split is on videos, not trials, specifically to avoid train/val overlap when the same video is seen by multiple subjects.

## Verification
- CV mean top-1 printed at the end of training should be ≥ 1/K_val (chance for 1 of ~140 held-out videos is ~0.7%); hitting > 5% means the loss is actually learning.
- Sub-01..04 test top-1 printed by the scoring loop should be above the baseline ridge; otherwise 06 will downweight this stream.
- `scores/decoder_sub-01..06.npz` all exist with `S.shape == (102, 102)` and `test_ids == TEST_IDS`.

## Next
- 06 fuses `scores/encoding_sub-XX.npz` + `scores/decoder_sub-XX.npz` → final submissions for sub-05/06.
