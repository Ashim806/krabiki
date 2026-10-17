# Stage 07 — Analysis plots

Notebook: [`07_analysis_for_report.ipynb`](../07_analysis_for_report.ipynb)

## Figures produced
1. `figs/encoding_curves.pdf` — per-bank CV Pearson r (mean across channels) vs. time, four panels (sub-01..04). Expected shape: early AlexNet layers rise around 100 ms, CLIP/DINOv2 rise around 200–300 ms, CLAP tracks audio onsets.
2. `figs/variance_partitioning.pdf` — stacked bars per subject with unique and shared R² for vision / semantic / audio groupings.
3. `figs/confusion_matrix.pdf` — fused-stream top-1 confusion matrices on sub-01..04. Diagonal count = top-1 hits.
4. `figs/results_summary.csv` — mean ± sd of top-1/top-5 per ablation tag across sub-01..04, including the fused ensemble.

## Dependencies
- `cache_04/pp_sub-XX.npz` (04's preprocessed EEG)
- `features/*.npz` (03)
- `scores/encoding_sub-XX.npz`, `scores/decoder_sub-XX.npz`
- `cache_04/results.csv` for the ablation history

## Decisions
- Variance partitioning uses three groups: **vision** (AlexNet + VideoMAE), **semantic** (CLIP + DINOv2), **audio** (CLAP + lowlevel). Semantic contains CLIP's text-aligned space plus DINOv2's pure-vision space, which usually overlaps heavily with the vision group — expect large `vision ∩ semantic` shared variance.
- Confusion matrix in 07 uses a **50/50** fuse so the figure is a clean baseline independent of 06's grid-searched `w`.
- Results CSV copies 04's ablation rows plus the fused row; it is the authoritative table for the report.

## Verification
- All three PDFs land under `figs/`.
- `encoding_curves.pdf` panels show at least one bank with r > 0.1 somewhere; otherwise either 03 or 04's preprocessing drifted.
- Variance-partitioning stacked bars never have negative contributions (we clip at 0 before plotting; large negative values before clipping indicate severely overlapping fits).
