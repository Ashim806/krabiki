# Plan: EEG video-identification pipeline (krabiki), notebook-based

## Context
SchubKI hackathon task: rank the 102 test videos for each EEG test recording (17 ch × 1500 @ 500 Hz). Scored on
sub-05/06 by top-1/top-5. The organiser baseline ([01_preprocess_and_train_model.ipynb](01_preprocess_and_train_model.ipynb),
[02_predict_movies.ipynb](02_predict_movies.ipynb)) is a per-subject AlexNet-conv2 → PCA → RidgeCV encoding model with
correlation identification. It reaches 3.9% / 11.8% on the sub-01 test set and has no held-out validation. Its submission
cell is broken (`+FEAT_TEST_DIR` TypeError, undefined `subject_num`/`eeg_arr`, rows in label-sorted order). The analysis
and the methods catalogue are in [README.md](README.md).
Decisions:
- A cloud GPU is available.
- Implementation is staged: improved encoding model, then contrastive decoder, then ensemble.
- **Everything is written as Jupyter notebooks.**
- **The user approves each edit manually.**

Data facts:
- 850–972 train recordings per subject.
- 695 train videos are shared by all 6 subjects.
- Test std ≈ 0.65× train, so normalise each split separately.
- Test row order is shuffled per subject.

## Notebooks (Python 3.11, conda env `krabiki`; 01/02 stay untouched as reference)
Each notebook follows the existing style: config cell at the top (`SUB`, paths, `SFREQ`), short markdown explanations
between steps, outputs cached to disk so later notebooks can load them.

- **`00_setup_and_data.ipynb`**
  - Builds the `requirements.txt` install cell.
  - Defines the loading helpers: `load_train(sub)`, `load_test(sub)` (original order), `load_test_labels(sub)`.
  - Data sanity plots: ERP per subject and train vs test variance.
  - Evaluation helpers: `topk(ranked, truth)`, `ranked_ids(S)` → (102,102) video IDs.
  - Submission writer with asserts: shape (102,102), int dtype, each row a permutation of 1001–1102, unsorted rows.
  - Helpers are copied verbatim into later notebooks (no `src/` package), so each notebook runs standalone.
- **`03_feature_extraction.ipynb`** (run on the GPU)
  - Time-resolved frames (16 per video) + audio track. Reuses the baseline `dump_frames` logic, adding audio decoding.
  - Features:
    - multiple AlexNet layers via Net2Brain
    - CLIP ViT-L/14 per frame
    - VideoMAE-v2 or V-JEPA 2 clip embeddings
    - BEATs/CLAP audio embeddings
    - low-level luminance, motion and audio envelope
  - Saved to `features/<model>.npz` keyed by video id (git-ignored).
- **`04_encoding_model.ipynb`**
  - Preprocessing:
    - band-pass 0.1–40 Hz
    - decimate to 100 Hz (as in the baseline)
    - per-split z-score
    - grand-ERP subtraction (train only)
    - MVNN
  - Model: banded ridge (himalaya) video→EEG, with identification weighted per channel×time by CV predictability.
  - One ablation cell per quick win. Results table on sub-01..04 test.
  - First reproduces the baseline ≈3.9% on sub-01.
- **`05_contrastive_decoder.ipynb`** (GPU)
  - EEG encoder: subject-specific input layer + ShallowConvNet/EEG-Conformer trunk.
  - Trained with InfoNCE against frozen video+audio embeddings, on all 6 subjects' train data.
  - Early stopping on train CV folds. Saves weights to `weights/`.
- **`06_ensemble_and_submission.ipynb`**
  - Loads the encoding + decoder scores and fuses them, with weights chosen on train CV.
  - Format check on sub-01 using the exact submission code path.
  - Writes `submissions/sub-05_ranked_candidate_list.npy` and `sub-06_…`.
- **`07_analysis_for_report.ipynb`**
  - Layer × latency encoding curves.
  - Visual / audio / semantic variance partitioning.
  - Confusion and failure cases.

## Workflow
- **Step 0:** save this plan into the project as `plans/PLAN.md` so it can be opened from the workspace.
  Each notebook's stage notes and results also go under `plans/`.
- Create notebooks one at a time with NotebookEdit. Pause for user approval on each edit, as requested.
- Rule guardrails:
  - Never fit or select templates using the `tst` labels of 01–04 (validation only).
  - The Hungarian one-to-one assignment stays an optional, separately reported cell until the organisers confirm it is allowed.

## Verification
- `00` + `04` baseline cell reproduces ≈0.039 / 0.118 on sub-01.
- Every stage adds a row to a mean ± sd top-1/top-5 table (sub-01..04 test and train CV), also copied into the README.
- `06`: the unsorted sub-01 submission, scored against `tst_stim/sub-01_tst_stim.npy`, matches the evaluation-path score.
  The sub-05/06 files pass the format asserts.
