# krabiki

Mind Reading Using Machine Learning and EEG (SchubKI National AI Hackathon 2026).
Task: given one EEG recording (17 ch × 1500 samples @ 500 Hz, 3 s from video onset), rank the 102 test videos
(1001–1102) by how likely each was to have produced it. Scored on sub-05/06 by top-1 / top-5 (chance 1.0% / 4.9%).
Score weights: 70% accuracy, 30% qualitative (soundness/novelty 15, code + reproducibility 10, presentation 5).

## Data facts (checked)

| sub | train recordings | test |
|-----|-----------------|------|
| 01 | 968 | 102 + labels |
| 02 | 972 | 102 + labels |
| 03 | 961 | 102 + labels |
| 04 | 947 | 102 + labels |
| 05 | 952 | 102, **labels withheld** |
| 06 | 850 | 102, **labels withheld** |

- 695 training videos were seen by **all six** subjects, so cross-subject alignment/pooling is possible.
- Test EEG std is ~0.65× the train std for every subject (≈0.30 vs ≈0.46). One hypothesis: test trials are averaged
  over 2–3 repeats (higher SNR). Practical consequence: **z-score train and test separately** (allowed: unsupervised).
- Test recording order differs per subject (shuffled independently).
- 1000 train + 102 test 3-s videos matches the layout of the BOLD Moments Dataset (Moments in Time clips). If that is
  the source, public captions and annotations exist for the videos. Still to verify.
- The 17 channels may be the 17 posterior channels commonly used with THINGS-EEG2 (O/PO/P sites). Still to verify; it
  matters for reusing THINGS-EEG2–trained models (BERG, NICE/ATM).

## Baseline pipeline (01 / 02 notebooks)

Per subject, an **encoding model** (video → predicted EEG). Identification is done by correlation.

1. Load `trn/sub-XX_trn.npz` → `(n, 17, 1500)`; `decimate` by 5 (FIR anti-alias) → 100 Hz, `(n, 17, 300)`.
2. Extract 8 evenly spaced frames per video with OpenCV.
3. Net2Brain AlexNet, layer `features.3` (conv2, low-level), flattened (139,968 dims), **averaged over the 8 frames**.
4. StandardScaler → PCA (95% variance, 723 comps) → StandardScaler.
5. `RidgeCV` (one shared alpha ≈ 1.3e4) maps features to all 17×300 = 5100 EEG values.
6. Identification: per-row z-score of measured and predicted EEG in samples 5:60 (≈50–600 ms), flattened.
   Similarity = Pearson correlation matrix, then rank by `argsort(-S)`.

Results (sub-01): train top-1 0.78 (**in-sample, so not meaningful**); test top-1 **3.9%**, top-5 **11.8%**, median rank 25.5.

### Issues found in the baseline code

- **No real validation.** The "train accuracy" scores the model on the same data it was fit on. Use K-fold CV on train,
  and the sub-01..04 test sets as the outer check. (The markdown also says 67% where the output shows 78%.)
- `02` cell 14: `np.load(+FEAT_TEST_DIR+...)` has a unary `+` on a string, which raises `TypeError` (stored output came from an older version).
- `02` cell 28: `np.save(f'sub-0{subject_num}...', eeg_arr)` uses undefined names. More importantly, `ranked` is built
  from `tst_eeg[order]` (label-sorted rows). **The submission must keep the original row order**, and `order` cannot be
  computed for sub-05/06 anyway. Build the submission from unsorted `tst_eeg`.
- Averaging frames removes the video's temporal dynamics, and conv2 only captures low-level information. No audio, although the videos had sound.
- Ridge fits all 300 time points, but scoring uses only 50–600 ms with every channel and time point weighted equally,
  including poorly predicted ones.
- The shared evoked response (ERP, the same for every video) is not removed. It dominates correlations and adds no
  information about which video was shown.
- No band-pass, artefact handling or noise normalisation. Each subject is modelled alone, ignoring the other 5 subjects' training data.
- Train and test are not normalised separately, despite the SNR/scale difference.

## Validation protocol (do this first)

- Inner: 5-fold CV on each subject's train set, scored with 102-way identification (random 102-video subsets), to tune hyperparameters.
- Outer: train on `trn` → score on `tst` of sub-01..04; report mean ± sd top-1/top-5. Never fit anything on `tst` labels.
- Reproduce the submission code path on sub-01 and score it with `tst_stim/sub-01_tst_stim.npy` (organiser-recommended format check).

## Possible methods

### Video encoding (stimulus features)
- **Low-level / hand-crafted** (early EEG components respond to these): luminance, contrast, colour, spatial-frequency energy,
  motion energy / optical flow magnitude over time, scene-cut timing, audio RMS envelope and onsets.
- **Image DNNs per frame**: multiple AlexNet/ResNet/CORnet layers (not only conv2), CLIP ViT-B/L, DINOv2, SigLIP.
- **Video foundation models**: VideoMAE (v2), V-JEPA (2), InternVideo2, X-CLIP / ViCLIP. They encode motion that frame averaging loses.
- **Audio models** (videos played with sound): BEATs, AST, wav2vec2/HuBERT, CLAP, Whisper encoder.
- **Semantic**: caption/label text embeddings (CLIP text, sentence-transformers), e.g. from generated captions or
  dataset annotations if public.
- Keep features **time-resolved** (per frame/window) rather than averaged. This allows temporal response function (TRF) style models.

### EEG preprocessing / reconstruction
- Band-pass (~0.1–40 Hz), clip/reject extreme samples, downsample to 100–200 Hz.
- No pre-stimulus baseline is available (epochs start at onset), so remove a per-epoch mean or detrend instead.
- **Subtract the grand-average ERP** before identification.
- **Multivariate noise normalisation (MVNN)** (Guggenmos et al. 2018, used in THINGS-EEG2). This whitens channels with the noise covariance.
- Per-split z-scoring (train vs test SNR differs). Time-window / channel selection from CV predictability.
- Cross-subject alignment: Euclidean alignment, or CCA / hyperalignment / shared response model on the 695 shared videos.

### Encoding models (video → EEG, the baseline direction)
- Ridge per time point / per channel with separate alphas. **Banded ridge** to combine several feature spaces (himalaya).
- Temporal receptive field (lagged features → EEG time course) using time-resolved video/audio features.
- Score only well-predicted channels × time points (weighted by CV correlation / noise ceiling), or use a Mahalanobis
  distance under the residual covariance instead of plain correlation.

### Decoding models (EEG → embedding)
- Linear/ridge map from EEG to CLIP/video embedding, then rank by cosine similarity to the 102 candidate embeddings.
- **Contrastive EEG–video alignment** (CLIP-style InfoNCE): EEG encoder (EEGNet, ShallowConvNet, EEG-Conformer, NICE/ATM
  architecture) trained against frozen video/image/audio embeddings. This is the state-of-the-art family for zero-shot
  EEG identification on THINGS-EEG.
- Pretrained EEG foundation models as encoder backbones: LaBraM, CBraMod, EEGPT, BIOT. Fine-tune them, watching for channel-set mismatch.

### Multilevel models (subjects × levels)
- Pool all 6 subjects' train data: shared backbone + subject-specific input/output layers (or subject embedding).
- Hierarchical/multi-level feature fusion: low-level features for early latencies, semantic features for later ones.
- Ensemble per-subject and pooled models; ensemble encoding-side and decoding-side similarity scores (rank or z-score fusion).

### Silico models (in-silico EEG)
- **BERG** (Brain Encoding Response Generator): pretrained encoding models (e.g. on THINGS-EEG2) that generate in-silico EEG
  for arbitrary images. Use them on video frames as an extra feature space, as a prior, or for pretraining data.
  Check channel mapping and sampling rate.
- Synthesise extra training pairs from many unlabelled clips (data augmentation for the decoder).

### Ranking / post-processing
- Each test video appears **exactly once** per subject, so a one-to-one assignment (Hungarian `linear_sum_assignment`,
  Sinkhorn) could raise top-1 a lot. ⚠ It could be read as "dataset hacking": **ask the organisers before using**, and
  report results with and without it.
- Test-time z-scoring of similarity rows/columns (removes recordings or videos that look similar to everything).

### Scientific-insight angle (qualitative 30%)
- Time-resolved RSA / encoding: which DNN layer (low → high) best explains which latency (P1/N1 vs. later).
- Visual vs. auditory vs. semantic feature contributions over time (variance partitioning with banded ridge).
- Noise ceiling per subject. Failure analysis: which videos are confused with each other, and why.

## Tools
- Net2Brain (feature extraction + RSA)
- "BERG" (Brain Encoding Response Generator), used to generate in-silico brain activity
- MNE-Python (filtering, MVNN), braindecode (EEG nets), himalaya (banded ridge), scikit-learn
- open_clip / HF transformers (CLIP, DINOv2, VideoMAE, V-JEPA, audio models), decord / PyAV (frame + audio decoding)
- `scipy.optimize.linear_sum_assignment` (see caveat above)

## Suggested plan (17–18 Oct work days)
1. Fix the baseline bugs and build a reusable `src/` pipeline with the validation protocol and a correct submission writer.
2. Quick wins on the encoding baseline: ERP subtraction, per-split z-scoring, multiple layers / CLIP features,
   predictability-weighted scoring.
3. Main model: contrastive EEG encoder against video (+ audio) embeddings, trained on all 6 subjects.
4. Ensemble encoding + decoding scores; ablations for the report; generate the sub-05/06 submissions.
