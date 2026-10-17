# Baseline review and upgrade plan

Review of the organizers' two notebooks (`01_preprocess_and_train_model`, `02_predict`) as pasted in the project. I haven't run them (no data in the project yet), so the items marked **check** are things I read from the code that you should confirm on your machine.

## 1. Bugs and risks in the code

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | train, `video_for` | `vids = sorted(glob(...))` then `vids[stim_id - 1]`. The predict notebook looks videos up as `/{stim_id}.mp4`, so files are probably **not zero-padded**. Then the string sort goes `1, 10, 100, 1000, 1001, …, 101, …` and stimulus 2 gets video 10, stimulus 3 gets video 100, etc. EEG and features would be misaligned for most of training. **Check this first**: print `video_for(2)`. | Look up by name: `os.path.join(STIM_DIR, f'{sid}.mp4')`, as the test notebook already does. |
| 2 | train, evaluation | The 67% top-1 is measured on the same trials the ridge was fitted on, so it says nothing about generalisation. | Use 5-fold CV inside the training set for every decision. Use sub-01..04 test labels only as a final sanity check. |
| 3 | predict, `features_for` | `np.load(+FEAT_TEST_DIR + ...)`: the unary `+` on a string raises `TypeError`. | Remove the `+`. |
| 4 | predict, last cell | `eeg_arr` and `subject_num` are undefined. | `np.save(f'sub-{SUB:02d}_ranked_candidate_list.npy', ranked)` |
| 5 | predict, row order | `S` is built from `tst_eeg[order]`, where `order` comes from the **labels**. Subjects 05/06 have no labels, and the submission must keep the original recording order. If you copy this cell, the rows end up in the wrong order (or it crashes). | For submission use `M = tst_eeg[...]` with **no** reordering. Keep `order` only in the validation code for 01..04. |
| 6 | train, RidgeCV | One alpha for all 17×300 outputs, chosen by GCV on the training fit. | `RidgeCV(alphas=..., alpha_per_target=True)`, or pick alpha by the CV ranking score, which is what you are scored on. |
| 7 | features | AlexNet `features.3` is conv2 (very low level), and the 8 frames are **averaged**. All temporal structure of a 3 s video is gone, and the EEG is a 3 s time course. | See section 2. |
| 8 | similarity | Correlation over flattened `channel×time` using only samples 5:60 (50–600 ms). That throws away 2.4 s of a 3 s video. Raw-µV channels with large variance dominate, and there's no hubness correction. | z-score each (channel, time) feature with training stats, weight time points by CV reliability, use the whole window, and normalise `S` (section 2, step 3). |

No leakage in the training notebook itself: scaler, PCA and ridge are fitted on training data only, and the test videos are only passed through the frozen AlexNet, which is allowed.

## 2. Upgrade path, in priority order

Rough guide: steps 1 to 3 are cheap and should give the biggest jump. The organizers said about 20% top-1 should be reachable.

### Step 0. Make the evaluation trustworthy (do this before anything else)
- Fix bugs 1, 3, 4 and 5.
- Write one function, `score(S) -> top1, top5`, and a 5-fold CV over training trials that builds a 102-way (or fold-sized) ranking and reports it. Use 102-candidate subsets so the numbers are comparable to test.
- Run this for every subject (01..06 all have training data) and report the mean.

### Step 1. Better visual features (organizer idea: video encoders, multi-task)
Swap AlexNet for features that are far more brain-predictive, and keep time:
- **Image models per frame:** CLIP ViT-L/14, SigLIP, DINOv2. Take 1 frame every ~0.25–0.5 s and keep them as a sequence instead of averaging.
- **Video models:** VideoMAE / VideoMAE v2, V-JEPA, X-CLIP, InternVideo. They encode motion, which the frame models miss.
- **Audio:** the clips have sound. Use BEATs, wav2vec2, CLAP or AudioMAE embeddings.
- **Multi-task (organizer idea):** low-level (Gabor energy, luminance, contrast, optical-flow magnitude per frame), depth (Depth-Anything), segmentation or semantic (CLIP text-aligned). Low-level luminance and contrast changes drive a lot of EEG variance, so they're cheap and strong.
- Try **several layers** (early, middle, late) and let CV pick or combine them.

### Step 2. A time-resolved encoding model
The current model maps one static vector to the whole 3 s response. Better:
- **Lagged / TRF model:** predict EEG at time *t* from features of frames in a window (t−400 ms … t). This is the standard temporal response function used for continuous stimuli, and it shares weights over time, so it needs far fewer parameters than 5100 independent outputs.
- Or keep the per-time-point ridge but feed it the time-binned feature sequence instead of the mean.
- Use banded ridge, or fit per feature space, then **ensemble the similarity matrices** (sum the z-scored `S` from each feature space). Ensembling different feature spaces is usually a reliable gain.

### Step 3. Better matching (no new training needed)
- **Feature weighting:** on CV, measure how well each (channel, time) is predicted (correlation of predicted vs real). Keep or weight only the reliable ones before computing similarity. This alone often beats the hand-picked 5:60 window.
- **Normalisation:** z-score EEG per channel. Z-scoring test EEG is explicitly allowed (unsupervised).
- **Hubness correction:** some predicted responses look similar to everything. Z-score `S` along columns as well as rows (or use CSLS / dual-softmax). That's cheap and usually a clear win in retrieval.
- **Ask the organizers** before exploiting the fact that each of the 102 videos appears exactly once per subject (Hungarian / Sinkhorn assignment). It can add a lot, but it's a grey zone under "no test data for training".

### Step 4. Decoding direction + cross-subject data (bigger effort)
- Train an EEG encoder (EEGNet / ATM / NICE-style, as in the THINGS-EEG work) with a **contrastive (CLIP-style) loss** to map EEG into the video embedding space. Rank by cosine similarity.
- **Pool training data from all 6 subjects** (~5,500 trials instead of ~900), with a small subject-specific input layer. With single, noisy trials this is the most important thing for a deep model to work at all.
- Use data augmentation (time shifts, channel dropout, noise) and mixup.
- Finally, combine this similarity with the encoding-model similarity from step 2.

### Step 5. In-silico brain models (organizer idea)
Use a public pretrained brain-encoding model that predicts brain responses from video (+audio), e.g. Meta's TRIBE (Algonauts 2025 winner) or models in Net2Brain. Use its predicted responses as a feature space in step 2. Check the release date is before the hackathon and that weights are public.

## 3. Suggested first day
1. Fix bugs, build the CV harness, and confirm the baseline numbers (and whether bug 1 was real; if it was, the fix alone may move test accuracy a lot).
2. Extract CLIP + DINOv2 per-frame features (time-preserved) and one video model.
3. Ridge with per-target alpha + reliability-weighted correlation + column z-scoring of `S`.
4. Ensemble feature spaces, check against sub-01..04 test, and write the submission files for 05/06 in original row order.
