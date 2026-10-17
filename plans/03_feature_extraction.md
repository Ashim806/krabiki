# Stage 03 — Feature extraction

Notebook: [`03_feature_extraction.ipynb`](../03_feature_extraction.ipynb)

## What it does
Per-video features, keyed by video id (train 1..1000, test 1001..1102), shared across all subjects. Each model block is independent and idempotent (checks the cache before recomputing).

## Caches
- `frames/<05d>/f00..f15.jpg` — 16 uniformly-spaced frames per video
- `audio/<05d>.wav` — mono 16 kHz
- `features/alexnet_<layer>.npz` — layers `features.{0,3,6,8,10}`, frame-averaged
- `features/clip_vitl14.npz` — CLIP ViT-L/14, frame-averaged, L2-normalised
- `features/dinov2_vitl14.npz` — DINOv2 ViT-L/14 CLS token, frame-averaged, L2-normalised (facebook/dinov2-large)
- `features/videomae_v2.npz` — VideoMAE-large on the 16 raw frames (HF proxy for VideoMAE-v2; swap to V-JEPA 2 by changing the ckpt)
- `features/clap.npz` — LAION CLAP audio features (BEATs replaced with CLAP: pinned HF ckpt, text-aligned space is reused by 05)
- `features/lowlevel.npz` — luminance mean/std, motion energy, audio RMS/ZCR/log-mel

## Decisions / deltas vs plan
- Image stack: AlexNet (5 layers) + CLIP ViT-L/14 + **DINOv2 ViT-L/14** (added after review). AlexNet → early visual (~100 ms); DINOv2 → mid-level ventral stream; CLIP → late semantic. Banded ridge in 04 picks the weight per channel×time.
- 16 frames (plan said 16; baseline uses 8).
- BEATs → CLAP: no pinned HF BEATs checkpoint; CLAP covers the audio band and its text-aligned space will be reused as the fixed target for the InfoNCE loss in 05.
- VideoMAE-v2 proxy = `MCG-NJU/videomae-large` (there is no official VideoMAE-v2 HF release yet). V-JEPA 2 is a drop-in via `AutoProcessor` / `AutoModel` when its checkpoint becomes available.
- Net2Brain is wrapped around a symlinked flat folder of frames because its API expects a directory of images.

## Verification before moving on
- `features/*.npz` all have `n=1102` and consistent dtype `float32`.
- Pick a random video id, confirm `load_features('clip_vitl14')[vid]` has unit norm.
- Rerun a single cell: idempotency check (no re-extraction, prints only the save line).

## Next
- Stage 04 (encoding model) will consume these banks as separate ridge "bands".
