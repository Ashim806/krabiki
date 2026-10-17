# Environment setup

## Local (Mac, drafting/light runs)
```bash
conda create -n krabiki python=3.11 -y
conda activate krabiki
pip install -r requirements.txt
pip freeze > requirements.lock.txt   # once, after first clean install
```

Local runs 00 and the small cells of 04 fine. Heavy feature extraction (03) and the contrastive decoder (05) must run on the Kaggle GPU.

## Kaggle (GPU kernel)
- Attach the dataset (contains `stimuli_train/`, `stimuli_test/`, `trn/`, `tst/`, `tst_stim/`).
- Turn **Internet** on so HF/Net2Brain checkpoints can download.
- First install cell:
  ```
  !pip install -q av soundfile librosa mne himalaya open_clip_torch transformers
  !pip install -q "net2brain @ git+https://github.com/cvai-roig-lab/Net2Brain.git"
  ```
  Net2Brain is not on PyPI; it must come from GitHub.
  Do **not** reinstall torch — Kaggle ships a CUDA build already.
- Set the working dir to `/kaggle/working` and point `TRAIN_DIR` / `TEST_DIR` / `STIM_*_DIR` at `/kaggle/input/<dataset>/...` at the top of each notebook before running.
- Cache HF models on `/kaggle/working/hf` so a kernel restart does not redownload:
  ```python
  import os; os.environ['HF_HOME'] = '/kaggle/working/hf'
  ```

## Checkpoints to expect on first run of 03
- AlexNet (torchvision, via Net2Brain) — small
- CLIP ViT-L/14 (OpenCLIP, OpenAI weights) — ~1.7 GB
- VideoMAE-large (`MCG-NJU/videomae-large`) — ~1.2 GB
- CLAP (`laion/clap-htsat-unfused`) — ~0.6 GB

## Reproducing baseline on sub-01
After 00 + 04's baseline cell, expect ≈3.9% top-1 / ≈11.8% top-5 on `tst/sub-01_tst.npy`. If you see different numbers, something in loading or preprocessing drifted.
