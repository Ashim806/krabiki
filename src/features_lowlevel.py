"""Low-level, time-resolved video features (Step 1a).

Per frame: luminance, contrast, edge energy, motion (change from previous frame).
Frames are put into fixed time bins by timestamp (fps differs between videos: 24-30),
so bin k always means the same time window after video onset.
"""
import os
import numpy as np
import cv2

from src.harness import DATA_DIR

BIN_S = 0.1          # 100 ms bins
DUR_S = 3.0          # video / EEG epoch length
N_BINS = int(round(DUR_S / BIN_S))
NAMES = ['luminance', 'contrast', 'edges', 'motion']
CACHE = f'{DATA_DIR}/cache/lowlevel_bin{int(BIN_S * 1000)}ms.npz'


def video_path(sid):
    return (f'{DATA_DIR}/stimuli_train/{sid:04d}.mp4' if sid <= 1000
            else f'{DATA_DIR}/stimuli_test/{sid}.mp4')


def frame_stats(path):
    """Return (times, stats) with stats of shape (n_frames, 4)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        edges = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean()
        motion = 0.0 if prev is None else np.abs(g - prev).mean()
        rows.append([g.mean(), g.std(), edges, motion])
        prev = g
    cap.release()
    return np.arange(len(rows)) / fps, np.array(rows, dtype=np.float32)


def bin_stats(times, stats):
    """Average frames into N_BINS time bins -> (N_BINS, 4). Empty bins copy the previous bin."""
    b = np.minimum((times / BIN_S).astype(int), N_BINS - 1)
    out = np.zeros((N_BINS, stats.shape[1]), np.float32)
    for k in range(N_BINS):
        out[k] = stats[b == k].mean(0) if (b == k).any() else out[k - 1]
    out[0, 3] = out[1, 3]    # motion of the first frame is undefined
    return out


def extract_all(stim_ids=range(1, 1103)):
    """{stim_id: (N_BINS, 4)}, cached to the gitignored data folder."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return dict(zip(z['ids'].tolist(), z['feats']))
    ids = list(stim_ids)
    feats = np.stack([bin_stats(*frame_stats(video_path(s))) for s in ids])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, ids=np.array(ids), feats=feats)
    return dict(zip(ids, feats))
