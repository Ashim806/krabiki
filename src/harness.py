"""Cross-validation harness: honest top-1 / top-5 estimates using training data only.

Every training trial gets a prediction from a model that never saw it (K-fold),
then trials are ranked in random groups of 102 candidates, exactly like the test.
"""
import os
import numpy as np
from scipy.signal import decimate
from sklearn.model_selection import KFold

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eeg-mind-reading-challenge')
SFREQ = 500          # Hz, raw EEG
DEC = 5              # downsample factor -> 100 Hz, same as the organizers' notebooks
N_CAND = 102         # candidates per ranking, same as the test set


def load_train(sub):
    """Return (stim_ids, eeg) with eeg of shape (n_trials, 17, 300) at 100 Hz."""
    trn = np.load(f'{DATA_DIR}/trn/sub-{sub:02d}_trn.npz')
    ids = np.array(sorted(trn.files, key=int), dtype=int)
    eeg = np.stack([trn[str(i)] for i in ids])
    return ids, decimate(eeg, DEC, axis=-1, ftype='fir', zero_phase=True)


def load_test(sub):
    """Return (eeg, labels) in recording order; labels is None for sub-05/06."""
    eeg = np.load(f'{DATA_DIR}/tst/sub-{sub:02d}_tst.npy')
    eeg = decimate(eeg, DEC, axis=-1, ftype='fir', zero_phase=True)
    lab_path = f'{DATA_DIR}/tst_stim/sub-{sub:02d}_tst_stim.npy'
    labels = np.load(lab_path) if os.path.exists(lab_path) else None
    return eeg, labels


def corr_similarity(M, P, window=slice(5, 60)):
    """Organizers' similarity: Pearson correlation of flattened channel x time (50-600 ms).

    M: real EEG (n, 17, T), P: predicted EEG (m, 17, T) -> S (n, m).
    """
    M = M[:, :, window].reshape(len(M), -1)
    P = P[:, :, window].reshape(len(P), -1)
    Mz = (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    Pz = (P - P.mean(1, keepdims=True)) / P.std(1, keepdims=True)
    return Mz @ Pz.T / M.shape[1]


def ranks_of_truth(S, true_col):
    """Rank (1 = best) of the true candidate in each row of S."""
    return (-S).argsort(1).argsort(1)[np.arange(len(S)), true_col] + 1


def score(S, true_col):
    r = ranks_of_truth(S, true_col)
    return {'top1': (r == 1).mean(), 'top5': (r <= 5).mean(), 'median_rank': float(np.median(r))}


def cross_validate(X, Y, fit_predict, similarity=corr_similarity, n_folds=5, n_repeats=20, seed=0):
    """X: (n, d) features, Y: (n, 17, T) EEG. fit_predict(X_tr, Y_tr, X_te) -> Y_te_pred.

    Each group of 102 candidates is drawn from ONE held-out fold, so all candidates come
    from a model that saw none of them (as on the real test). Mixing folds biases the
    score below chance: other folds' models were trained on the true trial.
    """
    rng = np.random.default_rng(seed)
    ranks = []
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        assert len(te) >= N_CAND, 'held-out fold smaller than 102 trials: use fewer folds'
        P = fit_predict(X[tr], Y[tr], X[te])
        for _ in range(n_repeats):
            idx = rng.choice(len(te), N_CAND, replace=False)
            ranks.append(ranks_of_truth(similarity(Y[te][idx], P[idx]), np.arange(N_CAND)))
    r = np.concatenate(ranks)
    return {'top1': (r == 1).mean(), 'top5': (r <= 5).mean(), 'median_rank': float(np.median(r))}


def ridge_baseline(X_tr, Y_tr, X_te):
    """Organizers' encoding model: scaler -> PCA(95%) -> scaler -> RidgeCV, unchanged."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver='full'),
                          StandardScaler(), RidgeCV(alphas=np.logspace(-3, 6, 20)))
    model.fit(X_tr, Y_tr.reshape(len(Y_tr), -1))
    return model.predict(X_te).reshape(len(X_te), *Y_tr.shape[1:])
