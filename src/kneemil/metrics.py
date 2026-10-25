"""Competition metric and bootstrap confidence interval."""

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(y_true, y_pred):
    """Mean ROC AUC over findings; findings with a single class are skipped.

    Returns ``(macro, per_finding)``; ``per_finding`` holds NaN for skipped findings.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per = []
    for j in range(y_true.shape[1]):
        y = y_true[:, j]
        per.append(roc_auc_score(y, y_pred[:, j]) if 0 < y.sum() < len(y) else np.nan)
    return float(np.nanmean(per)), per


def bootstrap_macro_auc(y_true, y_pred, n=1000, seed=0):
    """Bootstrap the macro AUC by resampling studies with replacement.

    Returns ``(mean, (low, high))``, the 95 % percentile interval. With small
    validation sets (58 studies, sometimes 9 positives) this interval shows how
    much of a difference between two models is noise.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rs, out = np.random.RandomState(seed), []
    for _ in range(n):
        i = rs.randint(0, len(y_true), len(y_true))
        a = [roc_auc_score(y_true[i, j], y_pred[i, j]) for j in range(y_true.shape[1])
             if 0 < y_true[i, j].sum() < len(i)]
        if a:
            out.append(np.mean(a))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(np.mean(out)), (float(lo), float(hi))
