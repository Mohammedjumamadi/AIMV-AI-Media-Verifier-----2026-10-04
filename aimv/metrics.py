"""Metrics, calibration and operating-point selection."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, iters: int = 200) -> float:
    """Temperature scaling on validation logits so probabilities mean what they say."""
    if len(np.unique(labels)) < 2:
        return 1.0
    lg = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))


def threshold_at_fpr(probs: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate on REAL media is <= target_fpr."""
    real = np.sort(probs[labels == 0])
    if len(real) == 0:
        return 0.5
    k = int(np.floor(target_fpr * len(real)))
    # Threshold strictly above the (n-k)-th highest real score.
    return float(min(1.0, real[len(real) - k - 1] + 1e-6)) if k < len(real) else 0.0


def eer(probs: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    labels = labels.astype(int)
    pred = (probs >= threshold).astype(int)
    out = {
        "n": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float((pred == labels).mean()) if len(labels) else float("nan"),
    }
    real, fake = labels == 0, labels == 1
    out["real_acc"] = float((pred[real] == 0).mean()) if real.any() else float("nan")
    out["fake_acc"] = float((pred[fake] == 1).mean()) if fake.any() else float("nan")
    out["balanced_acc"] = float(np.nanmean([out["real_acc"], out["fake_acc"]]))
    if real.any() and fake.any():
        out["auc"] = float(roc_auc_score(labels, probs))
        out["ap"] = float(average_precision_score(labels, probs))
        out["eer"] = eer(probs, labels)
    return out
