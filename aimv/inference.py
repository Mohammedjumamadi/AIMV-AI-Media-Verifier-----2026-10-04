"""Load trained detectors and score full images with tiling.

Resizing a whole receipt to 256 px destroys the very pixels that were edited, so
large images are scored as overlapping native-resolution tiles. The image score
combines the mean tile score (whole-image generation) with the top-k tiles
(local edits, e.g. a changed amount), and a per-tile heatmap is returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .augment import random_crop, resize_longest
from .config import Config, config_from_dict
from .data import load_rgb, to_tensor
from .metrics import sigmoid
from .models import build_model


@dataclass
class LoadedDetector:
    name: str
    model: torch.nn.Module
    cfg: Config
    temperature: float
    threshold: float
    device: torch.device


def load_detector(path: str | Path, device: torch.device) -> LoadedDetector:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    cfg.model.pretrained = False  # weights come from the checkpoint
    model = build_model(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return LoadedDetector(Path(path).parent.name or Path(path).stem, model, cfg,
                          float(ckpt.get("temperature", 1.0)), float(ckpt.get("threshold", 0.5)), device)


def tile_boxes(w: int, h: int, size: int, overlap: float = 0.25, max_tiles: int = 64):
    stride = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(1, w - size + 1), stride)) or [0]
    ys = list(range(0, max(1, h - size + 1), stride)) or [0]
    if xs[-1] + size < w:
        xs.append(w - size)
    if ys[-1] + size < h:
        ys.append(h - size)
    boxes = [(x, y) for y in ys for x in xs]
    if len(boxes) > max_tiles:  # subsample evenly on very large inputs
        idx = np.linspace(0, len(boxes) - 1, max_tiles).round().astype(int)
        boxes = [boxes[i] for i in idx]
    return boxes, xs, ys


@torch.inference_mode()
def score_image(det: LoadedDetector, img: Image.Image, tta: bool = True, batch: int = 16,
                max_tiles: int = 64) -> dict:
    size = det.cfg.data.crop_size
    orig_w = img.size[0]
    img = resize_longest(img, det.cfg.data.max_side)
    scale = orig_w / img.size[0]  # map tile boxes back to the uploaded image
    w, h = img.size
    if w < size or h < size:
        crops, boxes = [random_crop(img, size, train=False)], [(0, 0)]
        xs, ys = [0], [0]
    else:
        boxes, xs, ys = tile_boxes(w, h, size, max_tiles=max_tiles)
        crops = [img.crop((x, y, x + size, y + size)) for x, y in boxes]
    x = torch.stack([to_tensor(c) for c in crops])
    logits = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(det.device)
        out = det.model(xb)
        if tta and det.cfg.augment.hflip > 0:
            out = (out + det.model(torch.flip(xb, dims=[3]))) / 2
        logits.append(out.float().cpu())
    tile_logits = torch.cat(logits).numpy() / det.temperature
    tile_probs = sigmoid(tile_logits)
    k = max(1, math.ceil(0.2 * len(tile_probs)))
    topk = float(np.sort(tile_probs)[-k:].mean())
    mean = float(sigmoid(tile_logits.mean()))
    prob = max(mean, topk) if len(tile_probs) > 1 else mean

    heat = np.full((len(ys), len(xs)), np.nan)
    pos = {(bx, by): (j, i) for j, by in enumerate(ys) for i, bx in enumerate(xs)}
    for (bx, by), p in zip(boxes, tile_probs):
        heat[pos[(bx, by)]] = p
    tiles = [{"box": [round(bx * scale), round(by * scale), round(min(w, bx + size) * scale),
                      round(min(h, by + size) * scale)], "prob": float(p)}
             for (bx, by), p in zip(boxes, tile_probs)]
    return {"prob_fake": prob, "mean_tile_prob": mean, "topk_tile_prob": topk, "n_tiles": len(tile_probs),
            "threshold": det.threshold, "is_fake": prob >= det.threshold, "heatmap": heat,
            "tile_size": size, "tiles": tiles}


def verdict(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "AI-generated / AI-edited"
    if prob >= min(0.5, threshold) * 0.6:
        return "uncertain — needs human review"
    return "likely real"


def ensemble(results: list[dict]) -> dict:
    """Average calibrated log-odds; fake if any detector is above its own threshold by margin."""
    eps = 1e-6
    lo = [math.log((r["prob_fake"] + eps) / (1 - r["prob_fake"] + eps)) for r in results]
    prob = float(sigmoid(np.array(np.mean(lo))))
    margins = [r["prob_fake"] - r["threshold"] for r in results]
    thr = float(np.mean([r["threshold"] for r in results]))
    is_fake = prob >= thr or max(margins) > 0.25
    return {"prob_fake": prob, "threshold": thr, "is_fake": bool(is_fake),
            "verdict": "AI-generated / AI-edited" if is_fake else verdict(prob, thr)}


def load_image(path: str | Path) -> Image.Image:
    return load_rgb(path)
