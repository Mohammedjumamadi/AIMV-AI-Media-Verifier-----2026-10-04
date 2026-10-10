"""Evaluate a checkpoint on the test split: overall, per category, per generator,
and under real-world degradations (JPEG / resize chains).

    python -m aimv.evaluate --ckpt runs/documents/best.pt --split test --out runs/documents/eval.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augment import jpeg, random_crop, resize_longest
from .data import load_rgb, read_manifest, to_tensor
from .inference import load_detector
from .metrics import compute_metrics, sigmoid
from .train import pick_device, predict_loader

DEGRADATIONS = {
    "clean": lambda im: im,
    "jpeg90": lambda im: jpeg(im, 90),
    "jpeg75": lambda im: jpeg(im, 75),
    "jpeg50": lambda im: jpeg(im, 50),
    "whatsapp": lambda im: jpeg(resize_longest(im, 1600), 70),
    "facebook_x2": lambda im: jpeg(resize_longest(jpeg(resize_longest(im, 2048), 85), 1080), 75),
}


class EvalDataset(Dataset):
    def __init__(self, samples, root, crop, max_side, degrade):
        self.samples, self.root, self.crop, self.max_side, self.degrade = samples, Path(root), crop, max_side, degrade

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = self.degrade(resize_longest(load_rgb(self.root / s.path), self.max_side))
        return to_tensor(random_crop(img, self.crop, train=False)), torch.tensor(float(s.label)), i


def breakdown(samples, probs, labels, thr, key):
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[getattr(s, key)].append(i)
    out = {}
    for name, idx in sorted(groups.items()):
        idx = np.array(idx)
        out[name] = compute_metrics(probs[idx], labels[idx], thr)
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=None, help="defaults to the one used for training")
    ap.add_argument("--split", default="test")
    ap.add_argument("--degradations", nargs="*", default=list(DEGRADATIONS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    det = load_detector(args.ckpt, device)
    cfg = det.cfg
    samples = read_manifest(args.manifest or cfg.data.manifest, args.split, cfg.data.categories)
    if not samples:
        raise SystemExit(f"No samples in split {args.split!r}")

    report = {"checkpoint": args.ckpt, "split": args.split, "threshold": det.threshold,
              "temperature": det.temperature, "results": {}}
    for name in args.degradations:
        ds = EvalDataset(samples, cfg.data.root, cfg.data.crop_size, cfg.data.max_side, DEGRADATIONS[name])
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
        logits, labels, _ = predict_loader(det.model, dl, device, amp=True)
        probs = sigmoid(logits / det.temperature)
        res = {"overall": compute_metrics(probs, labels, det.threshold)}
        if name == "clean":
            res["per_category"] = breakdown(samples, probs, labels, det.threshold, "category")
            res["per_generator"] = breakdown(samples, probs, labels, det.threshold, "generator")
        report["results"][name] = res
        o = res["overall"]
        print(f"{name:<12} AUC={o.get('auc', float('nan')):.4f}  AP={o.get('ap', float('nan')):.4f}  "
              f"real_acc={o['real_acc']:.3f}  fake_acc={o['fake_acc']:.3f}")

    clean = report["results"].get("clean")
    if clean:
        print("\nper generator (clean):")
        for g, m in clean["per_generator"].items():
            acc = m["fake_acc"] if m["n_fake"] else m["real_acc"]
            print(f"  {g:<20} n={m['n']:<6} acc={acc:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
