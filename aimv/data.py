"""Manifest-driven datasets.

The manifest is a CSV with columns:
    path, label, category, generator, group, split
- label:     0 = real, 1 = fake (AI-generated or AI-edited)
- category:  photo / receipt / id_card / certificate / screenshot / ...
- generator: camera, gemini, chatgpt, flux, midjourney, copymove, ...
- group:     items sharing a group (a real receipt and its AI-edited copies)
             always land in the same split, so the test set never leaks.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, WeightedRandomSampler

from .augment import TrainAugment, random_crop, resize_longest
from .config import Config

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 200_000_000

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MANIFEST_FIELDS = ("path", "label", "category", "generator", "group", "split")


@dataclass
class Sample:
    path: str
    label: int
    category: str
    generator: str
    group: str
    split: str


def read_manifest(path: str | Path, split: str | None = None,
                  categories: list[str] | None = None) -> list[Sample]:
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = Sample(row["path"], int(row["label"]), row.get("category", "photo"),
                       row.get("generator", "unknown"), row.get("group", row["path"]),
                       row.get("split", "train"))
            if split and s.split != split:
                continue
            if categories and s.category not in categories:
                continue
            samples.append(s)
    return samples


def write_manifest(path: str | Path, samples: list[Sample]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for s in samples:
            w.writerow({k: getattr(s, k) for k in MANIFEST_FIELDS})


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return im.convert("RGB")


def to_tensor(img: Image.Image) -> torch.Tensor:
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float().div_(255)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (t - mean) / std


class MediaDataset(Dataset):
    def __init__(self, samples: list[Sample], cfg: Config, train: bool):
        self.samples = samples
        self.root = Path(cfg.data.root)
        self.crop = cfg.data.crop_size
        self.max_side = cfg.data.max_side
        self.train = train
        self.augment = TrainAugment(cfg.augment, self.crop) if train else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = resize_longest(load_rgb(self.root / s.path), self.max_side)
        img = self.augment(img) if self.train else random_crop(img, self.crop, train=False)
        return to_tensor(img), torch.tensor(float(s.label)), idx


def balanced_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    """Every (label, category) cell gets equal total weight per epoch."""
    keys = [(s.label, s.category) for s in samples]
    counts = Counter(keys)
    weights = torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def describe(samples: list[Sample]) -> str:
    c = Counter((s.split, s.category, "fake" if s.label else "real") for s in samples)
    lines = [f"{sp:>5} | {cat:<14} | {lab:<4} | {n}" for (sp, cat, lab), n in sorted(c.items())]
    return "\n".join(lines)
