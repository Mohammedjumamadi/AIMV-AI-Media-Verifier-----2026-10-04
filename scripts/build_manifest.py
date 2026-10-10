"""Scan a dataset folder and write data/manifest.csv with leak-free splits.

Expected layout (category and generator names are free-form):

    data/raw/real/<category>/[<source>/]...        e.g. real/receipt/moamalat/IMG_001.jpg
    data/raw/fake/<category>/<generator>/...       e.g. fake/receipt/gemini/IMG_001__amount.jpg

Pairing: a fake made FROM a real file should be named `<real_stem>__<anything>`
(IMG_001__amount.jpg). Both then share a group and always land in the same split.
Near-duplicates (same picture re-sent on WhatsApp, cropped, ...) are merged into
one group with a perceptual hash, so the test set never contains a copy of a
training image.

    python scripts/build_manifest.py --root data/raw --out data/manifest.csv \
        --val 0.1 --test 0.1 --holdout-generators grok
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aimv.data import Sample, describe, write_manifest  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def dhash(path: Path, size: int = 8) -> int | None:
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    except Exception:  # noqa: BLE001
        return None
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def scan(root: Path) -> list[Sample]:
    samples = []
    for label_name, label in (("real", 0), ("fake", 1)):
        base = root / label_name
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS or p.name.startswith("."):
                continue
            parts = p.relative_to(base).parts
            category = parts[0] if len(parts) > 1 else "photo"
            generator = parts[1] if len(parts) > 2 else ("camera" if label == 0 else "unknown")
            stem = p.stem.split("__")[0]
            samples.append(Sample(str(p), label, category, generator, f"{category}/{stem}", ""))
    return samples


def merge_near_duplicates(samples: list[Sample], max_dist: int) -> None:
    uf = UnionFind()
    hashes = {}
    for s in samples:
        uf.find(s.group)
        h = dhash(Path(s.path))
        if h is not None:
            hashes[s.path] = (h, s.group)
    # LSH: 4 bands of 16 bits; candidates share at least one band exactly.
    buckets = defaultdict(list)
    for path, (h, _) in hashes.items():
        for band in range(4):
            buckets[(band, (h >> (16 * band)) & 0xFFFF)].append(path)
    for members in buckets.values():
        if len(members) > 500:  # degenerate (blank) images: skip
            continue
        for i in range(len(members)):
            hi, gi = hashes[members[i]]
            for j in range(i + 1, len(members)):
                hj, gj = hashes[members[j]]
                if gi != gj and bin(hi ^ hj).count("1") <= max_dist:
                    uf.union(gi, gj)
    for s in samples:
        s.group = uf.find(s.group)


def assign_splits(samples, val: float, test: float, holdout: set[str], seed: int) -> None:
    def frac(group: str) -> float:
        h = hashlib.sha1(f"{seed}:{group}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    holdout_groups = {s.group for s in samples if s.generator in holdout}
    for s in samples:
        if s.group in holdout_groups:
            s.split = "test"  # unseen-generator test: whole group goes to test
            continue
        f = frac(s.group)
        s.split = "test" if f < test else "val" if f < test + val else "train"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--holdout-generators", nargs="*", default=[],
                    help="generators kept entirely for test (measures generalisation to unseen AI tools)")
    ap.add_argument("--dup-dist", type=int, default=4, help="max dHash Hamming distance for near-duplicates")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = scan(Path(args.root))
    if not samples:
        raise SystemExit(f"No images found under {args.root}/real or {args.root}/fake")
    if not args.no_dedup:
        merge_near_duplicates(samples, args.dup_dist)
    assign_splits(samples, args.val, args.test, set(args.holdout_generators), args.seed)
    write_manifest(args.out, samples)
    print(describe(samples))
    print("\ngenerators:", dict(Counter(s.generator for s in samples)))
    print(f"\nwrote {len(samples)} rows to {args.out}  (paths relative to the current directory: "
          f"set data.root='.')")


if __name__ == "__main__":
    main()
