"""Create locally-tampered training negatives from REAL documents you own.

Most forged receipts / certificates are a real scan with one region changed
(amount, date, name). This script builds that kind of training data in bulk with
classic manipulations, each saved with its binary mask:

    copymove  - copy a patch of the same document to another place
    splice    - paste a patch from a different real document
    smoothfill- erase a region and fill it with smoothed surroundings (cheap inpainting)
    resample  - rescale a region in place (re-typed / re-rendered text leaves this trace)

These complement — never replace — real edits made with Gemini, ChatGPT, Photoshop
generative fill, etc. (see docs/LIBYAN_DATA_GUIDE.md).

    python scripts/make_tamper_fakes.py --real data/raw/real/receipt --out data/raw/fake/receipt \
        --masks data/masks/receipt --per-image 2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def rand_box(w: int, h: int, lo: float = 0.05, hi: float = 0.25):
    bw = max(8, int(w * random.uniform(lo, hi)))
    bh = max(8, int(h * random.uniform(lo / 2, hi / 2)))
    x, y = random.randint(0, max(0, w - bw)), random.randint(0, max(0, h - bh))
    return x, y, x + bw, y + bh


def feathered_mask(size, box, feather: float):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather)) if feather > 0 else m


def copymove(img, donors):
    w, h = img.size
    src = rand_box(w, h)
    bw, bh = src[2] - src[0], src[3] - src[1]
    dx, dy = random.randint(0, w - bw), random.randint(0, h - bh)
    dst = (dx, dy, dx + bw, dy + bh)
    patch = Image.new("RGB", img.size)
    patch.paste(img.crop(src), dst[:2])
    return dst, patch


def splice(img, donors):
    donor = random.choice(donors) if donors else img
    w, h = img.size
    dst = rand_box(w, h)
    bw, bh = dst[2] - dst[0], dst[3] - dst[1]
    dw, dh = donor.size
    if dw < bw or dh < bh:
        donor = donor.resize((max(dw, bw), max(dh, bh)))
        dw, dh = donor.size
    sx, sy = random.randint(0, dw - bw), random.randint(0, dh - bh)
    patch = Image.new("RGB", img.size)
    patch.paste(donor.crop((sx, sy, sx + bw, sy + bh)), dst[:2])
    return dst, patch


def smoothfill(img, donors):
    w, h = img.size
    dst = rand_box(w, h, 0.04, 0.15)
    patch = img.filter(ImageFilter.GaussianBlur(random.uniform(6, 15)))
    return dst, patch


def resample(img, donors):
    w, h = img.size
    dst = rand_box(w, h)
    region = img.crop(dst)
    s = random.uniform(0.4, 0.8)
    small = region.resize((max(4, int(region.width * s)), max(4, int(region.height * s))), Image.BILINEAR)
    patch = Image.new("RGB", img.size)
    patch.paste(small.resize(region.size, Image.BICUBIC), dst[:2])
    return dst, patch


OPS = {"copymove": copymove, "splice": splice, "smoothfill": smoothfill, "resample": resample}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="folder of REAL documents of one category")
    ap.add_argument("--out", required=True, help="fake/<category> folder; one subfolder per operation")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--per-image", type=int, default=1)
    ap.add_argument("--ops", nargs="*", default=list(OPS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    paths = [p for p in sorted(Path(args.real).rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise SystemExit(f"no images in {args.real}")
    donor_pool = random.sample(paths, min(50, len(paths)))
    donors = [Image.open(p).convert("RGB") for p in donor_pool]
    n = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        for k in range(args.per_image):
            op = random.choice(args.ops)
            box, patch = OPS[op](img, [d for d, dp in zip(donors, donor_pool) if dp != p])
            mask = feathered_mask(img.size, box, random.choice((0, 0, 1, 2)))
            fake = Image.composite(patch, img, mask)
            out = Path(args.out) / f"synthetic_{op}" / f"{p.stem}__{op}{k}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            fake.save(out, quality=random.randint(80, 97))
            if args.masks:
                mp = Path(args.masks) / f"{p.stem}__{op}{k}.png"
                mp.parent.mkdir(parents=True, exist_ok=True)
                mask.point(lambda v: 255 if v > 127 else 0).save(mp)
            n += 1
    print(f"wrote {n} tampered images to {args.out}")


if __name__ == "__main__":
    main()
