"""Verify images: ensemble of trained detectors + metadata forensics.

    python -m aimv.predict --ckpt runs/photos_clip/best.pt runs/photos_artifact/best.pt \
        --input path/to/image_or_folder --out report.json --heatmaps heatmaps/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .forensics import inspect_file, summarize
from .inference import ensemble, load_detector, load_image, score_image
from .train import pick_device

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".avif"}


def iter_images(path: Path):
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def save_heatmap(img: Image.Image, heat: np.ndarray, path: Path) -> None:
    """Overlay tile probabilities (red = suspicious) on the image."""
    h = np.nan_to_num(heat, nan=0.0)
    hm = Image.fromarray((h * 255).astype(np.uint8), mode="L").resize(img.size, Image.BILINEAR)
    red = Image.new("RGB", img.size, (255, 0, 0))
    overlay = Image.composite(red, img, hm.point(lambda v: int(v * 0.55)))
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path, quality=90)


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more best.pt checkpoints")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--heatmaps", default=None, help="folder for suspicious-region overlays")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    detectors = [load_detector(p, device) for p in args.ckpt]
    for i, det in enumerate(detectors):  # unique names even if two runs share a folder name
        if [d.name for d in detectors].count(det.name) > 1:
            det.name = f"{det.name}_{i}"
    results = []
    for path in iter_images(Path(args.input)):
        try:
            img = load_image(path)
        except Exception as e:  # noqa: BLE001
            results.append({"file": str(path), "error": str(e)})
            continue
        per_model = {}
        for det in detectors:
            r = score_image(det, img, tta=not args.no_tta)
            if args.heatmaps and r["n_tiles"] > 1:
                save_heatmap(img, r["heatmap"], Path(args.heatmaps) / f"{path.stem}__{det.name}.jpg")
            per_model[det.name] = r
        combined = ensemble(list(per_model.values()))
        meta = inspect_file(path)
        rec = {
            "file": str(path),
            **combined,
            "models": {k: {kk: (float(v) if isinstance(v, (float, np.floating)) else v)
                           for kk, v in r.items() if kk != "heatmap"} for k, r in per_model.items()},
            "metadata": meta,
            "metadata_summary": summarize(meta),
        }
        results.append(rec)
        print(f"{rec['prob_fake']:.3f}  {rec['verdict']:<28} {path.name}   [{rec['metadata_summary']}]")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    return results


if __name__ == "__main__":
    main()
