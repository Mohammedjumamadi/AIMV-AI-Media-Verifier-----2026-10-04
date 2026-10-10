"""One-call analysis of an uploaded image: verdict, detailed Arabic explanation,
suspicious regions drawn on the image, and metadata evidence.

Models are loaded once and kept warm, so a request costs only the forward passes.
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .data import to_rgb
from .forensics import inspect_bytes
from .inference import LoadedDetector, ensemble, load_detector, score_image

KIND_AR = {"photo": "صورة (أشخاص / مشاهد)", "document": "مستند / فاتورة / إيصال"}


def open_upright(data: bytes) -> Image.Image:
    """Decode and apply the EXIF orientation (phone photos are often stored rotated)."""
    with Image.open(io.BytesIO(data)) as im:
        return to_rgb(ImageOps.exif_transpose(im))


def guess_kind(img: Image.Image) -> str:
    """Cheap heuristic: a document is mostly bright, unsaturated paper carrying thin dark strokes (text).

    Users can always override it in the UI; the label says it was an automatic guess.
    """
    small = img.copy()
    small.thumbnail((768, 768))
    gray = small.convert("L")
    a = np.asarray(gray, dtype=np.int16)
    local = np.asarray(gray.filter(ImageFilter.BoxBlur(12)), dtype=np.int16)
    sat = np.asarray(small.convert("HSV"), dtype=np.int16)[..., 1]
    paper = ((sat < 50) & (a > 130)).mean()
    ink = ((a < local - 35) & (local > 120)).mean()  # dark strokes on a bright neighbourhood
    return "document" if paper > 0.65 and ink > 0.005 else "photo"


def region_name(box, w: int, h: int) -> str:
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    row = "أعلى" if cy < 1 / 3 else "وسط" if cy < 2 / 3 else "أسفل"
    col = "اليسار" if cx < 1 / 3 else "الوسط" if cx < 2 / 3 else "اليمين"
    return "منتصف الصورة" if row == "وسط" and col == "الوسط" else f"{row} {col}"


def confidence_level(prob: float, thr: float) -> str:
    d = prob - thr if prob >= thr else thr - prob
    span = (1 - thr) if prob >= thr else thr
    r = d / max(span, 1e-6)
    return "عالية" if r > 0.6 else "متوسطة" if r > 0.25 else "منخفضة"


def suspicious_tiles(results: list[dict], thr: float, limit: int = 6) -> list[dict]:
    """Average tile probabilities across models that share the same tiling, keep the worst ones."""
    tiles = []
    for r in results:
        for t in r["tiles"]:
            if t["prob"] >= thr:
                tiles.append(t)
    tiles.sort(key=lambda t: -t["prob"])
    kept = []
    for t in tiles:  # drop heavily overlapping duplicates
        if all(_iou(t["box"], k["box"]) < 0.3 for k in kept):
            kept.append(t)
        if len(kept) >= limit:
            break
    return kept


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def annotate(img: Image.Image, regions: list[dict]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    lw = max(2, min(img.size) // 200)
    for i, r in enumerate(regions, 1):
        x0, y0, x1, y1 = r["box"]
        draw.rectangle((x0, y0, x1, y1), outline=(230, 30, 30, 255), width=lw, fill=(230, 30, 30, 40))
        label = f"{i}: {r['prob'] * 100:.0f}%"
        tw = 9 * len(label)
        draw.rectangle((x0, y0, x0 + tw, y0 + 16), fill=(230, 30, 30, 230))
        draw.text((x0 + 3, y0 + 2), label, fill=(255, 255, 255, 255))
    return out


def explain(kind: str, combined: dict, per_model: dict[str, dict], regions, meta: dict,
            size: tuple[int, int]) -> list[str]:
    prob, thr = combined["prob_fake"], combined["threshold"]
    lines = []
    if combined["is_fake"]:
        lines.append(f"النتيجة: مزيّفة أو معدّلة بالذكاء الاصطناعي، بنسبة {prob * 100:.1f}% "
                     f"(الثقة {confidence_level(prob, thr)}).")
    elif combined["verdict"].startswith("uncertain"):
        lines.append(f"النتيجة: غير محسومة ({prob * 100:.1f}%). تحتاج مراجعة بشرية.")
    else:
        lines.append(f"النتيجة: تبدو حقيقية، واحتمال التزييف {prob * 100:.1f}% "
                     f"(الثقة {confidence_level(prob, thr)}).")

    means = [r["mean_tile_prob"] for r in per_model.values()]
    tops = [r["topk_tile_prob"] for r in per_model.values()]
    mean, top = float(np.mean(means)), float(np.mean(tops))
    if combined["is_fake"]:
        if top - mean > 0.25 and regions:
            lines.append("نوع التزييف: تعديل موضعي. أغلب الصورة سليم، لكن أجزاء محددة منها معدّلة "
                         + ("(مثل تغيير المبلغ أو التاريخ أو الاسم)." if kind == "document"
                            else "(مثل تغيير وجه أو إضافة عنصر)."))
        else:
            lines.append("نوع التزييف: الصورة كلها تبدو مولّدة بالذكاء الاصطناعي، لا معدّلة جزئياً.")
    if regions:
        w, h = size
        where = "، ".join(f"({i}) {region_name(r['box'], w, h)} {r['prob'] * 100:.0f}%"
                         for i, r in enumerate(regions, 1))
        lines.append(f"المناطق المشبوهة (مؤطّرة بالأحمر): {where}.")

    if len(per_model) > 1:
        flagged = sum(r["is_fake"] for r in per_model.values())
        lines.append(f"اتفاق النماذج: {flagged} من {len(per_model)} نماذج تعتبرها مزيّفة.")

    if meta.get("markers"):
        lines.append("البيانات الوصفية تحتوي على علامات ذكاء اصطناعي أو مصدر: " + "؛ ".join(meta["markers"]) + ".")
    elif meta.get("camera"):
        lines.append(f"البيانات الوصفية فيها اسم كاميرا ({meta['camera']})، وهي قرينة ضعيفة على أن الصورة حقيقية.")
    else:
        lines.append("لا توجد بيانات وصفية. هذا طبيعي في الصور المرسلة عبر واتساب أو فيسبوك.")
    if meta.get("software"):
        lines.append(f"برنامج المعالجة المسجّل: {meta['software']}.")
    lines.append("تنبيه: النتيجة احتمالية. في القرارات المهمة يجب أن يراجعها إنسان.")
    return lines


class Analyzer:
    def __init__(self, groups: dict[str, list[str]], device: torch.device, max_tiles: int = 32,
                 tta: bool = False):
        self.device, self.max_tiles, self.tta = device, max_tiles, tta
        self.groups: dict[str, list[LoadedDetector]] = {
            kind: [load_detector(p, device) for p in paths] for kind, paths in groups.items() if paths
        }
        if not self.groups:
            raise ValueError("at least one checkpoint is required")
        self._lock = threading.Lock()
        self.warmup()

    def warmup(self) -> None:
        dummy = Image.new("RGB", (64, 64), (128, 128, 128))
        for dets in self.groups.values():
            for d in dets:
                score_image(d, dummy, tta=False, max_tiles=1)

    def resolve_kind(self, kind: str, img: Image.Image) -> tuple[str, bool]:
        if kind not in ("photo", "document"):
            kind = guess_kind(img)
            auto = True
        else:
            auto = False
        if kind not in self.groups:  # fall back to whatever is loaded
            kind = next(iter(self.groups))
        return kind, auto

    def analyze(self, data: bytes, kind: str = "auto") -> dict:
        t0 = time.perf_counter()
        img = open_upright(data)
        kind, auto = self.resolve_kind(kind, img)
        with self._lock:
            per_model = {}
            for i, det in enumerate(self.groups[kind]):
                name = det.name if det.name not in per_model else f"{det.name}_{i}"
                per_model[name] = score_image(det, img, tta=self.tta, max_tiles=self.max_tiles)
        combined = ensemble(list(per_model.values()))
        regions = suspicious_tiles(list(per_model.values()), combined["threshold"]) if combined["is_fake"] else []
        meta = inspect_bytes(data)
        return {
            **combined,
            "kind": kind,
            "kind_ar": KIND_AR[kind] + (" (تقدير تلقائي)" if auto else ""),
            "explanation": explain(kind, combined, per_model, regions, meta, img.size),
            "regions": regions,
            "models": {k: {"prob_fake": r["prob_fake"], "threshold": r["threshold"], "is_fake": bool(r["is_fake"]),
                           "mean_tile_prob": r["mean_tile_prob"], "topk_tile_prob": r["topk_tile_prob"],
                           "n_tiles": r["n_tiles"]} for k, r in per_model.items()},
            "metadata": meta,
            "image_size": list(img.size),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            "_annotated": annotate(img, regions),
        }


def load_analyzer(photo: list[str] | None, document: list[str] | None, device: torch.device,
                  max_tiles: int = 32, tta: bool = False) -> Analyzer:
    for p in (photo or []) + (document or []):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    return Analyzer({"photo": photo or [], "document": document or []}, device, max_tiles, tta)
