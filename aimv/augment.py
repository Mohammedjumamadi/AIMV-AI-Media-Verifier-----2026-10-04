"""Augmentations that simulate how media really reaches a verifier.

Images arriving from WhatsApp, Facebook, Telegram or a phone camera have been
re-compressed, resized, screenshotted or photographed off a screen. A detector
trained on pristine PNGs collapses on such inputs, so every degradation here is
applied to *both* classes with the same probability: the model must learn
generator artifacts, not file-format or pipeline differences.
"""

from __future__ import annotations

import io
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .config import AugmentConfig

SOCIAL_MAX_SIDES = (720, 1080, 1280, 1600, 2048)  # WhatsApp / FB / Telegram / IG sizes


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling=random.choice((0, 2)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_longest(img: Image.Image, side: int) -> Image.Image:
    w, h = img.size
    scale = side / max(w, h)
    if scale >= 1:
        return img
    resample = random.choice((Image.BILINEAR, Image.BICUBIC, Image.LANCZOS))
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample)


def rescale(img: Image.Image, lo: float = 0.6, hi: float = 1.4) -> Image.Image:
    s = random.uniform(lo, hi)
    w, h = img.size
    resample = random.choice((Image.BILINEAR, Image.BICUBIC, Image.LANCZOS))
    return img.resize((max(8, round(w * s)), max(8, round(h * s))), resample)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def color_jitter(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))


def recapture(img: Image.Image) -> Image.Image:
    """Rough simulation of photographing a screen/print: perspective, blur, noise."""
    w, h = img.size
    dx, dy = 0.04 * w, 0.04 * h
    j = lambda d: random.uniform(-d, d)  # noqa: E731
    # Source quadrilateral corners: upper-left, lower-left, lower-right, upper-right.
    quad = (
        j(dx), j(dy),
        j(dx), h + j(dy),
        w + j(dx), h + j(dy),
        w + j(dx), j(dy),
    )
    img = img.transform((w, h), Image.QUAD, quad, Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    img = gaussian_noise(img, random.uniform(2, 6))
    return jpeg(img, random.randint(60, 90))


def corner_mark(img: Image.Image) -> Image.Image:
    """Paste a watermark-like sparkle/logo in a corner.

    Gemini, Meta AI, etc. stamp a visible logo on generated images, and people
    crop it out or paste it on real photos. Adding similar marks to both classes
    stops the network from using the logo as a shortcut.
    """
    img = img.copy()
    w, h = img.size
    size = max(8, int(min(w, h) * random.uniform(0.03, 0.08)))
    margin = int(size * random.uniform(0.5, 1.5))
    cx = random.choice((margin + size, w - margin - size))
    cy = random.choice((margin + size, h - margin - size))
    draw = ImageDraw.Draw(img, "RGBA")
    alpha = random.randint(120, 230)
    color = (255, 255, 255, alpha) if random.random() < 0.7 else (0, 0, 0, alpha)
    if random.random() < 0.6:  # 4-point sparkle
        r_out, r_in = size, size * 0.18
        pts = []
        for i in range(8):
            ang = math.pi / 4 * i - math.pi / 2
            r = r_out if i % 2 == 0 else r_in
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        draw.polygon(pts, fill=color)
    elif random.random() < 0.5:
        draw.ellipse((cx - size, cy - size // 2, cx + size, cy + size // 2), fill=color)
    else:
        draw.text((cx - size, cy - size // 2), random.choice(("AI", "Imagine", "Meta AI", "★")), fill=color)
    return img


def random_crop(img: Image.Image, size: int, train: bool) -> Image.Image:
    """Crop at native resolution; pad (reflect) if the image is smaller than the crop."""
    w, h = img.size
    if w < size or h < size:
        arr = np.asarray(img)
        pad_h, pad_w = max(0, size - h), max(0, size - w)
        mode = "reflect" if pad_h < h and pad_w < w else "edge"
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        img = Image.fromarray(arr)
        w, h = img.size
    if train:
        x, y = random.randint(0, w - size), random.randint(0, h - size)
    else:
        x, y = (w - size) // 2, (h - size) // 2
    return img.crop((x, y, x + size, y + size))


class TrainAugment:
    def __init__(self, cfg: AugmentConfig, crop_size: int):
        self.cfg = cfg
        self.crop_size = crop_size

    def __call__(self, img: Image.Image) -> Image.Image:
        c, p = self.cfg, random.random
        if p() < c.corner_mark:
            img = corner_mark(img)
        if p() < c.social_resize:
            img = resize_longest(img, random.choice(SOCIAL_MAX_SIDES))
        if p() < c.rescale:
            img = rescale(img)
        if p() < c.color_jitter:
            img = color_jitter(img)
        if p() < c.blur:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.2)))
        if p() < c.noise:
            img = gaussian_noise(img, random.uniform(1, 8))
        if p() < c.recapture:
            img = recapture(img)
        if p() < c.double_jpeg:
            img = jpeg(img, random.randint(*c.jpeg_quality))
        if p() < c.format_normalize:
            img = jpeg(img, random.randint(*c.jpeg_quality))
        img = random_crop(img, self.crop_size, train=True)
        if p() < c.hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
