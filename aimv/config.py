"""Experiment configuration: typed dataclasses loaded from YAML with CLI overrides."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest: str = "data/manifest.csv"
    root: str = "."                      # paths in the manifest are relative to this
    categories: list[str] | None = None  # None = all categories in the manifest
    crop_size: int = 256                 # training crop at native resolution (no resize)
    num_workers: int = 4
    balance: bool = True                 # balanced sampling over (label, category)
    max_side: int = 4096                 # downscale giant images before cropping


@dataclass
class AugmentConfig:
    format_normalize: float = 1.0   # re-encode every sample as JPEG -> kills PNG-vs-JPEG shortcut
    jpeg_quality: tuple[int, int] = (45, 100)
    double_jpeg: float = 0.3        # WhatsApp/Facebook style re-compression chain
    social_resize: float = 0.3      # resize longest side to a social-media size
    rescale: float = 0.2            # random rescale 0.6-1.4
    blur: float = 0.1
    noise: float = 0.1
    color_jitter: float = 0.2
    recapture: float = 0.05         # photo-of-screen / photo-of-print simulation
    corner_mark: float = 0.15       # paste watermark-like marks on BOTH classes (anti-shortcut)
    hflip: float = 0.5              # set 0 for documents (text direction)


@dataclass
class ModelConfig:
    backbone: str = "convnext_small.fb_in22k_ft_in1k"
    pretrained: bool = True
    highpass: bool = True            # add fixed SRM noise-residual channels
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1
    freeze_backbone: bool = False    # linear probe (e.g. CLIP / DINOv2 features)
    unfreeze_last_n: int = 0         # with freeze_backbone: train only the last N blocks


@dataclass
class TrainConfig:
    output_dir: str = "runs/exp"
    epochs: int = 20
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-4                 # head LR
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.05
    ema_decay: float = 0.999         # 0 disables EMA
    amp: bool = True
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    seed: int = 42
    target_fpr: float = 0.01         # operating point: <=1% of real media flagged as fake
    max_steps_per_epoch: int = 0     # 0 = full epoch (useful for smoke tests)
    device: str = "auto"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, values: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value or {})
        elif isinstance(default, tuple) and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(d: dict[str, Any]) -> Config:
    return _build(Config, d)


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    """Load YAML and apply `section.key=value` overrides (values parsed as Python literals)."""
    raw: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        section, _, name = key.partition(".")
        if not name:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        raw.setdefault(section, {})[name] = parsed
    return config_from_dict(raw)
