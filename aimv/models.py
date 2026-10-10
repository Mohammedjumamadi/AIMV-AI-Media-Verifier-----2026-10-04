"""Detector models.

Two complementary families, both built on `timm` backbones:
- artifact models (CNN on native-resolution crops + SRM noise residuals) catch
  low-level generator fingerprints and local edits (changed digits on a receipt);
- semantic models (CLIP / DINOv2 features, frozen or partially tuned) generalise
  better to generators never seen in training.
`predict.py` ensembles them.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .config import ModelConfig

# Classic SRM high-pass kernels used in image forensics (Fridrich & Kodovsky).
_SRM = [
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
]
_SRM_SCALE = (4.0, 12.0, 2.0)


class HighPass(nn.Module):
    """Fixed SRM filters on luminance -> 3 noise-residual channels."""

    def __init__(self):
        super().__init__()
        k = torch.tensor(_SRM, dtype=torch.float32) / torch.tensor(_SRM_SCALE).view(3, 1, 1)
        self.register_buffer("kernel", k.unsqueeze(1))  # (3,1,5,5)
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x * self.luma).sum(1, keepdim=True)
        r = nn.functional.conv2d(y, self.kernel, padding=2)
        return torch.clamp(r, -3, 3)


class Detector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.highpass = HighPass() if cfg.highpass else None
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            in_chans=6 if cfg.highpass else 3,
            drop_path_rate=cfg.drop_path_rate if not cfg.freeze_backbone else 0.0,
        )
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(cfg.drop_rate),
            nn.Linear(dim, 1),
        )
        if cfg.freeze_backbone:
            self._freeze(cfg.unfreeze_last_n)

    def _freeze(self, last_n: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = getattr(self.backbone, "blocks", None) or getattr(self.backbone, "stages", None)
        if last_n > 0 and blocks is not None:
            for block in list(blocks)[-last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for name in ("norm", "fc_norm", "head_norm"):
                mod = getattr(self.backbone, name, None)
                if isinstance(mod, nn.Module):
                    for p in mod.parameters():
                        p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.highpass is not None:
            x = torch.cat([x, self.highpass(x)], dim=1)
        return self.head(self.backbone(x)).squeeze(1)

    def param_groups(self, lr: float, backbone_lr_mult: float, weight_decay: float):
        decay, no_decay, head = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("head."):
                head.append(p)
            elif p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr * backbone_lr_mult, "weight_decay": 0.0},
            {"params": head, "lr": lr, "weight_decay": weight_decay},
        ]
        return [g for g in groups if g["params"]]


def build_model(cfg: ModelConfig) -> Detector:
    return Detector(cfg)
