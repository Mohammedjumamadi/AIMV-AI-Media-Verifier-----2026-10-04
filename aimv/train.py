"""Train a real-vs-AI detector.

    python -m aimv.train --config configs/photos_artifact.yaml
    python -m aimv.train --config configs/documents.yaml --set train.epochs=30
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, load_config
from .data import MediaDataset, balanced_sampler, describe, read_manifest
from .metrics import compute_metrics, fit_temperature, sigmoid, threshold_at_fpr
from .models import build_model


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2**32)


class EMA:
    """Exponential moving average of weights; usually 0.5-1 AUC point better and more stable."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))  # warm-up for short runs
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


@torch.no_grad()
def predict_loader(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits, labels, idxs = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16 if device.type != "cuda" else torch.float16,
                            enabled=amp and device.type == "cuda"):
            out = model(x)
        logits.append(out.float().cpu())
        labels.append(y)
        idxs.append(idx)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), torch.cat(idxs).numpy()


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train(cfg: Config) -> dict:
    seed_everything(cfg.train.seed)
    device = pick_device(cfg.train.device)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))

    cats = cfg.data.categories
    train_s = read_manifest(cfg.data.manifest, "train", cats)
    val_s = read_manifest(cfg.data.manifest, "val", cats)
    if not train_s or not val_s:
        raise SystemExit("Manifest needs non-empty 'train' and 'val' splits (see scripts/build_manifest.py).")
    print(f"device={device}  train={len(train_s)}  val={len(val_s)}\n{describe(train_s + val_s)}")

    pin = device.type == "cuda"
    train_dl = DataLoader(
        MediaDataset(train_s, cfg, train=True),
        batch_size=cfg.train.batch_size,
        sampler=balanced_sampler(train_s) if cfg.data.balance else None,
        shuffle=not cfg.data.balance,
        num_workers=cfg.data.num_workers, pin_memory=pin, drop_last=len(train_s) > cfg.train.batch_size,
        worker_init_fn=worker_init, persistent_workers=cfg.data.num_workers > 0,
    )
    val_dl = DataLoader(MediaDataset(val_s, cfg, train=False), batch_size=cfg.train.batch_size,
                        num_workers=cfg.data.num_workers, pin_memory=pin)

    model = build_model(cfg.model).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={cfg.model.backbone} trainable_params={n_train / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.param_groups(cfg.train.lr, cfg.train.backbone_lr_mult,
                                               cfg.train.weight_decay))
    steps_per_epoch = len(train_dl) // cfg.train.grad_accum
    if cfg.train.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.train.max_steps_per_epoch)
    steps_per_epoch = max(1, steps_per_epoch)
    total = steps_per_epoch * cfg.train.epochs
    warmup = int(steps_per_epoch * cfg.train.warmup_epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup, total))
    use_amp = cfg.train.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    ema = EMA(model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
    eps = cfg.train.label_smoothing

    best_auc, best_epoch, history = -1.0, -1, []
    for epoch in range(cfg.train.epochs):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_dl, total=steps_per_epoch * cfg.train.grad_accum, desc=f"epoch {epoch + 1}",
                   leave=False)
        for i, (x, y, _) in enumerate(bar):
            if i >= steps_per_epoch * cfg.train.grad_accum:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            target = y * (1 - eps) + 0.5 * eps
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), target)
            scaler.scale(loss / cfg.train.grad_accum).backward()
            if (i + 1) % cfg.train.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema:
                    ema.update(model)
            running += loss.item() * len(x)
            seen += len(x)
            bar.set_postfix(loss=f"{running / seen:.4f}")

        eval_model = ema.module if ema else model
        logits, labels, _ = predict_loader(eval_model, val_dl, device, cfg.train.amp)
        m = compute_metrics(sigmoid(logits), labels)
        rec = {"epoch": epoch + 1, "train_loss": running / max(1, seen), "time_s": round(time.time() - t0, 1),
               **{f"val_{k}": v for k, v in m.items() if k not in ("n", "n_real", "n_fake", "threshold")}}
        history.append(rec)
        print(json.dumps(rec))
        auc = m.get("auc", m["balanced_acc"])
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch + 1
            save_checkpoint(out_dir / "best.pt", eval_model, cfg, logits, labels, rec)
        elif epoch + 1 - best_epoch >= cfg.train.early_stop_patience:
            print(f"early stop: no val improvement for {cfg.train.early_stop_patience} epochs")
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    summary = {"best_epoch": best_epoch, "best_val_auc": best_auc, "checkpoint": str(out_dir / "best.pt")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return summary


def save_checkpoint(path: Path, model, cfg: Config, val_logits, val_labels, record: dict) -> None:
    temperature = fit_temperature(val_logits, val_labels)
    probs = sigmoid(val_logits / temperature)
    thr = threshold_at_fpr(probs, val_labels, cfg.train.target_fpr)
    torch.save({
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "temperature": temperature,
        "threshold": thr,                    # calibrated prob threshold at target FPR
        "val_metrics": compute_metrics(probs, val_labels, thr),
        "record": record,
    }, path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="section.key=value")
    args = ap.parse_args(argv)
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
