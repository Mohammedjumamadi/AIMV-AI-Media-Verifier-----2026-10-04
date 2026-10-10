"""End-to-end smoke test on a synthetic dataset (CPU, no downloads).

"Real" images are noisy natural-ish textures; "fake" images are the same kind of
texture passed through a smoothing filter, mimicking the over-smooth spectra of
generators. The test checks the plumbing (manifest -> train -> eval -> predict),
not detection quality.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimv.augment import TrainAugment, corner_mark, recapture  # noqa: E402
from aimv.config import AugmentConfig, load_config  # noqa: E402
from aimv.metrics import compute_metrics, threshold_at_fpr  # noqa: E402


def make_image(rng: np.random.Generator, fake: bool, size=(160, 120)) -> Image.Image:
    base = rng.normal(128, 40, (size[1] // 8, size[0] // 8, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base).resize(size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 12, (size[1], size[0], 3))
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(2.0)) if fake else img


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    rng = np.random.default_rng(0)
    for i in range(40):
        real = make_image(rng, False)
        p = root / "raw/real/receipt/bank_a" / f"r{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        real.save(p, quality=92)
        fake = make_image(rng, True)
        gen = "gemini" if i % 2 else "chatgpt"
        q = root / f"raw/fake/receipt/{gen}" / f"r{i}__edit.png"
        q.parent.mkdir(parents=True, exist_ok=True)
        fake.save(q)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_manifest.py"), "--root", str(root / "raw"),
                    "--out", str(root / "manifest.csv"), "--val", "0.25", "--test", "0.25", "--no-dedup"],
                   check=True, cwd=root)
    return root


def test_manifest_groups_never_split(dataset):
    rows = list(csv.DictReader(open(dataset / "manifest.csv")))
    assert len(rows) == 80
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values()), "real/fake pair leaked across splits"
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_augment_shapes():
    rng = np.random.default_rng(1)
    img = make_image(rng, False, (300, 200))
    aug = TrainAugment(AugmentConfig(recapture=1.0, corner_mark=1.0, blur=1.0, noise=1.0), 128)
    out = aug(img)
    assert out.size == (128, 128) and out.mode == "RGB"
    assert recapture(img).size == img.size and corner_mark(img).size == img.size
    assert aug(make_image(rng, False, (40, 30))).size == (128, 128)  # small images are padded


def test_threshold_respects_fpr():
    rng = np.random.default_rng(2)
    probs = np.concatenate([rng.uniform(0, 0.6, 1000), rng.uniform(0.4, 1, 1000)])
    labels = np.array([0] * 1000 + [1] * 1000)
    thr = threshold_at_fpr(probs, labels, 0.01)
    m = compute_metrics(probs, labels, thr)
    assert 1 - m["real_acc"] <= 0.01 + 1e-9
    assert 0.9 < m["auc"] <= 1.0


def test_config_overrides():
    cfg = load_config(ROOT / "configs/documents.yaml", ["train.epochs=3", "augment.jpeg_quality=[50, 90]"])
    assert cfg.train.epochs == 3 and cfg.augment.jpeg_quality == (50, 90) and cfg.augment.hflip == 0.0


def test_train_evaluate_predict(dataset, tmp_path):
    from aimv import evaluate, predict
    from aimv.train import train

    cfg = load_config(ROOT / "configs/debug_cpu.yaml", [
        f"data.manifest='{dataset / 'manifest.csv'}'", "data.root='.'",
        f"train.output_dir='{tmp_path / 'run'}'", "train.epochs=2",
    ])
    summary = train(cfg)
    ckpt = Path(summary["checkpoint"])
    assert ckpt.exists()

    report = evaluate.main(["--ckpt", str(ckpt), "--split", "test", "--workers", "0",
                            "--degradations", "clean", "whatsapp", "--out", str(tmp_path / "eval.json")])
    assert set(report["results"]) == {"clean", "whatsapp"}
    assert "gemini" in report["results"]["clean"]["per_generator"]

    big = make_image(np.random.default_rng(3), True, (400, 300))
    big.save(tmp_path / "big.jpg")
    res = predict.main(["--ckpt", str(ckpt), str(ckpt), "--input", str(tmp_path / "big.jpg"),
                        "--out", str(tmp_path / "pred.json"), "--heatmaps", str(tmp_path / "hm")])
    assert len(res) == 1 and 0.0 <= res[0]["prob_fake"] <= 1.0
    assert res[0]["models"][next(iter(res[0]["models"]))]["n_tiles"] > 1
    assert list((tmp_path / "hm").glob("*.jpg"))
    json.loads((tmp_path / "pred.json").read_text())


def test_server_analyze(dataset, tmp_path):
    import io

    from fastapi.testclient import TestClient

    from aimv.analyze import guess_kind, load_analyzer
    from aimv.server import create_app
    from aimv.train import train

    cfg = load_config(ROOT / "configs/debug_cpu.yaml", [
        f"data.manifest='{dataset / 'manifest.csv'}'", f"train.output_dir='{tmp_path / 'run'}'", "train.epochs=1",
    ])
    ckpt = train(cfg)["checkpoint"]
    import torch
    analyzer = load_analyzer([ckpt], [ckpt], torch.device("cpu"), max_tiles=16)
    client = TestClient(create_app(analyzer))

    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"

    buf = io.BytesIO()
    make_image(np.random.default_rng(5), True, (500, 380)).save(buf, format="JPEG")
    for kind in ("auto", "document", "photo"):
        r = client.post("/api/analyze", files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")}, data={"kind": kind})
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0 <= body["prob_fake"] <= 1 and body["explanation"] and body["annotated_image"].startswith("data:image/jpeg")
        assert body["kind"] in ("photo", "document")
        if kind != "auto":
            assert body["kind"] == kind

    bad = client.post("/api/analyze", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert bad.status_code == 400

    paper = Image.new("RGB", (400, 600), (245, 245, 240))
    from PIL import ImageDraw
    d = ImageDraw.Draw(paper)
    for y in range(40, 560, 22):
        d.text((30, y), "AMOUNT : 1970.000 LYD  15/09/2023", fill=(20, 20, 20))
    assert guess_kind(paper) == "document"
    assert guess_kind(make_image(np.random.default_rng(6), False, (300, 300))) == "photo"
