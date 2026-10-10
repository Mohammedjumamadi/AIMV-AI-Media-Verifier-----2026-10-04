"""Web app: upload an image -> verdict + detailed explanation in seconds.

    python -m aimv.server --photo-ckpt runs/photos_clip/best.pt runs/photos_artifact/best.pt \
        --doc-ckpt runs/documents/best.pt --port 8000
    # open http://localhost:8000

API:  POST /api/analyze  (multipart: file=<image>, kind=auto|photo|document)
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import UnidentifiedImageError

from .analyze import Analyzer, load_analyzer
from .train import pick_device

MAX_UPLOAD_MB = 25
STATIC = Path(__file__).parent / "static"


def create_app(analyzer: Analyzer) -> FastAPI:
    app = FastAPI(title="AIMV — AI Media Verifier")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "models": {k: [d.name for d in v] for k, v in analyzer.groups.items()},
                "device": str(analyzer.device)}

    @app.post("/api/analyze")
    def analyze(file: UploadFile = File(...), kind: str = Form("auto")) -> dict:
        data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"الملف أكبر من {MAX_UPLOAD_MB} ميغابايت")
        try:
            report = analyzer.analyze(data, kind)
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(400, "الملف ليس صورة صالحة") from e
        annotated = report.pop("_annotated")
        annotated.thumbnail((1600, 1600))
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=85)
        report["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        report["filename"] = file.filename
        return report

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo-ckpt", nargs="*", default=[])
    ap.add_argument("--doc-ckpt", nargs="*", default=[])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-tiles", type=int, default=32, help="fewer = faster, more = finer localisation")
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation (2x slower)")
    args = ap.parse_args(argv)
    analyzer = load_analyzer(args.photo_ckpt, args.doc_ckpt, pick_device(args.device), args.max_tiles, args.tta)
    uvicorn.run(create_app(analyzer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
