"""Non-learned evidence: metadata and provenance markers.

These are *hints*, never proof: WhatsApp/Facebook strip metadata, and metadata
can be forged. They are reported next to the model score, not fused into it.
"""

from __future__ import annotations

from pathlib import Path

from PIL import ExifTags, Image

# Strings that generators / editors write into XMP, IPTC, C2PA (JUMBF) or PNG text chunks.
AI_MARKERS = {
    "trainedalgorithmicmedia": "IPTC DigitalSourceType says AI-generated",
    "compositewithtrainedalgorithmicmedia": "IPTC DigitalSourceType says AI-edited composite",
    "c2pa": "C2PA content credentials present (check the manifest's claims)",
    "jumbf": "JUMBF box present (usually C2PA)",
    "synthid": "SynthID mention (Google AI)",
    "google ai": "Google AI marker",
    "gemini": "Gemini marker",
    "imagen": "Imagen marker",
    "openai": "OpenAI marker",
    "dall-e": "DALL-E marker",
    "gpt-4o": "GPT-4o image marker",
    "midjourney": "Midjourney marker",
    "stable diffusion": "Stable Diffusion marker",
    "comfyui": "ComfyUI workflow embedded",
    "automatic1111": "A1111 parameters embedded",
    "\"prompt\"": "Embedded generation prompt",
    "firefly": "Adobe Firefly marker",
    "grok": "Grok / xAI marker",
    "flux": "FLUX marker",
}

_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def inspect_file(path: str | Path, scan_bytes: int = 512_000) -> dict:
    path = Path(path)
    report: dict = {"format": None, "size": None, "camera": None, "software": None,
                    "has_exif": False, "markers": []}
    try:
        with Image.open(path) as im:
            report["format"], report["size"] = im.format, list(im.size)
            exif = im.getexif()
            report["has_exif"] = len(exif) > 0
            make, model = exif.get(_TAGS["Make"]), exif.get(_TAGS["Model"])
            if make or model:
                report["camera"] = " ".join(str(x).strip() for x in (make, model) if x)
            if exif.get(_TAGS["Software"]):
                report["software"] = str(exif.get(_TAGS["Software"])).strip()
            for k, v in (im.info or {}).items():  # PNG text chunks (A1111 'parameters', ComfyUI 'prompt')
                if isinstance(v, str) and k.lower() in ("parameters", "prompt", "workflow"):
                    report["markers"].append(f"PNG text chunk '{k}' (generation parameters)")
    except Exception as e:  # noqa: BLE001 - report, never crash the verifier
        report["error"] = str(e)
        return report

    blob = path.read_bytes()[:scan_bytes].lower()
    for needle, meaning in AI_MARKERS.items():
        if needle.encode() in blob and meaning not in report["markers"]:
            report["markers"].append(meaning)
    return report


def summarize(report: dict) -> str:
    if report.get("markers"):
        return "metadata contains AI/provenance markers: " + "; ".join(report["markers"])
    if report.get("camera"):
        return f"camera EXIF present ({report['camera']}) — weak evidence of a real capture"
    return "no useful metadata (normal for images sent via WhatsApp/Facebook)"
