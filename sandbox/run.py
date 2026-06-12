"""MathMark drop-folder wrapper.

Usage:
    python sandbox/run.py embed <image>      # adds watermark, writes to sandbox/out/
    python sandbox/run.py verify <image>     # checks attribution, prints verdict

The config is loaded from ~/.mathmark/config.yaml (created by `mathmark init`).
All 6 layers are enabled by default.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

from mathmark import LayerType, WatermarkConfig, WatermarkPipeline
from mathmark.core.config import load_config, load_signature
from mathmark.utils.image_io import load_image, save_image
from mathmark.verify.extractor import verify_image

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "in"
OUT_DIR = ROOT / "out"
VERIFY_DIR = ROOT / "verify"
CFG_PATH = Path.home() / ".mathmark" / "config.yaml"

VERDICT_BADGE = {
    "STRONG_MATCH": "STRONG MATCH ✓✓",
    "PROBABLE_MATCH": "PROBABLE MATCH ✓",
    "WEAK_MATCH": "WEAK MATCH ?",
    "NO_MATCH": "NO MATCH ✗",
    "INCONCLUSIVE": "INCONCLUSIVE ?",
}


def _load_cfg() -> WatermarkConfig:
    if not CFG_PATH.exists():
        print(f"✗ Config not found: {CFG_PATH}")
        print(f"  Run once:  mathmark init --name '你的名字'")
        print(f"  Or:       python sandbox/run.py demo   (auto-setup + run)")
        sys.exit(1)
    cfg = load_config(CFG_PATH)
    # Force all 6 layers on, regardless of what's in the yaml
    from mathmark.core.config import _parse_layers
    cfg.enabled_layers = _parse_layers("all")
    # If the yaml has no inline signature but ~/.mathmark/signatures/*.json does,
    # wire it up so L4 (semantic) runs.
    if cfg.semantic.signature is None:
        sig_dir = Path.home() / ".mathmark" / "signatures"
        if sig_dir.exists():
            for p in sig_dir.glob("*.json"):
                try:
                    cfg.semantic.signature = load_signature(p)
                    break
                except Exception:
                    continue
    return cfg


def cmd_embed(src: Path) -> Path:
    if not src.is_file():
        print(f"✗ Not a file: {src}")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / src.name

    cfg = _load_cfg()
    pipe = WatermarkPipeline(cfg)

    pil = load_image(src, mode="RGB")
    image = np.array(pil, dtype=np.uint8)

    manifest = dst.with_suffix(dst.suffix + ".manifest.json")
    result = pipe.process(image, output_path=dst, manifest_path=manifest)

    print(f"[embed] {src} → {dst}  ({result.total_duration_ms:.0f} ms)")
    for layer, rep in result.layer_reports.items():
        mark = "✓" if rep.success else "✗"
        print(f"   {mark} {layer.value:18}  {rep.message[:70]}")
    print(f"   manifest → {manifest}")
    return dst


def cmd_verify(src: Path) -> None:
    if not src.is_file():
        print(f"✗ Not a file: {src}")
        sys.exit(1)
    cfg = _load_cfg()
    res = verify_image(src, cfg)

    print(f"[verify] {src}")
    print(f"   overall score : {res.confidence:.3f}")
    print(f"   verdict       : {res.verdict.value}")
    for layer, ev in res.layer_evidence.items():
        mark = "✓" if ev.matched else "✗"
        print(f"   {mark} {layer.value:18}  {ev.confidence:.3f}")
    if res.signer_info is not None:
        print(f"   attributed to : {res.signer_info}")


def cmd_demo() -> None:
    """Generate a synthetic math-image, embed it, then verify it. No config needed."""
    from mathmark.crypto.keys import generate_keypair, save_keypair
    from mathmark.core.config import save_config, save_signature
    from mathmark import MathSignature, SignatureProblem

    print("[demo] generating synthetic lesson image...")
    img = Image.new("RGB", (1024, 768), "white")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        small = font
    draw.text((50, 50),  "Math Lesson: Quadratic Equations", fill="black", font=font)
    draw.text((50, 120), "Problem: x^2 - 5x + 6 = 0",         fill="black", font=font)
    draw.text((50, 200), "Step 1: 设 x 为未知数, 我们有",         fill="black", font=small)
    draw.text((50, 240), "        x^2 - 5x + 6 = 0",         fill="black", font=small)
    draw.text((50, 300), "Step 2: 化简得 (x-2)(x-3) = 0",        fill="black", font=small)
    draw.text((50, 340), "Step 3: 整理得 x = 2 或 x = 3",        fill="black", font=small)
    draw.text((50, 400), "故 解集为 {2, 3}",                    fill="black", font=font)
    draw.text((50, 470), "Q.E.D.",                          fill="black", font=font)
    sample = IN_DIR / "_demo_input.png"
    IN_DIR.mkdir(parents=True, exist_ok=True)
    img.save(sample)
    print(f"[demo] wrote {sample}")

    # One-time config + signature if absent
    if not CFG_PATH.exists():
        print("[demo] first run — setting up config + keys...")
        (CFG_PATH.parent / "keys").mkdir(parents=True, exist_ok=True)
        priv = CFG_PATH.parent / "keys" / "private.pem"
        pub  = CFG_PATH.parent / "keys" / "public.pem"
        kp = generate_keypair(algorithm="ec-p256")
        save_keypair(kp, priv, pub)

        sig = MathSignature(teacher_id="T-DEMO", teacher_name="数学王老师")
        sig.symbol.conclusion_markers = ["∴", "Q.E.D.", "故"]
        sig.step.transition_words = ["化简得", "整理得", "代入"]
        sig.example.signature_problems = [
            SignatureProblem(
                id="demo-001",
                problem="x^2 - 5x + 6 = 0",
                expected_factoring="(x-2)(x-3) = 0",
            )
        ]
        sig_path = CFG_PATH.parent / "signatures" / "demo.json"
        save_signature(sig, sig_path)

        from mathmark.core.config import create_default_config
        cfg = create_default_config(teacher_id=sig.teacher_id, teacher_name=sig.teacher_name)
        cfg.teacher_private_key_path = priv
        cfg.teacher_public_key_path  = pub
        cfg.metadata.copyright = f"© {sig.teacher_name} 2026"
        cfg.semantic.signature = sig
        save_config(cfg, CFG_PATH)
        print(f"[demo] config written: {CFG_PATH}")
        print(f"[demo] signature  : {sig_path}  (loaded automatically next run)")

    cmd_embed(sample)
    cmd_verify(OUT_DIR / sample.name)


def main(argv: Optional[list[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    if argv[0] == "demo":
        cmd_demo(); return
    if len(argv) < 2 or argv[0] not in ("embed", "verify"):
        print(__doc__)
        sys.exit(1)
    op, target = argv[0], Path(argv[1]).expanduser().resolve()
    if op == "embed":
        cmd_embed(target)
    else:
        cmd_verify(target)


if __name__ == "__main__":
    main()
