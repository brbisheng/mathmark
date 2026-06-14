# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MathMark — a multi-layer anti-AI watermark tool for math teaching content. A teacher signs a figure once, then any leak can be traced back through 6 stacked evidence layers (L1 perceptual hash → L2 visible → L3a/b/c frequency-domain → L4 math semantic → L5 EXIF/XMP → L6 C2PA manifest). The README's threat model is the 2024 NeurIPS SOTA attack that strips 95.7% of single-layer watermarks, so multi-layer redundancy is the point — don't propose removing "redundant" layers.

## Quick start

```bash
pip install -e .                            # core install (CPU only)
pip install -e ".[all]"                     # + OCR, TrustMark, C2PA, GUI, report, dev

# zero-config self-test (no init needed; auto-creates config + signature)
python sandbox/run.py demo
```

`demo` is the fastest way to confirm the pipeline works — it generates a synthetic lesson image, embeds all 6 layers, then verifies. Use it as the first check whenever you touch the layers; it exits non-zero if anything regresses.

## Common commands

| Task | Command |
|---|---|
| Init teacher identity | `mathmark init --name "王老师"` |
| Configure signature | `mathmark sign --problem "x^2-5x+6=0" --variant "x^2-7x+12=0"` |
| Add conclusion markers | `mathmark sign --marker "∴" --marker "故"` |
| Embed watermark | `mathmark embed input.png marked/ --layers all` (positional args, no `--input`) |
| Embed a folder | `mathmark embed ./in/ ./out/ --batch` |
| Verify | `mathmark verify marked.png` |
| Verify + PDF report | `mathmark verify marked.png --report evidence.pdf` |
| Run tests | `python -m pytest tests/ -v` |
| Env check | `mathmark doctor` |
| Smoke test | `python sandbox/run.py demo` |

The `embed`/`verify`/`extract` commands use **positional paths** (CLI uses `typer.Argument`), not `--input`/`--output`. The README had this wrong at one point — if you see it broken elsewhere, fix it the same way as `cli/main.py:embed`.

## Architecture

```
src/mathmark/
├── cli/main.py            ← Typer app, all subcommands. Edit here for CLI changes.
├── core/
│   ├── config.py          ← WatermarkConfig dataclass + save/load YAML. Copyright/contact roundtrip bug lived here.
│   ├── pipeline.py        ← Orchestrates L1..L6 in order. Save_image call at the end needs exif_bytes.
│   └── types.py           ← All dataclasses (LayerType, VisibleSettings, WatermarkResult, …)
├── layers/
│   ├── l1_fingerprint.py  ← pHash + dHash, never modifies pixels
│   ├── l2_visible.py      ← tiled watermark + adversarial perturbation. {teacher_id} placeholder substituted here.
│   ├── l3a_trustmark.py   ← Adobe TrustMark via ONNX; falls back to DCT when lib missing
│   ├── l3b_dwt_dct_svd.py ← uses invisible-watermark lib (DWT-DCT-SVD)
│   ├── l3c_cox_spread.py  ← 1-bit spread spectrum; presence detector
│   ├── l4_semantic.py     ← math signature recognition (calls OCR + recognizer)
│   ├── l5_metadata.py     ← EXIF/XMP writer
│   └── l6_c2pa.py         ← C2PA manifest sidecar (simplified fallback when c2pa-python not installed)
├── semantic/              ← OCR (tesseract/paddleocr/mock), recognizer, matchers
├── crypto/                ← EC keypair + SHA-256
├── verify/                ← extractor + report
├── attacks/               ← diffusion_regen, local_blur, social_compress (for resistance tests)
└── utils/image_io.py      ← save_image MUST receive exif_bytes for PNG; L5 result goes through numpy roundtrip.
```

**Layer execution order** is fixed in `core/pipeline.py`. Don't reorder L3a/b/c — they share the DCT domain and break each other if alpha is too aggressive. Verified when running demo: L3a in fallback mode only writes to Y, never per-channel.

**State that survives across embed/verify**: only the L6 manifest sidecar (`.manifest.json`) is persistent. L1 verify uses it to find the reference phash; L5 verify reads EXIF directly from the file; L4/L3 don't need persistence (they're derived from `teacher_id:teacher_name` deterministically via `_derive_secret` in `pipeline.py:42`).

## Conventions

- **Machine-readable IDs in watermarks**: L2 default text is `"© {teacher_id} {teacher_name}"` — substitute happens in `l2_visible.process`. Keep this format; OCR-ing a screenshot of leaked content should yield the teacher ID directly.
- **EXIF survives PNG**: `save_image` only writes EXIF when `exif_bytes=` is passed explicitly. The pipeline captures it from L5's return dict. If you add a new layer that modifies EXIF, thread the bytes through the same path.
- **Optional deps are tiered**: `[deep]`, `[ocr]`, `[crypto_advanced]`, `[cli]`, `[gui]`, `[report]`, `[dev]`. `[all]` is heavy (PyQt5 + paddlepaddle don't install on minimal envs like Colab). For Colab use core + `[ocr,cli]`; skip `[gui]`, `[dev]`, `[report]`.
- **Fallbacks are degraded, not broken**: L3a falls back to a simple DCT scheme when TrustMark isn't installed — BER after PNG roundtrip is ~0.4 and the recognizer treats that as "weak indication" rather than match. Don't try to "fix" the fallback to be stronger; it cascades into breaking L3b. Install `trustmark onnxruntime` to get the real path.

## Testing

- `tests/test_cli.py` — CLI smoke tests (uses `typer.testing.CliRunner` for init; calls embed/benchmark as functions to skip runner version mismatches).
- `tests/test_pipeline.py` — full pipeline roundtrip on a synthetic image.
- `tests/test_layers.py`, `test_semantic.py`, `test_attack_resistance.py` — per-layer + attack resistance.
- The most useful regression check is `python sandbox/run.py demo` — it exercises every layer end-to-end and prints per-layer confidence.

## Do not

- Reorder the 6 layers in the pipeline.
- Add per-channel modification to L3a fallback (breaks L3b).
- Make `save_image` rely on `pil_img.info["exif"]` to roundtrip — that only works for JPEG, PNG needs explicit `exif=...`.
- Skip `warnings.filterwarnings("ignore")` in the demo / examples — the L3a and L6 "not installed" warnings are expected and noisy.
- Add CLI options to `embed` that aren't in `typer.Argument` / `typer.Option` in `cli/main.py` — the README has a history of fabricated options (`--input`, `--content-type`, `inject` subcommand) that don't exist.

## Git workflow

- No hooks are configured today; if/when they are, do not bypass them.
- Ask how to handle uncommitted changes before starting.
- Create a new branch when no clear branch exists.
- Commit frequently.
