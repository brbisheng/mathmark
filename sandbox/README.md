# MathMark sandbox

A tiny "drop folder" wrapper for trying MathMark without learning the full CLI.

## Layout

```
sandbox/
├── in/      ← put images here that you want watermarked
├── out/     ← watermarked images appear here, with .manifest.json
├── verify/  ← put suspicious images here that you want to check
└── run.py   ← the wrapper
```

## Quick start (zero config)

```bash
python3 sandbox/run.py demo
```

This will:
1. Generate a synthetic math-lesson image → `sandbox/in/_demo_input.png`
2. Create `~/.mathmark/config.yaml`, keys, and a demo signature (first run only)
3. Embed a 6-layer watermark → `sandbox/out/_demo_input.png`
4. Verify it back, printing per-layer confidence and a verdict

No manual `mathmark init` needed. Use this first to confirm the pipeline works at all.

## Real usage

```bash
# 0. One-time setup (only if you didn't run `demo` first)
mathmark init --name "你的名字"
mathmark sign --problem "x^2-5x+6=0" --variant "x^2-7x+12=0"

# 1. Drop a math-lesson image into sandbox/in/, then:
python3 sandbox/run.py embed sandbox/in/lesson1.png
#   → watermarked image + manifest in sandbox/out/

# 2. To test whether a stolen/suspect image carries your signature:
python3 sandbox/run.py verify sandbox/out/lesson1.png
#   → overall score + per-layer breakdown + verdict
```

## What the verdict means

| Verdict | Meaning |
|---|---|
| `STRONG_MATCH` | High confidence the image is yours (multiple layers agree) |
| `PROBABLE_MATCH` | Likely yours, useful for follow-up investigation |
| `WEAK_MATCH` | Some signal, but not enough to act on |
| `NO_MATCH` | No signature detected |
| `INCONCLUSIVE` | Could not decide (e.g. image too small, layers failed) |

## Optional: the `verify/` folder

`verify/` is a convention only — it exists so you can keep suspect images
separate from images you own. `run.py verify` accepts any path; the
folder name is just a habit.

## Reverting

Outputs in `sandbox/out/` and configs in `~/.mathmark/` are local and
git-ignored. Delete them to start fresh:

```bash
rm -rf sandbox/out/* ~/.mathmark/
```

## Notes

- All 6 layers are enabled (L1 pHash, L2 visible, L3a TrustMark, L3b
  DWT-DCT-SVD, L3c Cox, L4 semantic, L5 metadata, L6 C2PA). If a
  layer's optional dependency is missing, it falls back gracefully and
  reports `⊘` in the per-layer output.
- TrustMark (L3a) needs the `trustmark` Python package and an ONNX model
  at `~/.mathmark/models/trustmark.onnx`. Without it, L3a falls back to
  DWT-DCT — the pipeline still works, just with one fewer layer.
