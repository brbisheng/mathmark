"""Shared secret derivation.

`derive_secret` is the single source of truth for the per-teacher / per-image
secret bytes. Both embed (pipeline) and verify (extractor) must call this
function so the L3 family produces and tests the same bits.

Why a shared module (and not a private `_derive_secret` in pipeline.py)?
- The extractor is in `src/mathmark/verify/`, which historically did not
  import from `core/pipeline.py` to avoid a circular import. Lifting this
  helper into a small, dependency-free module breaks the cycle and removes
  the silent divergence (audit finding B4: "extractor derives secret
  without teacher_text_content").
"""

from __future__ import annotations

from typing import Optional

from .types import WatermarkConfig
from ..crypto.hashing import sha256_bytes_raw


def derive_secret(config: WatermarkConfig, teacher_text: Optional[str] = None) -> bytes:
    """Derive a 32-byte per-(teacher, image) secret.

    Args:
        config: The watermark configuration (teacher_id + teacher_name).
        teacher_text: Optional teacher source text. If provided, only the
            first 64 characters are mixed in so the secret stays stable
            across minor edits.

    Returns:
        32 raw bytes (SHA-256 of `"{teacher_id}:{teacher_name}"` optionally
        suffixed with `":{teacher_text[:64]}"`).
    """
    base = f"{config.teacher_id}:{config.teacher_name}"
    if teacher_text:
        base += f":{teacher_text[:64]}"
    return sha256_bytes_raw(base.encode("utf-8"))


def bits_from_secret(secret: bytes, n_bits: int = 32) -> bytes:
    """Expand a 32-byte secret into `n_bits` 0/1 bytes.

    The first `n_bits // 8` bytes of the secret provide the bits; if
    `n_bits > 8 * len(secret)` the remainder is zero-padded. Uses the
    secret's own entropy so the bits depend on the full secret, not on a
    short teacher_id (audit finding B9: "L3b bits derived from short
    teacher_id").
    """
    n_bytes = (n_bits + 7) // 8
    raw = secret[:n_bytes]
    if len(raw) < n_bytes:
        raw = raw + b"\x00" * (n_bytes - len(raw))
    return raw
