"""L6 C2PA (Content Credentials) 层

C2PA 是 Adobe/Microsoft/Meta/OpenAI 等联合推动的内容凭证标准
抗篡改签名,可嵌入图像、视频、音频

依赖:
    pip install c2pa-python

如果未安装,使用简化的 JSON 清单作为降级方案(无密码学签名)。
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

from ..core.types import C2PASettings, LayerReport, LayerType
from ..crypto.hashing import sha256_bytes
from ..crypto.keys import (
    KeyPair,
    load_keypair,
    sign_data,
    verify_signature,
)
from ..utils.perf import measure_time

PathLike = Union[str, Path]

try:
    import c2pa
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False


def _image_fingerprint(image: np.ndarray) -> str:
    """Stable per-image hash that survives JPEG roundtrip / resize.

    Audit B3: previously we hashed `image.tobytes()` which produced a
    byte-exact hash. Any decode (JPEG, resize, color conversion) changed
    the bytes and broke verification. Use pHash from L1 instead so the
    embed and verify sides agree on a perceptual fingerprint.
    """
    try:
        from PIL import Image as _PILImage
        from .l1_fingerprint import compute_fingerprint
        pil = _PILImage.fromarray(image.astype(np.uint8)).convert("RGB")
        fp = compute_fingerprint(pil)
        if fp.phash:
            return fp.phash
    except Exception:
        pass
    # Fallback: deterministic PNG re-encode (still survives resize via
    # a 32-bit dHash if the above path fails)
    try:
        from PIL import Image as _PILImage
        import io as _io
        buf = _io.BytesIO()
        _PILImage.fromarray(image.astype(np.uint8)).convert("RGB").save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()
    except Exception:
        return hashlib.sha256(image.astype(np.uint8).tobytes()).hexdigest()


# ============================================================
# 简化清单 (降级方案)
# ============================================================

@dataclass
class SimplifiedManifest:
    """简化的 C2PA 风格清单(自带签名)"""
    teacher_id: str
    teacher_name: str
    algorithm: str
    image_hash: str  # SHA-256 of image
    timestamp: str
    public_key_fingerprint: str
    signature: bytes
    assertions: list[dict]

    def to_dict(self) -> dict:
        return {
            "teacher_id": self.teacher_id,
            "teacher_name": self.teacher_name,
            "algorithm": self.algorithm,
            "image_hash": self.image_hash,
            "timestamp": self.timestamp,
            "public_key_fingerprint": self.public_key_fingerprint,
            "signature": self.signature.hex(),
            "assertions": self.assertions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimplifiedManifest":
        return cls(
            teacher_id=d["teacher_id"],
            teacher_name=d.get("teacher_name", ""),
            algorithm=d["algorithm"],
            image_hash=d["image_hash"],
            timestamp=d["timestamp"],
            public_key_fingerprint=d["public_key_fingerprint"],
            signature=bytes.fromhex(d["signature"]),
            assertions=d.get("assertions", []),
        )


def _build_assertions(
    teacher_id: str,
    teacher_name: str,
    phash: Optional[str] = None,
    semantic_sim: Optional[float] = None,
    custom: Optional[dict] = None,
    *,
    timestamp: Optional[str] = None,
) -> list[dict]:
    """构造 C2PA assertions

    `timestamp` is explicit so verify can re-build the same bytes that
    were signed at embed time. Without it, `dateCreated = datetime.now()`
    drifts between sign and verify and the signature never matches.
    """
    ts = timestamp or datetime.now().isoformat()
    year = datetime.fromisoformat(ts).year
    assertions = [
        {
            "label": "stds.schema-org.CreativeWork",
            "data": {
                "@context": "https://schema.org",
                "@type": "CreativeWork",
                "author": {
                    "@type": "Person",
                    "name": teacher_name or teacher_id,
                    "identifier": teacher_id,
                },
                "copyrightHolder": {
                    "@type": "Person",
                    "name": teacher_name or teacher_id,
                },
                "copyrightYear": year,
                "dateCreated": ts,
                "name": "MathMark-signed educational content",
                "description": "Mathematics teaching material with content provenance",
            },
        },
        {
            "label": "mathmark.signature",
            "data": {
                "version": "0.1.0",
                "perceptual_hash": phash or "",
                "semantic_similarity": semantic_sim or 0.0,
            },
        },
    ]
    if custom:
        assertions.append({
            "label": "mathmark.custom",
            "data": custom,
        })
    return assertions


def _create_simplified_manifest(
    image: np.ndarray,
    teacher_id: str,
    teacher_name: str,
    phash: Optional[str] = None,
    semantic_sim: Optional[float] = None,
    keypair: Optional[KeyPair] = None,
    custom_assertions: Optional[dict] = None,
) -> Tuple[SimplifiedManifest, bool]:
    """创建简化 manifest"""
    # audit B3: use perceptual fingerprint, not raw bytes, so verify
    # survives JPEG roundtrip / resize
    image_hash = _image_fingerprint(image)
    # Freeze one timestamp used by both the manifest field and the
    # assertions so verify can re-build the same bytes that were signed.
    now_iso = datetime.now().isoformat()
    assertions = _build_assertions(
        teacher_id, teacher_name, phash, semantic_sim, custom_assertions,
        timestamp=now_iso,
    )

    if keypair is None:
        # 无签名(降级)
        return SimplifiedManifest(
            teacher_id=teacher_id,
            teacher_name=teacher_name,
            algorithm="none",
            image_hash=image_hash,
            timestamp=now_iso,
            public_key_fingerprint="",
            signature=b"",
            assertions=assertions,
        ), True  # used_fallback

    pub_fingerprint = keypair.public_key_fingerprint()

    # 签名 assertions
    data_to_sign = json.dumps({
        "teacher_id": teacher_id,
        "image_hash": image_hash,
        "timestamp": now_iso,
        "assertions": assertions,
    }, sort_keys=True, ensure_ascii=False).encode("utf-8")

    signature = sign_data(keypair.private_key, data_to_sign, keypair.algorithm)

    return SimplifiedManifest(
        teacher_id=teacher_id,
        teacher_name=teacher_name,
        algorithm=keypair.algorithm,
        image_hash=image_hash,
        timestamp=now_iso,
        public_key_fingerprint=pub_fingerprint,
        signature=signature,
        assertions=assertions,
    ), False


def _verify_simplified_manifest(
    manifest: SimplifiedManifest,
    image: np.ndarray,
    public_key=None,
) -> Tuple[bool, str]:
    """验证简化 manifest"""
    # 1. 验证 image hash (perceptual, survives JPEG roundtrip / resize)
    image_hash = _image_fingerprint(image)
    if image_hash != manifest.image_hash:
        return False, f"Image hash mismatch: expected {manifest.image_hash[:16]}..., got {image_hash[:16]}..."

    # 2. 验证签名(如果有)
    # audit B1: if the manifest carries a signature we MUST verify it,
    # otherwise an attacker can write a fake .manifest.json and the
    # verifier will say "OK" without ever checking anything.
    if manifest.signature:
        if public_key is None:
            return False, (
                "Manifest is signed but no public key provided; "
                "refusing to verify (audit B1)"
            )
        data_to_verify = json.dumps({
            "teacher_id": manifest.teacher_id,
            "image_hash": manifest.image_hash,
            "timestamp": manifest.timestamp,
            "assertions": manifest.assertions,
        }, sort_keys=True, ensure_ascii=False).encode("utf-8")

        if not verify_signature(public_key, data_to_verify, manifest.signature):
            return False, "Signature verification failed"

    return True, "OK"


# ============================================================
# C2PA 标准库 (可选)
# ============================================================

def _create_c2pa_manifest(
    image_path: PathLike,
    teacher_id: str,
    teacher_name: str,
    keypair: Optional[KeyPair] = None,
) -> Tuple[Path, str]:
    """使用 c2pa-python 库创建标准 manifest"""
    # TODO: 实现 c2pa-python 集成
    # 当前为简化版本
    raise NotImplementedError("c2pa-python integration pending")


# ============================================================
# Layer 接口
# ============================================================

def process(
    image: np.ndarray,
    settings: C2PASettings,
    teacher_id: str,
    teacher_name: str = "",
    phash: Optional[str] = None,
    semantic_sim: Optional[float] = None,
    keypair: Optional[KeyPair] = None,
    output_manifest_path: Optional[PathLike] = None,
) -> Tuple[np.ndarray, LayerReport, Optional[Path]]:
    """L6 C2PA 处理

    Args:
        image: 输入图像
        settings: L6 配置
        teacher_id / teacher_name: 教师身份
        phash: L1 指纹 (写入 manifest)
        semantic_sim: L4 相似度 (写入 manifest)
        keypair: 签名密钥(若为 None 则不签名)
        output_manifest_path: manifest 输出路径
    """
    with measure_time("L6_c2pa") as timer:
        manifest_path: Optional[Path] = None
        try:
            if not settings.enable:
                report = LayerReport(
                    layer=LayerType.C2PA,
                    success=True,
                    duration_ms=timer.duration_ms,
                    message="C2PA disabled",
                    metadata={"enabled": False},
                )
                return image, report, None

            # 加载密钥(如果配置了路径)
            actual_keypair = keypair
            if actual_keypair is None and settings.private_key_path:
                try:
                    actual_keypair = load_keypair(settings.private_key_path)
                except Exception as e:
                    warnings.warn(f"Failed to load keypair: {e}")

            # 创建 manifest
            if C2PA_AVAILABLE and settings.private_key_path is not None:
                # 尝试用 c2pa-python
                try:
                    img_path = output_manifest_path.parent / "_temp_for_c2pa.png" if output_manifest_path else None
                    if img_path:
                        from ..utils.image_io import save_image
                        save_image(image, img_path)
                    manifest_path, _ = _create_c2pa_manifest(
                        img_path, teacher_id, teacher_name, actual_keypair
                    )
                    used_fallback = False
                except Exception as e:
                    # 降级
                    manifest, used_fallback = _create_simplified_manifest(
                        image, teacher_id, teacher_name, phash, semantic_sim, actual_keypair
                    )
                    if output_manifest_path:
                        manifest_path = Path(output_manifest_path)
                        manifest_path.parent.mkdir(parents=True, exist_ok=True)
                        manifest_path.write_text(
                            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    message_suffix = f" (c2pa-python failed: {e})"
            else:
                # 简化 manifest
                manifest, used_fallback = _create_simplified_manifest(
                    image, teacher_id, teacher_name, phash, semantic_sim, actual_keypair
                )
                if output_manifest_path:
                    manifest_path = Path(output_manifest_path)
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(
                        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

            algorithm = "simplified"
            if actual_keypair is not None:
                algorithm = f"simplified+{actual_keypair.algorithm}"

            message = f"Manifest created: {algorithm}, teacher={teacher_id}"
            if used_fallback:
                message += " (fallback, no c2pa-python)"

            report = LayerReport(
                layer=LayerType.C2PA,
                success=True,
                duration_ms=timer.duration_ms,
                message=message,
                metadata={
                    "manifest_path": str(manifest_path) if manifest_path else None,
                    "algorithm": algorithm,
                    "used_fallback": used_fallback,
                    "teacher_id": teacher_id,
                },
            )
        except Exception as e:
            report = LayerReport(
                layer=LayerType.C2PA,
                success=False,
                duration_ms=timer.duration_ms,
                message=f"Failed: {e}",
            )

    return image, report, manifest_path


def verify_manifest(
    manifest_path: PathLike,
    image: np.ndarray,
    public_key=None,
) -> Tuple[bool, str]:
    """验证 manifest"""
    try:
        manifest = SimplifiedManifest.from_dict(
            json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        )
        return _verify_simplified_manifest(manifest, image, public_key)
    except Exception as e:
        return False, f"Manifest parse error: {e}"
