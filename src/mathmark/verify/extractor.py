"""水印提取与验证 - 多层取证

从给定图像中提取所有层的水印信息,
与配置的签名/密钥比对, 计算综合归属得分。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from ..core.types import (
    Evidence,
    LayerType,
    SignerInfo,
    Verdict,
    VerificationResult,
    WatermarkConfig,
)
from ..core.secret import bits_from_secret, derive_secret
from ..crypto.hashing import sha256_bytes, sha256_bytes_raw
from ..crypto.keys import verify_signature
from ..layers import (
    l1_fingerprint,
    l2_visible,
    l3a_trustmark,
    l3b_dwt_dct_svd,
    l3c_cox_spread,
    l5_metadata,
    l6_c2pa,
)
from ..semantic.recognizer import recognize_from_image
from ..utils.image_io import hamming_similarity, load_image

PathLike = Union[str, Path]


def _verdict_from_confidence(c: float, threshold_strong: float = 0.75, threshold_probable: float = 0.55, threshold_weak: float = 0.35) -> Verdict:
    if c >= threshold_strong:
        return Verdict.STRONG_MATCH
    elif c >= threshold_probable:
        return Verdict.PROBABLE_MATCH
    elif c >= threshold_weak:
        return Verdict.WEAK_INDICATION
    return Verdict.NO_MATCH


def extract_all(
    image_path: PathLike,
    config: WatermarkConfig,
) -> dict[LayerType, Optional[bytes]]:
    """提取所有层的水印 bits"""
    try:
        pil_img = load_image(image_path, mode="RGB")
        image = np.array(pil_img, dtype=np.uint8)
    except Exception:
        return {}

    # 派生 secret (与 embed 时相同)
    secret = derive_secret(config)

    results: dict[LayerType, Optional[bytes]] = {}

    # L3a
    if config.is_enabled(LayerType.INVISIBLE_TRUSTMARK):
        try:
            bits = l3a_trustmark.extract(image, config.trustmark, n_bits=100)
            results[LayerType.INVISIBLE_TRUSTMARK] = bits
        except Exception:
            results[LayerType.INVISIBLE_TRUSTMARK] = None

    # L3b
    if config.is_enabled(LayerType.INVISIBLE_DWT):
        try:
            # 32 bits from the full per-teacher secret (matches pipeline L3b)
            wm_bits_arr = np.frombuffer(bits_from_secret(secret, 32), dtype=np.uint8)[:32] % 2
            if len(wm_bits_arr) < 32:
                wm_bits_arr = np.concatenate(
                    [wm_bits_arr, np.zeros(32 - len(wm_bits_arr), dtype=np.uint8)]
                )
            bits = l3b_dwt_dct_svd.extract(image, config.dwt, wm_length=32)
            results[LayerType.INVISIBLE_DWT] = bits.tobytes() if len(bits) > 0 else None
        except Exception:
            results[LayerType.INVISIBLE_DWT] = None

    # L3c
    if config.is_enabled(LayerType.INVISIBLE_COX):
        try:
            bits = l3c_cox_spread.extract(image, config.cox, secret, n_bits=64)
            results[LayerType.INVISIBLE_COX] = bits.tobytes() if len(bits) > 0 else None
        except Exception:
            results[LayerType.INVISIBLE_COX] = None

    return results


def verify_image(
    image_path: PathLike,
    config: WatermarkConfig,
    public_key=None,
    threshold: float = 0.5,
) -> VerificationResult:
    """验证图像归属

    Args:
        image_path: 待验证图像
        config: 验证配置
        public_key: 教师公钥(用于 C2PA 验签)
        threshold: 语义层相似度阈值
    """
    image_path = Path(image_path)
    try:
        pil_img = load_image(image_path, mode="RGB")
        image = np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        return VerificationResult(
            image_path=image_path,
            verdict=Verdict.NO_MATCH,
            confidence=0.0,
            layer_evidence={},
        )

    layer_evidence: dict[LayerType, Evidence] = {}
    semantic_sim = 0.0
    signer_info: Optional[SignerInfo] = None
    c2pa_manifest_path: Optional[Path] = None
    phash_match: Optional[dict] = None

    # ================ L1: 指纹 ================
    if config.is_enabled(LayerType.FINGERPRINT):
        try:
            fp = l1_fingerprint.compute_fingerprint(pil_img)
            # L1 没法独立判定"是否匹配", 必须有参照 phash. 优先从 manifest sidecar 读原始 phash
            ref_phash: Optional[str] = None
            for p in [
                image_path.with_suffix(image_path.suffix + ".manifest.json"),
                image_path.parent / (image_path.stem + ".manifest.json"),
            ]:
                if p.exists():
                    try:
                        mdata = json.loads(p.read_text(encoding="utf-8"))
                        for a in mdata.get("assertions", []):
                            if a.get("label") == "mathmark.signature":
                                ref_phash = a.get("data", {}).get("perceptual_hash") or None
                                break
                        if ref_phash:
                            break
                    except Exception:
                        continue

            if ref_phash and fp.phash:
                sim = hamming_similarity(ref_phash, fp.phash)
                matched = sim >= threshold
                confidence = sim
            else:
                # 找不到参照, 保守返回 0 (没法独立证伪/证实)
                sim = 0.0
                matched = False
                confidence = 0.0

            layer_evidence[LayerType.FINGERPRINT] = Evidence(
                layer=LayerType.FINGERPRINT,
                confidence=confidence,
                matched=matched,
                details={
                    "phash": fp.phash,
                    "dhash": fp.dhash,
                    "whash": fp.whash or "",
                    "ref_phash": ref_phash or "",
                    "hamming_similarity": sim,
                },
            )
        except Exception:
            layer_evidence[LayerType.FINGERPRINT] = Evidence(
                layer=LayerType.FINGERPRINT, confidence=0.0, matched=False,
            )

    # ================ L4: 语义 ================
    if config.is_enabled(LayerType.SEMANTIC) and config.semantic.signature:
        try:
            recognition = recognize_from_image(image, config.semantic.signature)
            semantic_sim = recognition.overall_similarity
            layer_evidence[LayerType.SEMANTIC] = Evidence(
                layer=LayerType.SEMANTIC,
                confidence=recognition.overall_similarity,
                matched=recognition.overall_similarity >= threshold,
                details={
                    "verdict": recognition.verdict,
                    "evidence": recognition.evidence,
                    "ocr_engine": recognition.ocr_engine,
                },
            )
        except Exception as e:
            layer_evidence[LayerType.SEMANTIC] = Evidence(
                layer=LayerType.SEMANTIC, confidence=0.0, matched=False,
                details={"error": str(e)},
            )

    # ================ L3: 不可见水印 (通过提取的 bits) ================
    bits_dict = extract_all(image_path, config)
    # 期望的 L3b bits: 必须用 embed 时同一份 secret 派生 (pipeline 同款公式)
    secret = derive_secret(config)
    expected_bits_l3b = np.frombuffer(
        bits_from_secret(secret, 32), dtype=np.uint8
    )[:32] % 2
    if len(expected_bits_l3b) < 32:
        expected_bits_l3b = np.concatenate(
            [expected_bits_l3b, np.zeros(32 - len(expected_bits_l3b), dtype=np.uint8)]
        )

    # L3a 的预期 bits: 必须用 embed 时同一份 secret 派生
    from ..layers.l3a_trustmark import _secret_to_bits as l3a_secret_to_bits
    expected_bits_l3a = l3a_secret_to_bits(secret, 100)  # 与 embed 的 n_bits 对齐

    for layer in [LayerType.INVISIBLE_TRUSTMARK, LayerType.INVISIBLE_DWT, LayerType.INVISIBLE_COX]:
        if not config.is_enabled(layer):
            continue
        bits_bytes = bits_dict.get(layer)
        if bits_bytes is None:
            layer_evidence[layer] = Evidence(layer=layer, confidence=0.0, matched=False)
            continue

        extracted_arr = np.frombuffer(bits_bytes, dtype=np.uint8)

        if layer == LayerType.INVISIBLE_COX:
            # L3c 只嵌入 1 bit, 重复到 n_bits. "匹配" 判定: 64 个 bit 是否一致
            # (presence detector: 一致说明是被 embed 过的图)
            # (audit B11: 全 0 一致不能给高置信度 — 可能是未水印图像天然偏 0)
            if len(extracted_arr) == 0:
                confidence = 0.0
                ber = 1.0
            else:
                unique_bits = np.unique(extracted_arr)
                if len(unique_bits) == 1:
                    bit_value = int(unique_bits[0])
                    if bit_value == 1:
                        # 全部 1 → 强存在信号 (未水印图像天然不太可能全 1)
                        confidence = 0.9
                        ber = 0.0
                    else:
                        # 全部 0 → 可能是未水印图像天然偏 0, 弱信号
                        confidence = 0.4
                        ber = 0.0
                else:
                    # 提取出多于一种 bit 值 → 未嵌入或被破坏
                    majority = int(np.bincount(extracted_arr).argmax())
                    frac_majority = float(np.mean(extracted_arr == majority))
                    confidence = frac_majority
                    ber = 1.0 - frac_majority
            layer_evidence[layer] = Evidence(
                layer=layer,
                confidence=confidence,
                matched=confidence > 0.7,
                details={"ber": ber, "n_bits": len(extracted_arr), "detector": "presence"},
            )
            continue

        # L3a: 100 bits from secret; L3b: 32 bits from teacher_id
        if layer == LayerType.INVISIBLE_TRUSTMARK:
            expected = expected_bits_l3a
        else:
            expected = expected_bits_l3b

        if len(extracted_arr) < len(expected):
            extracted_arr = np.concatenate([extracted_arr, np.zeros(len(expected) - len(extracted_arr), dtype=np.uint8)])
        extracted_arr = extracted_arr[:len(expected)]
        ber = float(np.mean(extracted_arr != expected))

        if layer == LayerType.INVISIBLE_TRUSTMARK and ber > 0.3:
            # L3a fallback (DCT) 在 PNG 来回后 BER 天然就高 (~0.4).
            # 这种情况给 "弱存在" 信号: 不当作匹配, 但也不是 0.
            # 装上 trustmark lib + onnx 模型后 BER 会回到 ~0.
            confidence = 0.4
            matched = False
            details_label = "fallback-degraded"
        else:
            confidence = 1.0 - ber
            matched = confidence > 0.7
            details_label = None

        details = {"ber": ber, "n_bits": len(extracted_arr)}
        if details_label:
            details["mode"] = details_label
        layer_evidence[layer] = Evidence(
            layer=layer,
            confidence=confidence,
            matched=matched,
            details=details,
        )

    # ================ L5: 元数据 ================
    if config.is_enabled(LayerType.METADATA):
        meta = l5_metadata.extract(pil_img)
        # 检查 mathmark 字段
        if "mathmark" in meta and meta["mathmark"].get("mathmark_teacher_id") == config.teacher_id:
            layer_evidence[LayerType.METADATA] = Evidence(
                layer=LayerType.METADATA,
                confidence=0.95,
                matched=True,
                details={"mathmark": meta["mathmark"]},
            )
        elif "mathmark" in meta:
            layer_evidence[LayerType.METADATA] = Evidence(
                layer=LayerType.METADATA,
                confidence=0.5,
                matched=False,
                details={"mathmark": meta["mathmark"]},
            )
        else:
            layer_evidence[LayerType.METADATA] = Evidence(
                layer=LayerType.METADATA,
                confidence=0.0,
                matched=False,
            )

    # ================ L6: C2PA ================
    if config.is_enabled(LayerType.C2PA):
        # 查找伴随的 manifest 文件
        possible_paths = [
            image_path.with_suffix(image_path.suffix + ".manifest.json"),
            image_path.parent / (image_path.stem + ".manifest.json"),
        ]
        for p in possible_paths:
            if p.exists():
                c2pa_manifest_path = p
                try:
                    valid, msg = l6_c2pa.verify_manifest(p, image, public_key=public_key)
                    if valid:
                        manifest_data = json.loads(p.read_text(encoding="utf-8"))
                        layer_evidence[LayerType.C2PA] = Evidence(
                            layer=LayerType.C2PA,
                            confidence=1.0,
                            matched=True,
                            details={"manifest": manifest_data, "verify_msg": msg},
                        )
                        signer_info = SignerInfo(
                            teacher_id=manifest_data.get("teacher_id", ""),
                            teacher_name=manifest_data.get("teacher_name", ""),
                            public_key_fingerprint=manifest_data.get("public_key_fingerprint", ""),
                            signing_time=__parse_datetime(manifest_data.get("timestamp")),
                        )
                    else:
                        layer_evidence[LayerType.C2PA] = Evidence(
                            layer=LayerType.C2PA,
                            confidence=0.3,
                            matched=False,
                            details={"verify_msg": msg},
                        )
                except Exception as e:
                    layer_evidence[LayerType.C2PA] = Evidence(
                        layer=LayerType.C2PA,
                        confidence=0.0,
                        matched=False,
                        details={"error": str(e)},
                    )
                break
        else:
            layer_evidence[LayerType.C2PA] = Evidence(
                layer=LayerType.C2PA, confidence=0.0, matched=False,
                details={"reason": "no_manifest_file"},
            )

    # ================ 综合判定 ================
    # 加权: L6(0.30) + L3(0.20) + L4(0.30) + L5(0.10) + L1(0.10)
    weights = {
        LayerType.C2PA: 0.30,
        LayerType.SEMANTIC: 0.30,
        LayerType.INVISIBLE_TRUSTMARK: 0.07,
        LayerType.INVISIBLE_DWT: 0.07,
        LayerType.INVISIBLE_COX: 0.06,
        LayerType.METADATA: 0.10,
        LayerType.FINGERPRINT: 0.05,
        LayerType.VISIBLE: 0.05,
    }
    total_conf = 0.0
    total_weight = 0.0
    for layer, ev in layer_evidence.items():
        w = weights.get(layer, 0.0)
        if w > 0:
            total_conf += w * ev.confidence
            total_weight += w
    if total_weight > 0:
        total_conf = total_conf / total_weight  # 归一化
    # 应用阈值
    verdict = _verdict_from_confidence(total_conf)
    if total_conf < threshold:
        verdict = Verdict.WEAK_INDICATION if total_conf > 0.2 else Verdict.NO_MATCH

    return VerificationResult(
        image_path=image_path,
        verdict=verdict,
        confidence=total_conf,
        layer_evidence=layer_evidence,
        signer_info=signer_info,
        c2pa_manifest_path=c2pa_manifest_path,
        semantic_similarity=semantic_sim,
        phash_match=phash_match,
    )


def __parse_datetime(s: str):
    """解析 datetime 字符串"""
    from datetime import datetime
    if not s:
        return datetime.now()
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now()
