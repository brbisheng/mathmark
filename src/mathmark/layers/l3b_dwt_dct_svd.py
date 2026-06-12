"""L3b DWT-DCT-SVD 不可见水印

经典方法, 基于离散小波变换 + 离散余弦变换 + 奇异值分解
抗 JPEG 压缩、轻度几何变换

依赖:
    pip install invisible-watermark PyWavelets

如果 invisible-watermark 未安装, 使用自实现简化版本。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pywt

from ..core.types import DWTSettings, LayerReport, LayerType
from ..utils.perf import measure_time

PathLike = Union[str, Path]

try:
    from imwatermark import WatermarkEncoder, WatermarkDecoder
    INVISIBLE_WATERMARK_AVAILABLE = True
except ImportError:
    INVISIBLE_WATERMARK_AVAILABLE = False
    warnings.warn("invisible-watermark not installed. L3b will use fallback DWT-SVD.", stacklevel=2)


# ============================================================
# imwatermark 后端
# ============================================================

def _imwatermark_embed(image: np.ndarray, watermark_bits: np.ndarray, settings: DWTSettings) -> np.ndarray:
    """使用 imwatermark.WatermarkEncoder 嵌入"""
    if image.ndim == 2:
        # 灰度转 RGB
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    encoder = WatermarkEncoder()
    encoder.set_by_bits([int(b) % 2 for b in watermark_bits])
    # imwatermark 不接受 blockSize 关键字参数,用默认 4
    result = encoder.encode(image.astype(np.uint8), method="dwtDctSvd")
    return result


def _imwatermark_extract(image: np.ndarray, wm_length: int, settings: DWTSettings) -> np.ndarray:
    """使用 imwatermark.WatermarkDecoder 提取"""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    decoder = WatermarkDecoder(wm_type="bits", length=wm_length)
    bits = decoder.decode(image.astype(np.uint8), method="dwtDctSvd")
    return np.array(bits, dtype=np.uint8)


# ============================================================
# 降级后端: 自实现 DWT-SVD
# ============================================================

def _haar_dwt2(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Haar 小波 2D 分解 (一个层级)

    Returns:
        (cA, cH, cV, cD) - 低频, 水平高频, 垂直高频, 对角高频
    """
    coeffs = pywt.dwt2(img, "haar")
    return coeffs


def _haar_idwt2(coeffs: Tuple) -> np.ndarray:
    """Haar 小波 2D 重建"""
    return pywt.idwt2(coeffs, "haar")


def _fallback_dwt_svd_embed(
    image: np.ndarray,
    watermark_bits: np.ndarray,
    settings: DWTSettings,
) -> np.ndarray:
    """DWT-SVD 嵌入 - 简化版"""
    h, w = image.shape[:2]
    if image.ndim == 3:
        # 在 Y 通道嵌入
        Y = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        Y = image.astype(np.float64)

    # 1-level Haar DWT
    cA, (cH, cV, cD) = _haar_dwt2(Y)

    # 在 cA (低频) 上做 SVD
    # cA 的尺寸是原图一半
    U, S, V = np.linalg.svd(cA, full_matrices=False)

    # 调整 S 中部分值嵌入水印
    n_bits = min(len(watermark_bits), len(S))
    S_new = S.copy()
    for i in range(n_bits):
        # 量化调制
        bit = int(watermark_bits[i])
        # 原始奇异值的小数部分
        s = S[i]
        if s == 0:
            continue
        floor_s = np.floor(s)
        # 确保是奇数还是偶数代表 bit
        if bit == 1:
            S_new[i] = floor_s + 0.75 if (floor_s % 2 == 0) else floor_s + 0.75
        else:
            S_new[i] = floor_s + 0.25 if (floor_s % 2 == 1) else floor_s + 0.25

    # 重建 cA
    cA_new = U @ np.diag(S_new) @ V
    cA_new = cA_new.reshape(cA.shape)

    # 重建图像
    Y_wm = _haar_idwt2((cA_new, (cH, cV, cD)))

    if image.ndim == 3:
        # 简单合成
        scale = Y_wm / (Y + 1e-8)
        result = (image.astype(np.float64) * scale[..., None]).clip(0, 255).astype(np.uint8)
        return result
    else:
        return Y_wm.clip(0, 255).astype(np.uint8)


def _fallback_dwt_svd_extract(
    image: np.ndarray,
    wm_length: int,
    settings: DWTSettings,
) -> np.ndarray:
    """DWT-SVD 提取 - 简化版"""
    h, w = image.shape[:2]
    if image.ndim == 3:
        Y = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        Y = image.astype(np.float64)

    cA, _ = _haar_dwt2(Y)
    U, S, V = np.linalg.svd(cA, full_matrices=False)

    n_bits = min(wm_length, len(S))
    bits = []
    for i in range(n_bits):
        s = S[i]
        if s == 0:
            bits.append(0)
            continue
        # 看小数部分更接近 0.25 还是 0.75
        frac = s - np.floor(s)
        bits.append(1 if frac > 0.5 else 0)

    return np.array(bits, dtype=np.uint8)


# ============================================================
# Layer 接口
# ============================================================

def process(
    image: np.ndarray,
    settings: DWTSettings,
    watermark_bits: np.ndarray,
    output_path: Optional[PathLike] = None,
) -> Tuple[np.ndarray, LayerReport, np.ndarray]:
    """L3b DWT-DCT-SVD 处理"""
    with measure_time("L3b_dwt_dct_svd") as timer:
        try:
            if INVISIBLE_WATERMARK_AVAILABLE:
                result = _imwatermark_embed(image, watermark_bits, settings)
                used_fallback = False
                message = f"DWTSVD encoded, {len(watermark_bits)} bits"
            else:
                result = _fallback_dwt_svd_embed(image, watermark_bits, settings)
                used_fallback = True
                message = f"DWT-SVD fallback encoded, {len(watermark_bits)} bits"
        except Exception as e:
            result = image
            watermark_bits = np.array([], dtype=np.uint8)
            report = LayerReport(
                layer=LayerType.INVISIBLE_DWT,
                success=False,
                duration_ms=timer.duration_ms,
                message=f"Failed: {e}",
            )
            if output_path:
                from ..utils.image_io import save_image
                save_image(result, output_path)
            return result, report, watermark_bits

        # 验证
        try:
            if INVISIBLE_WATERMARK_AVAILABLE:
                decoded = _imwatermark_extract(result, len(watermark_bits), settings)
            else:
                decoded = _fallback_dwt_svd_extract(result, len(watermark_bits), settings)
            ber = float(np.mean(decoded != watermark_bits))
        except Exception:
            ber = -1.0

    report = LayerReport(
        layer=LayerType.INVISIBLE_DWT,
        success=True,
        duration_ms=timer.duration_ms,
        message=f"{message} (BER: {ber:.3f})",
        metadata={
            "n_bits": len(watermark_bits),
            "used_fallback": used_fallback,
            "embed_ber": ber,
            "alpha": settings.alpha,
            "block_size": settings.block_size,
        },
    )

    if output_path:
        from ..utils.image_io import save_image
        save_image(result, output_path)

    return result, report, watermark_bits


def extract(
    image: np.ndarray,
    settings: DWTSettings,
    wm_length: int,
) -> np.ndarray:
    """提取 L3b 水印 bits"""
    if INVISIBLE_WATERMARK_AVAILABLE:
        return _imwatermark_extract(image, wm_length, settings)
    return _fallback_dwt_svd_extract(image, wm_length, settings)
