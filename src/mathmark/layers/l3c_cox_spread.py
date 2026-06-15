"""L3c Cox 扩频水印

经典 Cox 扩频水印 (1997), 简单鲁棒
基于频域(DCT)嵌入伪随机序列, 抗一般压缩

完全 CPU 友好, 仅需 numpy + scipy
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy.fft import dct, idct

from ..core.types import CoxSettings, LayerReport, LayerType
from ..utils.perf import measure_time

PathLike = Union[str, Path]


def _generate_pn_sequence(length: int, seed: int) -> np.ndarray:
    """生成伪随机 ±1 序列"""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=length).astype(np.float32)


def _dct2d(img: np.ndarray) -> np.ndarray:
    """2D DCT"""
    return dct(dct(img.astype(np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2d(coeffs: np.ndarray) -> np.ndarray:
    """2D IDCT"""
    return idct(idct(coeffs.astype(np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")


def _embed_in_luminance(
    image: np.ndarray,
    pn_seq: np.ndarray,
    strength: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 Y 通道嵌入 Cox 扩频水印

    Returns:
        watermarked_image, cA_original, cA_watermarked
    """
    h, w = image.shape[:2]

    # 转 YCbCr
    if image.ndim == 3:
        Y = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        Y = image.astype(np.float64)

    # 2D DCT
    Y_dct = _dct2d(Y)

    # 嵌入到中低频带 (排除了 DC 和最高频, 中间区域)
    h_, w_ = Y_dct.shape
    # 选择中间 1/3 频段
    y_start, y_end = h_ // 3, 2 * h_ // 3
    x_start, x_end = w_ // 3, 2 * w_ // 3
    region = Y_dct[y_start:y_end, x_start:x_end]
    n_region = region.size

    # 截断/补齐 pn 序列到 region 大小
    if len(pn_seq) < n_region:
        # 用 PRNG 扩展
        from numpy.random import default_rng
        rng = default_rng(int(pn_seq[0] * 1000) if len(pn_seq) > 0 else 42)
        pn_full = rng.choice([-1.0, 1.0], size=n_region).astype(np.float32)
    else:
        pn_full = pn_seq[:n_region]
    pn = pn_full.reshape(region.shape)

    # 自适应强度 (基于区域幅度)
    magnitude = np.abs(region) + 1e-6
    # Cox 嵌入: I' = I + alpha * |I| * W
    region_wm = region + strength * magnitude * pn
    Y_dct_wm = Y_dct.copy()
    Y_dct_wm[y_start:y_end, x_start:x_end] = region_wm

    # 逆 DCT
    Y_wm = _idct2d(Y_dct_wm)

    if image.ndim == 3:
        # 还原 RGB
        # 保持原色调,只调整亮度
        scale = Y_wm / (Y + 1e-8)
        result = (image.astype(np.float64) * scale[..., None]).clip(0, 255).astype(np.uint8)
    else:
        result = Y_wm.clip(0, 255).astype(np.uint8)

    return result, Y_dct, Y_dct_wm


def _extract_from_luminance(
    image: np.ndarray,
    pn_seq: np.ndarray,
    original_dct: Optional[np.ndarray] = None,
) -> np.ndarray:
    """提取 Cox 水印

    如果有原始 DCT, 用非盲提取 (更鲁棒)
    否则用盲提取 (基于相关检测)
    """
    h, w = image.shape[:2]
    if image.ndim == 3:
        Y = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        Y = image.astype(np.float64)

    Y_dct = _dct2d(Y)
    h_, w_ = Y_dct.shape
    y_start, y_end = h_ // 3, 2 * h_ // 3
    x_start, x_end = w_ // 3, 2 * w_ // 3
    region = Y_dct[y_start:y_end, x_start:x_end]
    n_region = region.size
    # 处理 PN 序列长度不匹配
    if len(pn_seq) < n_region:
        from numpy.random import default_rng
        rng = default_rng(int(pn_seq[0] * 1000) if len(pn_seq) > 0 else 42)
        pn_full = rng.choice([-1.0, 1.0], size=n_region).astype(np.float32)
    else:
        pn_full = pn_seq[:n_region]
    pn = pn_full.reshape(region.shape)

    if original_dct is not None:
        # 非盲提取: 减去原图
        orig_region = original_dct[y_start:y_end, x_start:x_end]
        diff = region - orig_region
        # 相关性
        correlation = np.sum(diff * pn)
        return np.array([1 if correlation > 0 else 0], dtype=np.uint8)
    else:
        # 盲提取: 相关性 (不太可靠)
        correlation = np.sum(region * pn)
        return np.array([1 if correlation > 0 else 0], dtype=np.uint8)


# ============================================================
# Layer 接口
# ============================================================

def process(
    image: np.ndarray,
    settings: CoxSettings,
    secret: bytes,
    n_bits: int = 64,
    output_path: Optional[PathLike] = None,
) -> Tuple[np.ndarray, LayerReport, np.ndarray]:
    """L3c Cox 扩频处理

    注意: Cox 扩频通常只携带 1 bit (有/无),
    扩展到 n_bits 是通过区域划分实现(简化版本用 1 bit 即可)。
    """
    with measure_time("L3c_cox_spread") as timer:
        try:
            # 简化: Cox 携带 1 bit (经典 Cox 方案就是 1 bit)
            # 多 bit 实现需要区域划分,这里直接做 1 bit 嵌入
            # 使用固定长度 64*64 与 extract() 对齐, 避免 embed/extract PN 长度不匹配
            # (audit B5: extract 之前硬编码 64*64, embed 用 image.shape[0]*image.shape[1])
            base_seed = settings.seed + int.from_bytes(secret[:4], "big")
            pn = _generate_pn_sequence(64 * 64, base_seed)

            result, Y_orig, Y_wm = _embed_in_luminance(image, pn, settings.strength)
            extracted = _extract_from_luminance(result, pn, Y_orig)
            bit_value = int(extracted[0])

            # 重复 1 bit 到 n_bits (在 n_bits 字段中)
            bits = [bit_value] * n_bits

            report = LayerReport(
                layer=LayerType.INVISIBLE_COX,
                success=True,
                duration_ms=timer.duration_ms,
                message=f"Cox spread spectrum encoded, 1 bit (replicated to {n_bits})",
                metadata={
                    "n_bits": n_bits,
                    "strength": settings.strength,
                    "seed": settings.seed,
                },
            )
            bits_arr = np.array(bits, dtype=np.uint8)
        except Exception as e:
            result = image
            bits_arr = np.array([], dtype=np.uint8)
            report = LayerReport(
                layer=LayerType.INVISIBLE_COX,
                success=False,
                duration_ms=timer.duration_ms,
                message=f"Failed: {e}",
            )

    if output_path:
        from ..utils.image_io import save_image
        save_image(result, output_path)

    return result, report, bits_arr


def extract(
    image: np.ndarray,
    settings: CoxSettings,
    secret: bytes,
    n_bits: int = 64,
) -> np.ndarray:
    """提取 Cox 水印 bits (盲提取)

    Cox 嵌入时只用了 1 个 seed (settings.seed + secret[:4]_int),
    所以 extract 也必须用同一个 seed 才能可靠地读回那个 bit。
    然后把读到的 1 bit 复制 n_bits 次, 跟 embed 端对齐。
    """
    base_seed = settings.seed + int.from_bytes(secret[:4], "big")
    pn = _generate_pn_sequence(64 * 64, base_seed)
    extracted = _extract_from_luminance(image, pn, None)
    bit_value = int(extracted[0])
    return np.full(n_bits, bit_value, dtype=np.uint8)
