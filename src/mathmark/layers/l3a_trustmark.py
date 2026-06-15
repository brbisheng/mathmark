"""L3a TrustMark 不可见水印

Adobe TrustMark (2024) - 一种鲁棒的通用图像水印方法
使用 ONNX 模型在 CPU 上推理,100-bit payload + BCH ECC

依赖:
    pip install onnxruntime trustmark
    # 需要下载预训练模型到 models/trustmark.onnx

如果未安装,自动降级到基于 DWT-DCT 的轻量级方案。
"""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ..core.types import LayerReport, LayerType, TrustMarkSettings
from ..utils.perf import measure_time

PathLike = Union[str, Path]

# 尝试导入 trustmark
try:
    import trustmark
    TRUSTMARK_AVAILABLE = True
except ImportError:
    TRUSTMARK_AVAILABLE = False


# TrustMark 公开的模型 URL
TRUSTMARK_MODEL_URL = "https://github.com/adobe/trustmark/releases/download/v0.5/trustmark.onnx"


def _ensure_model(model_path: PathLike) -> Path:
    """确保模型存在,否则下载"""
    model_path = Path(model_path)
    if model_path.exists():
        return model_path

    # 尝试自动下载
    try:
        import urllib.request
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading TrustMark model to {model_path}...")
        urllib.request.urlretrieve(TRUSTMARK_MODEL_URL, str(model_path))
        return model_path
    except Exception as e:
        raise FileNotFoundError(
            f"TrustMark model not found at {model_path} and auto-download failed: {e}\n"
            f"Please download from {TRUSTMARK_MODEL_URL} and place at {model_path}"
        )


def _secret_to_bits(secret: bytes, n_bits: int = 100) -> np.ndarray:
    """将 secret bytes 转换为固定长度的 bit 数组"""
    # 用 SHA-256 扩展为足够长度
    expanded = secret
    while len(expanded) * 8 < n_bits:
        expanded = hashlib.sha256(expanded).digest() + expanded
    # 取前 n_bits 位
    bits = []
    for byte in expanded:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
            if len(bits) >= n_bits:
                return np.array(bits, dtype=np.uint8)
    return np.array(bits[:n_bits], dtype=np.uint8)


def _bits_to_secret(bits: np.ndarray, n_bytes: int = 16) -> bytes:
    """bit 数组转回 secret bytes"""
    # 截断到 8 的倍数
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    result = bytearray()
    for i in range(0, n, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i + j])
        result.append(byte)
    return bytes(result[:n_bytes])


# ============================================================
# TrustMark 后端
# ============================================================

class TrustMarkBackend:
    """TrustMark 模型包装 (单例,避免重复加载)

    Audit B17: the previous class-level singleton meant switching model_path
    between calls was silently ignored (only the first caller's model stayed
    loaded). Key the cache by resolved model path so different models can
    coexist; reset only on a real model swap.
    """

    # path -> instance
    _instances: dict[str, "TrustMarkBackend"] = {}

    def __init__(self, model_path: PathLike):
        if not TRUSTMARK_AVAILABLE:
            raise RuntimeError("trustmark library not available")

        self.model_path = _ensure_model(model_path)
        # trustmark 库加载模型
        self.tm = trustmark.TrustMark(
            encoding_type=trustmark.TrustMark.Encoding.BCH_5,
            model_path=str(self.model_path),
        )

    @classmethod
    def get_instance(cls, model_path: Optional[PathLike] = None) -> "TrustMarkBackend":
        if model_path is None:
            # 默认路径
            model_path = Path.home() / ".mathmark" / "models" / "trustmark.onnx"
        # 用 resolve 后的绝对路径作为 key, 避免相对/绝对/符号链接被当作不同的模型
        key = str(Path(model_path).expanduser().resolve())
        if key not in cls._instances:
            cls._instances[key] = cls(model_path)
        return cls._instances[key]

    def embed(self, image: np.ndarray, payload_bits: np.ndarray) -> np.ndarray:
        """嵌入水印"""
        from PIL import Image
        pil_img = Image.fromarray(image.astype(np.uint8))
        # TrustMark 需要 PIL Image
        watermarked = self.tm.encode(pil_img, payload_bits, wm_app="mathmark")
        return np.array(watermarked, dtype=np.uint8)

    def decode(self, image: np.ndarray, n_bits: int = 100) -> np.ndarray:
        """提取水印 bits"""
        from PIL import Image
        pil_img = Image.fromarray(image.astype(np.uint8))
        # 尝试多种 wm_app 以兼容
        try:
            payload, _ = self.tm.decode(pil_img, wm_app="mathmark")
        except Exception:
            # 不带 wm_app 尝试
            payload, _ = self.tm.decode(pil_img)
        return np.array(payload, dtype=np.uint8)


# ============================================================
# 降级方案:简化版 DCT 扩频 (无需 GPU/外部模型)
# ============================================================

def _dct2d_8x8(block: np.ndarray) -> np.ndarray:
    """8x8 DCT"""
    from scipy.fft import dct, idct
    return dct(dct(block.astype(np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2d_8x8(block: np.ndarray) -> np.ndarray:
    """8x8 逆 DCT"""
    from scipy.fft import dct, idct
    return idct(idct(block.astype(np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")


def _fallback_embed(
    image: np.ndarray,
    payload_bits: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """降级方案:基于 8x8 DCT 中频的水印嵌入

    抗 JPEG 压缩, 抗轻量噪声。

    注意: 此 fallback 仅供 trustmark 模型不可用时降级使用. 实际生效请装:
        pip install trustmark onnxruntime
        # 并下载 trustmark.onnx 到 ~/.mathmark/models/
    Fallback 在 PNG 保存/读取后 BER 较高 (~0.4), 不足以作为
    唯一证据, 仅作为 'MathMark 处理过' 的弱信号.
    """
    h, w = image.shape[:2]
    if image.ndim == 3:
        # 只在 Y 通道 (luminance) 嵌入
        Y = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        Y = image.astype(np.float64)

    # 嵌入到中频系数 (位置 3,4 / 4,3)
    bit_idx = 0
    n_bits = len(payload_bits)
    h8, w8 = h // 8, w // 8
    total_blocks = h8 * w8

    if n_bits > total_blocks:
        # 用 PRNG 选择位置
        rng = np.random.default_rng(42)
        positions = rng.choice(total_blocks, n_bits, replace=False)
    else:
        positions = range(n_bits)

    watermarked_Y = Y.copy()
    for pos in positions:
        by, bx = pos // w8, pos % w8
        block = Y[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8].copy()
        dct_block = _dct2d_8x8(block)

        # 中频系数: (4, 5) 和 (5, 4) - 抗 JPEG 的中频带
        c1 = dct_block[3, 4]
        c2 = dct_block[4, 3]
        bit = int(payload_bits[bit_idx])
        bit_idx += 1

        if bit == 1:
            if c1 < c2:
                dct_block[3, 4], dct_block[4, 3] = c2 + alpha, c1 - alpha
        else:
            if c1 > c2:
                dct_block[3, 4], dct_block[4, 3] = c2 - alpha, c1 + alpha

        idct_block = _idct2d_8x8(dct_block)
        watermarked_Y[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8] = idct_block
        if bit_idx >= n_bits:
            break

    # 重新合成 RGB
    if image.ndim == 3:
        # Audit B10: 之前用 multiplicative scale 把 Y 差扩散到 R/G/B,
        # 对彩色图像会把饱和的色块推得偏色/溢出. 改用 additive delta,
        # 让 L3b/L4 不被 Y-channel 之外的二次扰动影响 (CLAUDE.md 也不准
        # 在 L3a fallback 里写 per-channel). delta 加到所有通道保留色相.
        delta = (watermarked_Y - Y)
        result = (image.astype(np.float64) + delta[..., None]).clip(0, 255).astype(np.uint8)
        return result
    else:
        return watermarked_Y.astype(np.uint8)


def _fallback_decode(image: np.ndarray, n_bits: int) -> np.ndarray:
    """降级方案:提取 DCT 水印 bits"""
    h, w = image.shape[:2]
    if image.ndim == 3:
        Y = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        Y = image.astype(np.float64)

    h8, w8 = h // 8, w // 8
    total_blocks = h8 * w8
    n_bits = min(n_bits, total_blocks)

    # 用相同 PRNG 重建位置
    rng = np.random.default_rng(42)
    if n_bits > total_blocks:
        positions = rng.choice(total_blocks, n_bits, replace=False)
    else:
        positions = range(n_bits)

    bits = []
    for pos in positions:
        by, bx = pos // w8, pos % w8
        block = Y[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8]
        dct_block = _dct2d_8x8(block)
        c1 = dct_block[3, 4]
        c2 = dct_block[4, 3]
        bits.append(1 if c1 > c2 else 0)

    return np.array(bits, dtype=np.uint8)


# ============================================================
# Layer 接口实现
# ============================================================

@dataclass
class L3AResult:
    image: np.ndarray
    bits: np.ndarray
    used_fallback: bool


def process(
    image: np.ndarray,
    settings: TrustMarkSettings,
    secret: bytes,
    n_bits: int = 100,
    output_path: Optional[PathLike] = None,
) -> Tuple[np.ndarray, LayerReport, np.ndarray]:
    """L3a TrustMark 处理

    优先使用 Adobe TrustMark 模型, 否则使用降级 DCT 方案。
    """
    payload_bits = _secret_to_bits(secret, n_bits)

    with measure_time("L3a_trustmark") as timer:
        try:
            if TRUSTMARK_AVAILABLE and settings.model_path is not None:
                backend = TrustMarkBackend.get_instance(settings.model_path)
                result = backend.embed(image, payload_bits)
                used_fallback = False
                message = f"TrustMark encoded, {n_bits} bits"
            else:
                # 降级方案
                result = _fallback_embed(image, payload_bits, alpha=0.1)
                used_fallback = True
                if not TRUSTMARK_AVAILABLE:
                    message = "DCT fallback (trustmark lib not installed)"
                else:
                    message = "DCT fallback (no model path)"
        except Exception as e:
            result = image
            payload_bits = np.array([], dtype=np.uint8)
            used_fallback = True
            report = LayerReport(
                layer=LayerType.INVISIBLE_TRUSTMARK,
                success=False,
                duration_ms=timer.duration_ms,
                message=f"Failed: {e}",
            )
            if output_path:
                from ..utils.image_io import save_image
                save_image(result, output_path)
            return result, report, payload_bits

    if used_fallback:
        # 验证提取
        try:
            decoded = _fallback_decode(result, n_bits)
            ber = np.mean(decoded != payload_bits)
        except Exception:
            ber = -1
        message = f"{message} (BER after embed: {ber:.3f})"
    else:
        ber = -1
        message = f"{message} (validation deferred to verify)"

    report = LayerReport(
        layer=LayerType.INVISIBLE_TRUSTMARK,
        success=True,
        duration_ms=timer.duration_ms,
        message=message,
        metadata={
            "n_bits": n_bits,
            "used_fallback": used_fallback,
            "embed_ber": ber,
        },
    )

    if output_path:
        from ..utils.image_io import save_image
        save_image(result, output_path)

    return result, report, payload_bits


def extract(
    image: np.ndarray,
    settings: TrustMarkSettings,
    n_bits: int = 100,
) -> Optional[bytes]:
    """提取 L3a 水印 bits"""
    try:
        if TRUSTMARK_AVAILABLE and settings.model_path is not None:
            backend = TrustMarkBackend.get_instance(settings.model_path)
            bits = backend.decode(image, n_bits)
        else:
            bits = _fallback_decode(image, n_bits)
        return _bits_to_secret(bits)
    except Exception:
        return None
