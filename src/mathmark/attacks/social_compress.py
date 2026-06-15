"""社媒压缩模拟 - 模拟各平台的图像处理"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def wechat_compress(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """模拟微信公众号压缩

    微信特征:
    - JPEG 重压缩 (Q≈75)
    - 限制最大宽度 1080px
    - 颜色降级
    """
    pil = Image.fromarray(image.astype(np.uint8))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    # 限制宽度
    max_w = 1080
    if pil.width > max_w:
        ratio = max_w / pil.width
        new_size = (max_w, int(pil.height * ratio))
        pil = pil.resize(new_size, Image.LANCZOS)

    # JPEG 重压缩
    import io
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    result = Image.open(buf)
    return np.array(result.convert("RGB"), dtype=np.uint8)


def xiaohongshu_compress(image: np.ndarray, quality: int = 80) -> np.ndarray:
    """模拟小红书压缩

    小红书特征:
    - 3:4 比例裁切(可选)
    - JPEG 质量约 80
    - 锐化滤镜
    """
    pil = Image.fromarray(image.astype(np.uint8))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    # 锐化
    pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # JPEG 压缩
    import io
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def douyin_compress(image: np.ndarray, quality: int = 78) -> np.ndarray:
    """模拟抖音/今日头条压缩"""
    return wechat_compress(image, quality=quality)  # 算法类似


def weibo_compress(image: np.ndarray, quality: int = 85) -> np.ndarray:
    """模拟微博压缩"""
    return wechat_compress(image, quality=quality)


def jpeg_recompress(image: np.ndarray, quality: int = 70, n_iterations: int = 1) -> np.ndarray:
    """多次 JPEG 重压缩"""
    import io
    result = image.astype(np.uint8)
    for _ in range(n_iterations):
        pil = Image.fromarray(result)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        result = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return result


def resize_attack(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """尺寸缩放攻击"""
    if not (0 < scale <= 1):
        raise ValueError(f"resize_attack: scale must be in (0, 1], got {scale}")
    pil = Image.fromarray(image.astype(np.uint8))
    new_size = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
    pil = pil.resize(new_size, Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


def crop_attack(image: np.ndarray, crop_ratio: float = 0.1) -> np.ndarray:
    """边缘裁切攻击"""
    if not (0 <= crop_ratio < 0.5):
        raise ValueError(f"crop_attack: crop_ratio must be in [0, 0.5), got {crop_ratio}")
    h, w = image.shape[:2]
    cy, cx = int(h * crop_ratio), int(w * crop_ratio)
    if h - 2 * cy <= 0 or w - 2 * cx <= 0:
        raise ValueError(
            f"crop_attack: image too small ({h}x{w}) for crop_ratio {crop_ratio}"
        )
    return image[cy:h-cy, cx:w-cx]


# 延迟导入 ImageFilter
from PIL import ImageFilter  # noqa: E402
