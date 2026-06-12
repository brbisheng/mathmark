"""局部攻击模拟 - GradCAM 局部模糊, 修复攻击等"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def gaussian_blur_attack(
    image: np.ndarray,
    kernel_size: int = 15,
    sigma: float = 3.0,
    mask: np.ndarray = None,
) -> np.ndarray:
    """高斯模糊攻击 (可指定 mask 做局部)"""
    blurred = cv2.GaussianBlur(image.astype(np.float32), (kernel_size, kernel_size), sigma)
    if mask is None:
        return np.clip(blurred, 0, 255).astype(np.uint8)
    # 应用 mask
    mask_3d = np.stack([mask] * 3, axis=-1) if mask.ndim == 2 else mask
    result = image.astype(np.float32) * (1 - mask_3d) + blurred * mask_3d
    return np.clip(result, 0, 255).astype(np.uint8)


def local_blur_attack(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    kernel_size: int = 21,
) -> np.ndarray:
    """GradCAM 风格的局部模糊攻击 - 只模糊指定 bbox

    Args:
        image: 输入图像
        bbox: (x, y, w, h) 模糊区域
        kernel_size: 模糊核
    """
    x, y, w, h = bbox
    result = image.copy()
    roi = result[y:y+h, x:x+w]
    blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 5.0)
    result[y:y+h, x:x+w] = blurred_roi
    return result


def inpainting_attack(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
) -> np.ndarray:
    """图像修复攻击 - 模拟 LaMa/MAT 修复

    Args:
        image: 输入图像 (RGB)
        mask: 0/1 区域 (1=待修复)
        method: "telea" / "ns"
    """
    if mask.ndim == 2:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)

    if method == "telea":
        result = cv2.inpaint(image.astype(np.uint8), mask_uint8, 3, cv2.INPAINT_TELEA)
    elif method == "ns":
        result = cv2.inpaint(image.astype(np.uint8), mask_uint8, 3, cv2.INPAINT_NS)
    else:
        raise ValueError(f"Unknown method: {method}")
    return result


def add_noise(image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """高斯噪声攻击"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def brightness_attack(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """亮度调整攻击"""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def contrast_attack(image: np.ndarray, factor: float = 0.7) -> np.ndarray:
    """对比度调整攻击"""
    mean = image.mean()
    result = mean + factor * (image.astype(np.float32) - mean)
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# GradCAM 启发式: 找到图像中"最可能被水印"的区域
# ============================================================

def gradcam_heuristic_mask(
    original: np.ndarray,
    watermarked: np.ndarray,
    threshold: float = 0.95,
) -> np.ndarray:
    """启发式: 找到与原图差异最大的区域

    用于模拟 GradCAM 风格的攻击 (假设攻击者知道差异大的地方可能是水印)
    """
    diff = np.abs(watermarked.astype(np.float32) - original.astype(np.float32)).mean(axis=-1)
    if diff.max() > 0:
        diff = diff / diff.max()
    return (diff > threshold).astype(np.uint8)
