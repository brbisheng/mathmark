"""扩散重生成攻击模拟

NeurIPS 2024 挑战赛冠军方案: VAE 规避 + 干净噪声扩散再生
本模块是简化模拟,使用低成本的图像处理模拟"扩散重生成"的效果。

注意: 真正的扩散模型重生成需要 GPU 和大模型,这里用 OpenCV 模拟
"语义改变但视觉相似"的效果,用于评估传统水印的脆弱性。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def _edge_preserving_smoothing(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """边缘保持平滑 - 模拟扩散平滑"""
    return cv2.bilateralFilter(image.astype(np.uint8), d, sigma_color, sigma_space)


def semantic_regen_simulation(
    image: np.ndarray,
    strength: float = 0.7,
) -> np.ndarray:
    """模拟扩散重生成 - 用传统图像处理近似

    步骤:
    1. 边缘保持平滑 (模拟扩散的语义去噪)
    2. 颜色直方图匹配 (模拟语义色彩稳定)
    3. 局部纹理合成 (模拟细节再生)

    Args:
        image: 输入图像
        strength: 重生成强度 (0-1)
    """
    result = image.astype(np.float32)

    # 1. 边缘保持平滑
    smoothed = _edge_preserving_smoothing(image)

    # 2. 与原图混合
    result = (1 - strength) * image.astype(np.float32) + strength * smoothed.astype(np.float32)

    # 3. 颜色微调 (模拟扩散的色彩稳定化)
    result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
    # 轻微颜色扰动
    hsv = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] += np.random.normal(0, 2, hsv[..., 0].shape) * strength  # 色调微调
    hsv[..., 1] *= 1.0 + np.random.normal(0, 0.05, hsv[..., 1].shape) * strength  # 饱和度
    # 亮度通道保持(简化,避免广播问题)
    hsv[..., 2] = hsv[..., 2]
    hsv = np.clip(hsv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def caption_guided_regen_simulation(
    image: np.ndarray,
    n_iterations: int = 3,
) -> np.ndarray:
    """模拟"视觉释义"攻击 - 多次平滑+锐化+颜色抖动"""
    result = image.astype(np.float32)
    for i in range(n_iterations):
        # 平滑
        smoothed = _edge_preserving_smoothing(result.astype(np.uint8), d=5, sigma_color=50, sigma_space=50)
        # 锐化
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 1.5
        sharpened = cv2.filter2D(smoothed, -1, kernel)
        # 混合
        result = 0.5 * result + 0.5 * sharpened.astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


def full_diffusion_attack_simulation(
    image: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """完整的"扩散攻击"模拟 - 组合多种效果"""
    np.random.seed(seed)

    # 1. 边缘保持平滑 (主要的去水印效果)
    result = _edge_preserving_smoothing(image, d=15, sigma_color=100, sigma_space=100)

    # 2. 模拟局部"重绘"
    h, w = result.shape[:2]
    block_size = 64
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            # 在小块上做平滑+微调
            block = result[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue
            # 简单均值滤波
            block_smooth = cv2.blur(block, (3, 3))
            result[y:y+block_size, x:x+block_size] = block_smooth

    # 3. 颜色直方图均衡 (改变整体色调)
    ycrcb = cv2.cvtColor(result, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    return result


def noise_regen_simulation(
    image: np.ndarray,
    noise_level: float = 0.05,
    denoise_strength: int = 7,
) -> np.ndarray:
    """模拟"从噪声再生" - 加噪声再去噪"""
    # 1. 加噪
    noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 2. 非局部均值去噪 (模拟扩散的去噪过程)
    result = cv2.fastNlMeansDenoisingColored(noisy, None, denoise_strength, denoise_strength, 7, 21)

    return result
