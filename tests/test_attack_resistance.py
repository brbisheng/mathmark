"""攻击抵抗测试

测试水印在各种攻击下的鲁棒性
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from mathmark import LayerType, WatermarkConfig, WatermarkPipeline
from mathmark.attacks import (
    add_noise,
    brightness_attack,
    caption_guided_regen_simulation,
    contrast_attack,
    crop_attack,
    douyin_compress,
    full_diffusion_attack_simulation,
    gaussian_blur_attack,
    jpeg_recompress,
    local_blur_attack,
    noise_regen_simulation,
    resize_attack,
    semantic_regen_simulation,
    wechat_compress,
    xiaohongshu_compress,
)
from mathmark.core.config import load_signature
from mathmark.utils.image_io import load_image_rgb


@pytest.fixture
def watermarked_image(tmp_path) -> np.ndarray:
    """创建带水印的测试图像"""
    rng = np.random.default_rng(42)
    img = rng.integers(50, 200, (512, 512, 3), dtype=np.uint8)

    sig = load_signature("signatures/default.json")
    cfg = WatermarkConfig(teacher_id="T-TEST", teacher_name="Test Teacher")
    cfg.semantic.signature = sig
    pipeline = WatermarkPipeline(cfg)
    result = pipeline.process(img)
    return result.image


class TestSocialMediaCompression:
    """社媒压缩测试"""

    def test_wechat_compression_robustness(self, watermarked_image):
        """微信压缩后水印应仍可检测"""
        attacked = wechat_compress(watermarked_image, quality=75)
        assert attacked.shape == watermarked_image.shape
        # 视觉上应该有变化 (压缩不是无损)
        diff = np.abs(attacked.astype(np.float32) - watermarked_image.astype(np.float32)).mean()
        assert diff > 0.5, "压缩未生效"

    def test_xiaohongshu_compression(self, watermarked_image):
        """小红书压缩"""
        attacked = xiaohongshu_compress(watermarked_image, quality=80)
        assert attacked.shape == watermarked_image.shape

    def test_douyin_compression(self, watermarked_image):
        """抖音压缩"""
        attacked = douyin_compress(watermarked_image)
        assert attacked.shape == watermarked_image.shape

    def test_jpeg_recompression(self, watermarked_image):
        """多次 JPEG 重压缩"""
        attacked = jpeg_recompress(watermarked_image, quality=70, n_iterations=3)
        assert attacked.shape == watermarked_image.shape


class TestGeometricAttacks:
    """几何攻击测试"""

    def test_resize_attack(self, watermarked_image):
        """尺寸缩放"""
        attacked = resize_attack(watermarked_image, scale=0.5)
        assert attacked.shape[0] == 256
        assert attacked.shape[1] == 256

    def test_crop_attack(self, watermarked_image):
        """边缘裁切"""
        attacked = crop_attack(watermarked_image, crop_ratio=0.1)
        # 裁切后尺寸应该略小
        assert attacked.shape[0] < watermarked_image.shape[0]
        assert attacked.shape[1] < watermarked_image.shape[1]


class TestDiffusionAttacks:
    """扩散攻击模拟 (SOTA 攻击)"""

    def test_semantic_regen(self, watermarked_image):
        """语义重生成模拟"""
        attacked = semantic_regen_simulation(watermarked_image, strength=0.7)
        assert attacked.shape == watermarked_image.shape

    def test_caption_guided_regen(self, watermarked_image):
        """视觉释义攻击"""
        attacked = caption_guided_regen_simulation(watermarked_image, n_iterations=2)
        assert attacked.shape == watermarked_image.shape

    def test_full_diffusion_attack(self, watermarked_image):
        """完整扩散攻击"""
        attacked = full_diffusion_attack_simulation(watermarked_image, seed=42)
        assert attacked.shape == watermarked_image.shape
        # 攻击后应该和原图有差异
        diff = np.abs(attacked.astype(np.float32) - watermarked_image.astype(np.float32)).mean()
        assert diff > 1.0, "扩散攻击未生效"

    def test_noise_regen(self, watermarked_image):
        """噪声再生攻击 (NeurIPS 2024 风格)"""
        attacked = noise_regen_simulation(watermarked_image, noise_level=0.05)
        assert attacked.shape == watermarked_image.shape


class TestLocalAttacks:
    """局部攻击测试"""

    def test_gaussian_blur(self, watermarked_image):
        """全图高斯模糊"""
        attacked = gaussian_blur_attack(watermarked_image, kernel_size=15, sigma=3.0)
        assert attacked.shape == watermarked_image.shape

    def test_local_blur_gradcam_style(self, watermarked_image):
        """GradCAM 风格局部模糊"""
        h, w = watermarked_image.shape[:2]
        # 攻击右下角 100x100 区域
        attacked = local_blur_attack(
            watermarked_image,
            bbox=(w - 150, h - 150, 100, 100),
        )
        assert attacked.shape == watermarked_image.shape

    def test_inpainting_attack(self, watermarked_image):
        """图像修复攻击"""
        # 创建 mask: 中心区域待修复
        h, w = watermarked_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cy, cx = h // 2, w // 2
        mask[cy-50:cy+50, cx-50:cx+50] = 1
        attacked = gaussian_blur_attack(
            watermarked_image, kernel_size=21, sigma=5.0,
            mask=mask.astype(np.float32),
        )
        assert attacked.shape == watermarked_image.shape


class TestNoiseAttacks:
    """噪声攻击测试"""

    def test_gaussian_noise(self, watermarked_image):
        """高斯噪声"""
        attacked = add_noise(watermarked_image, sigma=10.0)
        assert attacked.shape == watermarked_image.shape

    def test_brightness_attack(self, watermarked_image):
        """亮度调整"""
        attacked = brightness_attack(watermarked_image, factor=1.5)
        assert attacked.shape == watermarked_image.shape

    def test_contrast_attack(self, watermarked_image):
        """对比度调整"""
        attacked = contrast_attack(watermarked_image, factor=0.7)
        assert attacked.shape == watermarked_image.shape


class TestEndToEndRobustness:
    """端到端鲁棒性测试"""

    def test_pipiline_after_attack(self, watermarked_image):
        """攻击后重新跑 pipeline 应仍能工作"""
        attacked = wechat_compress(watermarked_image, quality=70)
        # 验证仍能运行 pipeline
        sig = load_signature("signatures/default.json")
        cfg = WatermarkConfig(teacher_id="T-TEST", teacher_name="Test Teacher")
        cfg.semantic.signature = sig
        pipeline = WatermarkPipeline(cfg)
        result = pipeline.process(attacked)
        assert result.total_duration_ms > 0
