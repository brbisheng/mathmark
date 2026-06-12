"""端到端 Pipeline 测试"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from mathmark import LayerType, WatermarkConfig, WatermarkPipeline
from mathmark.core.config import load_signature
from mathmark.utils.image_io import save_image


@pytest.fixture
def test_image() -> np.ndarray:
    """创建测试图像"""
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def basic_config() -> WatermarkConfig:
    """基础配置"""
    sig = load_signature("signatures/default.json")
    cfg = WatermarkConfig(teacher_id="T-TEST", teacher_name="Test Teacher")
    cfg.semantic.signature = sig
    return cfg


class TestPipeline:
    """测试 6 层 Pipeline"""

    def test_all_layers_run(self, test_image, basic_config):
        """所有层都应执行成功"""
        pipeline = WatermarkPipeline(basic_config)
        result = pipeline.process(test_image)
        assert len(result.layer_reports) == 8
        # L1 L2 L5 L6 应该总是成功
        assert result.layer_reports[LayerType.FINGERPRINT].success
        assert result.layer_reports[LayerType.VISIBLE].success
        assert result.layer_reports[LayerType.METADATA].success
        # L4 取决于 OCR, mock 模式下也算成功
        assert result.layer_reports[LayerType.SEMANTIC].success

    def test_optional_layers(self, test_image, basic_config):
        """选择性启用层"""
        basic_config.enabled_layers = {
            LayerType.FINGERPRINT,
            LayerType.VISIBLE,
        }
        pipeline = WatermarkPipeline(basic_config)
        result = pipeline.process(test_image)
        assert len(result.layer_reports) == 2

    def test_phash_present(self, test_image, basic_config):
        """L1 pHash 应当存在"""
        pipeline = WatermarkPipeline(basic_config)
        result = pipeline.process(test_image)
        assert result.phash is not None
        assert len(result.phash) > 0

    def test_invisible_watermark_bits(self, test_image, basic_config):
        """L3 不可见水印 bits 应当可提取"""
        pipeline = WatermarkPipeline(basic_config)
        result = pipeline.process(test_image)
        # L3a/L3b 应当有 bits
        if LayerType.INVISIBLE_TRUSTMARK in result.extracted_bits:
            assert len(result.extracted_bits[LayerType.INVISIBLE_TRUSTMARK]) > 0

    def test_output_image_shape(self, test_image, basic_config):
        """输出图像形状应保持"""
        pipeline = WatermarkPipeline(basic_config)
        result = pipeline.process(test_image)
        assert result.image.shape == test_image.shape
        assert result.image.dtype == np.uint8

    def test_total_duration(self, test_image, basic_config):
        """总耗时应在合理范围"""
        pipeline = WatermarkPipeline(basic_config)
        result = pipeline.process(test_image)
        # 256x256 图, 应该在 5 秒内
        assert result.total_duration_ms < 5000

    def test_save_and_load(self, test_image, basic_config, tmp_path):
        """保存后能加载"""
        output_path = tmp_path / "test_output.png"
        pipeline = WatermarkPipeline(basic_config)
        result = pipeline.process(test_image, output_path=output_path)
        assert output_path.exists()
        # 加载验证
        from PIL import Image
        loaded = np.array(Image.open(output_path))
        assert loaded.shape[2] == 3


class TestBenchmark:
    """性能基准测试"""

    def test_benchmark_runs(self, test_image, basic_config):
        """基准测试应能运行"""
        pipeline = WatermarkPipeline(basic_config)
        result = pipeline.benchmark(test_image, n_iterations=2)
        assert result.mean_duration_ms > 0
        assert result.n_iterations == 2
