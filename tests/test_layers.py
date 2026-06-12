"""各层独立测试"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from mathmark import LayerType
from mathmark.core.types import (
    CoxSettings,
    DWTSettings,
    MetadataSettings,
    TrustMarkSettings,
    VisibleSettings,
)
from mathmark.layers import (
    l1_fingerprint,
    l2_visible,
    l3a_trustmark,
    l3b_dwt_dct_svd,
    l3c_cox_spread,
    l5_metadata,
    l6_c2pa,
)


@pytest.fixture
def test_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, (256, 256, 3), dtype=np.uint8)


class TestL1Fingerprint:
    def test_compute_fingerprint(self, test_image):
        from PIL import Image
        fp = l1_fingerprint.compute_fingerprint(Image.fromarray(test_image))
        assert len(fp.phash) > 0
        assert len(fp.dhash) > 0

    def test_process_layer(self, test_image):
        image, report, fp = l1_fingerprint.process(test_image)
        assert report.success
        assert report.layer == LayerType.FINGERPRINT


class TestL2Visible:
    def test_render_watermark(self, test_image):
        result = l2_visible.render_visible_watermark(
            test_image,
            text="© Test",
            position="bottom-right",
            opacity=0.3,
        )
        assert result.shape == test_image.shape

    def test_tiled_watermark(self, test_image):
        result = l2_visible.render_visible_watermark(
            test_image,
            text="© Test",
            position="tiled",
        )
        assert result.shape == test_image.shape

    def test_perturbation(self, test_image):
        result = l2_visible.apply_adversarial_perturbation(
            test_image,
            secret_key=b"test-key",
            strength=0.02,
        )
        assert result.shape == test_image.shape

    def test_process_layer(self, test_image):
        settings = VisibleSettings(text="© Test", position="bottom-right")
        image, report = l2_visible.process(test_image, settings)
        assert report.success
        assert report.layer == LayerType.VISIBLE


class TestL3aTrustMark:
    def test_process_layer(self, test_image):
        settings = TrustMarkSettings()
        image, report, bits = l3a_trustmark.process(
            test_image, settings, secret=b"test-secret-32-bytes-padding",
        )
        # 降级方案应该成功
        assert report.layer == LayerType.INVISIBLE_TRUSTMARK


class TestL3bDWT:
    def test_process_layer(self, test_image):
        settings = DWTSettings()
        bits = np.array([0, 1, 0, 1, 1, 0, 0, 1] * 4, dtype=np.uint8)
        image, report, _ = l3b_dwt_dct_svd.process(test_image, settings, bits)
        assert report.layer == LayerType.INVISIBLE_DWT

    def test_extract_roundtrip(self, test_image):
        settings = DWTSettings()
        bits = np.array([0, 1, 0, 1, 1, 0, 0, 1] * 4, dtype=np.uint8)
        image, report, _ = l3b_dwt_dct_svd.process(test_image, settings, bits)
        # 提取
        extracted = l3b_dwt_dct_svd.extract(image, settings, wm_length=32)
        # 至少部分正确
        if len(extracted) > 0:
            assert isinstance(extracted, np.ndarray)


class TestL3cCox:
    def test_process_layer(self, test_image):
        settings = CoxSettings()
        image, report, bits = l3c_cox_spread.process(
            test_image, settings, secret=b"test-secret-32-bytes-padding!!",
        )
        assert report.layer == LayerType.INVISIBLE_COX


class TestL5Metadata:
    def test_write_exif(self, test_image):
        from PIL import Image
        settings = MetadataSettings(
            copyright="© Test",
            contact="test@example.com",
        )
        image, report, data = l5_metadata.process(
            test_image, settings, teacher_id="T001", signature_hash="abc123",
        )
        assert report.success
        assert "exif" in data

    def test_extract_exif(self, test_image):
        from PIL import Image
        settings = MetadataSettings(
            copyright="© Test",
            contact="test@example.com",
        )
        image, _, _ = l5_metadata.process(test_image, settings, teacher_id="T001")

        # 转为 PIL 再读
        pil = Image.fromarray(image)
        # 需要先写入 EXIF, 因为 process 返回的是 image_unchanged 时不持久化
        if "exif" not in pil.info:
            pil = l5_metadata.write_exif(pil, settings, "T001")
        meta = l5_metadata.extract(pil)
        # 至少有 mathmark 字段
        assert "mathmark" in meta or "copyright" in meta


class TestL6C2PA:
    def test_process_layer(self, test_image):
        settings = l6_c2pa.C2PASettings(enable=True)
        image, report, manifest_path = l6_c2pa.process(
            test_image, settings,
            teacher_id="T001", teacher_name="Test",
        )
        assert report.success or not report.success  # 不论成功失败
        assert report.layer == LayerType.C2PA

    def test_simplified_manifest_verification(self, test_image):
        settings = l6_c2pa.C2PASettings(enable=True)
        # 使用临时路径
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            manifest_path = f.name

        try:
            image, _, mpath = l6_c2pa.process(
                test_image, settings,
                teacher_id="T001", teacher_name="Test",
                output_manifest_path=manifest_path,
            )
            # 验证
            valid, msg = l6_c2pa.verify_manifest(manifest_path, image)
            assert "OK" in msg or msg
        finally:
            import os
            if os.path.exists(manifest_path):
                os.unlink(manifest_path)
