"""Pipeline 编排

按顺序调用 6 层处理, 输出 WatermarkResult
支持单图、批处理、性能监控
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from ..crypto.hashing import sha256_bytes, sha256_bytes_raw
from ..crypto.keys import KeyPair, load_keypair
from ..layers import (
    l1_fingerprint,
    l2_visible,
    l3a_trustmark,
    l3b_dwt_dct_svd,
    l3c_cox_spread,
    l4_semantic,
    l5_metadata,
    l6_c2pa,
)
from ..utils.perf import get_memory_mb, measure_time
from .secret import bits_from_secret, derive_secret
from .types import (
    BenchmarkResult,
    LayerReport,
    LayerType,
    WatermarkConfig,
    WatermarkResult,
)

PathLike = Union[str, Path]


# Back-compat: pipeline used to expose `_derive_secret` as a private helper.
# Re-export under the old name so existing imports keep working; new code
# should import from `core.secret` directly.
_derive_secret = derive_secret


class WatermarkPipeline:
    """水印处理流水线"""

    def __init__(self, config: WatermarkConfig, keypair: Optional[KeyPair] = None):
        self.config = config
        self.keypair = keypair
        # 尝试加载密钥
        if keypair is None and config.teacher_private_key_path:
            try:
                self.keypair = load_keypair(config.teacher_private_key_path)
            except Exception:
                pass

    def process(
        self,
        image: np.ndarray,
        output_path: Optional[PathLike] = None,
        manifest_path: Optional[PathLike] = None,
        injection_report_path: Optional[PathLike] = None,
        teacher_text_content: Optional[str] = None,
        content_type=None,  # ContentType
        progress_callback: Optional[Callable[[LayerType, float], None]] = None,
    ) -> WatermarkResult:
        """处理单张图像

        Args:
            image: 输入图像 (H, W, 3) uint8
            output_path: 输出图像路径
            manifest_path: C2PA manifest 输出路径
            injection_report_path: L4 注入报告输出路径
            teacher_text_content: 老师源文本内容(若有)
            content_type: 内容类型
            progress_callback: 进度回调 (layer, fraction)
        """
        result = WatermarkResult(
            image=image,
            teacher_id=self.config.teacher_id,
            timestamp=datetime.now(),
        )
        secret = _derive_secret(self.config, teacher_text_content)
        total_start = time.perf_counter()
        n_layers = len(self.config.enabled_layers) or 1
        done_layers = 0

        def _progress(layer: LayerType, fraction: float = 1.0) -> None:
            nonlocal done_layers
            if progress_callback and fraction >= 1.0:
                done_layers += 1
                progress_callback(layer, done_layers / n_layers)

        # ================ L1: 指纹 ================
        if self.config.is_enabled(LayerType.FINGERPRINT):
            try:
                result.image, report, fp = l1_fingerprint.process(result.image)
                result.layer_reports[LayerType.FINGERPRINT] = report
                result.phash = fp.phash
                _progress(LayerType.FINGERPRINT)
            except Exception as e:
                result.layer_reports[LayerType.FINGERPRINT] = LayerReport(
                    layer=LayerType.FINGERPRINT, success=False, duration_ms=0,
                    message=f"Exception: {e}",
                )

        # ================ L2: 可见水印 ================
        if self.config.is_enabled(LayerType.VISIBLE):
            try:
                result.image, report = l2_visible.process(
                    result.image,
                    self.config.visible,
                    secret_key=secret,
                    teacher_id=self.config.teacher_id,
                    teacher_name=self.config.teacher_name,
                )
                result.layer_reports[LayerType.VISIBLE] = report
                _progress(LayerType.VISIBLE)
            except Exception as e:
                result.layer_reports[LayerType.VISIBLE] = LayerReport(
                    layer=LayerType.VISIBLE, success=False, duration_ms=0,
                    message=f"Exception: {e}",
                )

        # ================ L3a: TrustMark ================
        if self.config.is_enabled(LayerType.INVISIBLE_TRUSTMARK):
            try:
                result.image, report, bits = l3a_trustmark.process(
                    result.image,
                    self.config.trustmark,
                    secret=secret,
                )
                result.layer_reports[LayerType.INVISIBLE_TRUSTMARK] = report
                result.extracted_bits[LayerType.INVISIBLE_TRUSTMARK] = bits.tobytes() if len(bits) > 0 else b""
                _progress(LayerType.INVISIBLE_TRUSTMARK)
            except Exception as e:
                result.layer_reports[LayerType.INVISIBLE_TRUSTMARK] = LayerReport(
                    layer=LayerType.INVISIBLE_TRUSTMARK, success=False, duration_ms=0,
                    message=f"Exception: {e}",
                )

        # ================ L3b: DWT-DCT-SVD ================
        if self.config.is_enabled(LayerType.INVISIBLE_DWT):
            try:
                # 32 bits derived from the full per-(teacher, image) secret
                # (audit B9: short teacher_id produced mostly-zero bits)
                wm_bits_arr = np.frombuffer(bits_from_secret(secret, 32), dtype=np.uint8)[:32] % 2
                if len(wm_bits_arr) < 32:
                    wm_bits_arr = np.concatenate(
                        [wm_bits_arr, np.zeros(32 - len(wm_bits_arr), dtype=np.uint8)]
                    )
                result.image, report, bits = l3b_dwt_dct_svd.process(
                    result.image,
                    self.config.dwt,
                    wm_bits_arr,
                )
                result.layer_reports[LayerType.INVISIBLE_DWT] = report
                result.extracted_bits[LayerType.INVISIBLE_DWT] = bits.tobytes() if len(bits) > 0 else b""
                _progress(LayerType.INVISIBLE_DWT)
            except Exception as e:
                result.layer_reports[LayerType.INVISIBLE_DWT] = LayerReport(
                    layer=LayerType.INVISIBLE_DWT, success=False, duration_ms=0,
                    message=f"Exception: {e}",
                )

        # ================ L3c: Cox ================
        if self.config.is_enabled(LayerType.INVISIBLE_COX):
            try:
                result.image, report, bits = l3c_cox_spread.process(
                    result.image,
                    self.config.cox,
                    secret=secret,
                )
                result.layer_reports[LayerType.INVISIBLE_COX] = report
                result.extracted_bits[LayerType.INVISIBLE_COX] = bits.tobytes() if len(bits) > 0 else b""
                _progress(LayerType.INVISIBLE_COX)
            except Exception as e:
                result.layer_reports[LayerType.INVISIBLE_COX] = LayerReport(
                    layer=LayerType.INVISIBLE_COX, success=False, duration_ms=0,
                    message=f"Exception: {e}",
                )

        # ================ L4: 语义水印 ================
        if self.config.is_enabled(LayerType.SEMANTIC):
            try:
                result.image, report, sem_data = l4_semantic.process(
                    result.image,
                    self.config.semantic,
                    teacher_text_content=teacher_text_content,
                    content_type=content_type,
                    output_injection_report=injection_report_path,
                )
                result.layer_reports[LayerType.SEMANTIC] = report
                result.semantic_injection_log = sem_data
                _progress(LayerType.SEMANTIC)
            except Exception as e:
                result.layer_reports[LayerType.SEMANTIC] = LayerReport(
                    layer=LayerType.SEMANTIC, success=False, duration_ms=0,
                    message=f"Exception: {e}",
                )

        # ================ L5: 元数据 ================
        if self.config.is_enabled(LayerType.METADATA):
            try:
                # 构造签名 hash 用于写入 EXIF
                sig_hash = ""
                if self.config.semantic and self.config.semantic.signature:
                    sig_hash = sha256_bytes(
                        json.dumps(self.config.semantic.signature.to_dict()).encode("utf-8")
                    )[:32]
                result.image, report, meta_data = l5_metadata.process(
                    result.image,
                    self.config.metadata,
                    teacher_id=self.config.teacher_id,
                    signature_hash=sig_hash,
                )
                result.layer_reports[LayerType.METADATA] = report
                # 把 L5 构造的 EXIF bytes 留着, 后面 save_image 时用, 避免 numpy roundtrip 丢失
                result.exif_bytes = meta_data.get("exif_bytes")
                # 同理, XMP packet 也留给 save_image 写真正的 APP1/iTXt
                result.xmp_bytes = meta_data.get("xmp_bytes")
                _progress(LayerType.METADATA)
            except Exception as e:
                result.layer_reports[LayerType.METADATA] = LayerReport(
                    layer=LayerType.METADATA, success=False, duration_ms=0,
                    message=f"Exception: {e}",
                )

        # ================ L6: C2PA ================
        if self.config.is_enabled(LayerType.C2PA):
            try:
                semantic_sim = 0.0
                if LayerType.SEMANTIC in result.layer_reports:
                    semantic_sim = result.layer_reports[LayerType.SEMANTIC].metadata.get("similarity", 0.0)
                result.image, report, manifest_p = l6_c2pa.process(
                    result.image,
                    self.config.c2pa,
                    teacher_id=self.config.teacher_id,
                    teacher_name=self.config.teacher_name,
                    phash=result.phash,
                    semantic_sim=semantic_sim,
                    keypair=self.keypair,
                    output_manifest_path=manifest_path,
                )
                result.layer_reports[LayerType.C2PA] = report
                result.c2pa_manifest_path = manifest_p
                _progress(LayerType.C2PA)
            except Exception as e:
                result.layer_reports[LayerType.C2PA] = LayerReport(
                    layer=LayerType.C2PA, success=False, duration_ms=0,
                    message=f"Exception: {e}",
                )

        total_end = time.perf_counter()
        result.total_duration_ms = (total_end - total_start) * 1000.0

        # 保存输出
        if output_path:
            from ..utils.image_io import save_image
            save_image(
                result.image,
                output_path,
                quality=95,
                exif_bytes=getattr(result, "exif_bytes", None),
                xmp_bytes=getattr(result, "xmp_bytes", None),
            )

        return result

    def process_batch(
        self,
        images: List[np.ndarray],
        output_paths: Optional[List[PathLike]] = None,
        progress_callback: Optional[Callable[[int, int, WatermarkResult], None]] = None,
        manifest_paths: Optional[List[Optional[PathLike]]] = None,
    ) -> List[WatermarkResult]:
        """批处理多张图像

        Args:
            images: 输入图像列表
            output_paths: 输出图像路径 (与 images 同长度, None 表示不写盘)
            progress_callback: 进度回调 (i+1, n, result)
            manifest_paths: audit B15 — 每张图的 C2PA manifest 输出路径,
                缺省时按 output_path 推导为 <output>.manifest.json
        """
        results = []
        n = len(images)
        for i, img in enumerate(images):
            out_path = output_paths[i] if output_paths else None
            if manifest_paths is not None:
                mpath = manifest_paths[i]
            elif out_path is not None:
                mpath = Path(out_path).with_suffix(Path(out_path).suffix + ".manifest.json")
            else:
                mpath = None
            result = self.process(
                img,
                output_path=out_path,
                manifest_path=mpath,
            )
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, n, result)
        return results

    def benchmark(
        self,
        image: np.ndarray,
        n_iterations: int = 3,
    ) -> BenchmarkResult:
        """性能基准测试"""
        # audit B22: 之前 n_iterations=0 会让 mean_duration_ms 用未赋值的 durations 数组算, 抛 NameError
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")

        import tracemalloc
        durations = []
        layer_breakdown: dict[LayerType, float] = {}

        # audit B16: 之前 memory_peak_mb 是 mem_after - mem_before (delta),
        # 内存被释放时会变成负数. 改用 tracemalloc 取真正的峰值.
        tracemalloc.start()
        try:
            for i in range(n_iterations):
                result = self.process(image.copy())
                durations.append(result.total_duration_ms)
                for layer, report in result.layer_reports.items():
                    layer_breakdown[layer] = layer_breakdown.get(layer, 0) + report.duration_ms
                if i == 0:
                    layer_breakdown = {k: v for k, v in layer_breakdown.items()}
            _, peak_bytes = tracemalloc.get_traced_memory()
            peak_mb = peak_bytes / (1024 * 1024)
        finally:
            tracemalloc.stop()

        # 平均分解
        layer_breakdown = {k: v / n_iterations for k, v in layer_breakdown.items()}

        durations_arr = np.array(durations)
        return BenchmarkResult(
            image_size=image.shape[:2],
            n_iterations=n_iterations,
            mean_duration_ms=float(durations_arr.mean()),
            std_duration_ms=float(durations_arr.std()),
            min_duration_ms=float(durations_arr.min()),
            max_duration_ms=float(durations_arr.max()),
            memory_peak_mb=peak_mb,
            layer_breakdown=layer_breakdown,
        )
