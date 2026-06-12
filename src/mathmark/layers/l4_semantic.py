"""L4 数学语义水印层

这是 MathMark 的核心创新层:
- 不修改图像像素
- 不依赖难复现的深度学习模型
- 通过识别数学老师的"教学风格"指纹, 建立不可伪造的归属

工作流程:
1. 加水印时: 生成签名建议 (供老师参考) + 提取图像中的现有签名特征
2. 验证时: OCR 识别图中内容, 与配置的签名比对, 计算相似度

注意: 这一层实际上不会"修改"图像, 而是:
- 在 metadata 中记录签名配置(供后续验证)
- 生成"建议清单"给老师
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ..core.types import LayerReport, LayerType, SemanticSettings
from ..semantic.recognizer import (
    RecognitionResult,
    recognize_from_image,
    recognize_from_text,
)
from ..semantic.injector import InjectionReport, create_injection_report
from ..utils.perf import measure_time

PathLike = Union[str, Path]


def process(
    image: np.ndarray,
    settings: SemanticSettings,
    teacher_text_content: Optional[str] = None,
    content_type=None,  # ContentType enum
    output_injection_report: Optional[PathLike] = None,
) -> Tuple[np.ndarray, LayerReport, dict]:
    """L4 语义水印处理

    这一层不修改图像,而是:
    1. 提取图像中已有的签名特征(用于取证)
    2. 生成签名建议报告(给老师参考)

    Args:
        image: 输入图像
        settings: L4 配置
        teacher_text_content: 老师提供的源文本(若为 PPT/LaTeX/MD 内容)
        content_type: 内容类型
        output_injection_report: 注入报告输出路径

    Returns:
        (image_unchanged, report, result_dict)
    """
    with measure_time("L4_semantic") as timer:
        result_dict = {
            "recognition": None,
            "injection_report": None,
        }

        if settings.signature is None:
            report = LayerReport(
                layer=LayerType.SEMANTIC,
                success=False,
                duration_ms=timer.duration_ms,
                message="No signature configured - L4 skipped",
                metadata={},
            )
            return image, report, result_dict

        try:
            # 步骤1: 识别图中现有签名
            recognition = recognize_from_image(
                image,
                settings.signature,
                ocr_engine="auto",
            )
            result_dict["recognition"] = recognition.to_dict()

            # 步骤2: 如果有源文本, 也做一次识别
            if teacher_text_content:
                text_recognition = recognize_from_text(
                    teacher_text_content,
                    settings.signature,
                )
                result_dict["text_recognition"] = text_recognition.to_dict()

            # 步骤3: 生成签名建议
            from ..core.types import ContentType
            ct = content_type or ContentType.MIXED
            injection_report = create_injection_report(
                content_type=ct,
                signature=settings.signature,
                topic=None,
                auto_inject_log=None,
            )
            result_dict["injection_report"] = injection_report.to_dict()

            if output_injection_report:
                from ..semantic.injector import save_injection_report
                save_injection_report(injection_report, output_injection_report)

            # 决定 success
            sim = recognition.overall_similarity
            message = (
                f"Signature recognition: similarity={sim:.3f}, "
                f"verdict={recognition.verdict}, "
                f"n_evidence={len(recognition.evidence)}"
            )

            report = LayerReport(
                layer=LayerType.SEMANTIC,
                success=True,
                duration_ms=timer.duration_ms,
                message=message,
                metadata={
                    "similarity": sim,
                    "verdict": recognition.verdict,
                    "teacher_id": settings.signature.teacher_id,
                    "n_evidence": len(recognition.evidence),
                    "evidence": recognition.evidence,
                },
            )
        except Exception as e:
            report = LayerReport(
                layer=LayerType.SEMANTIC,
                success=False,
                duration_ms=timer.duration_ms,
                message=f"Failed: {e}",
            )

    return image, report, result_dict


def extract(image: np.ndarray, settings: SemanticSettings) -> Optional[RecognitionResult]:
    """从图像中提取签名特征(用于验证)"""
    if settings.signature is None:
        return None
    return recognize_from_image(image, settings.signature)
