"""签名识别器 - 从图像/文本中识别数学老师签名

组合三个匹配器 (符号/步骤/例题) 加权打分
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from ..core.types import MathSignature
from .example_db import ExampleMatch, match_examples
from .ocr import OCRResult, get_engine
from .step_matcher import StepMatch, match_steps
from .symbol_matcher import SymbolMatch, match_symbols

PathLike = Union[str, Path]


# 加权: 数学老师的内容中, 符号 > 步骤 > 例题 (例题可能被改)
DEFAULT_WEIGHTS = {
    "symbol": 0.40,
    "step": 0.30,
    "example": 0.20,
    "visual": 0.10,
}


@dataclass
class RecognitionResult:
    """签名识别完整结果"""
    symbol_match: Optional[SymbolMatch] = None
    step_match: Optional[StepMatch] = None
    example_match: Optional[ExampleMatch] = None
    overall_similarity: float = 0.0
    verdict: str = "NO_MATCH"
    matched_signatures: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    ocr_engine: str = ""

    def to_dict(self) -> dict:
        return {
            "overall_similarity": self.overall_similarity,
            "verdict": self.verdict,
            "matched_signatures": self.matched_signatures,
            "evidence": self.evidence,
            "ocr_engine": self.ocr_engine,
            "symbol_match": self.symbol_match.to_dict() if self.symbol_match else None,
            "step_match": self.step_match.to_dict() if self.step_match else None,
            "example_match": self.example_match.to_dict() if self.example_match else None,
        }


def recognize_from_text(
    text: str,
    signature: MathSignature,
    weights: Optional[dict] = None,
) -> RecognitionResult:
    """从文本识别签名(直接给出 full_text)"""
    # 构造假 OCRResult
    ocr_result = OCRResult(
        tokens=[],
        full_text=text,
        lines=text.split("\n"),
        engine="direct",
    )
    return recognize_from_ocr(ocr_result, signature, weights)


def recognize_from_image(
    image: np.ndarray,
    signature: MathSignature,
    ocr_engine: str = "auto",
    weights: Optional[dict] = None,
) -> RecognitionResult:
    """从图像识别签名(自动 OCR)"""
    eng = get_engine(ocr_engine)
    ocr_result = eng.recognize(image)
    return recognize_from_ocr(ocr_result, signature, weights)


def recognize_from_ocr(
    ocr_result: OCRResult,
    signature: MathSignature,
    weights: Optional[dict] = None,
) -> RecognitionResult:
    """从 OCR 结果识别签名"""
    weights = weights or DEFAULT_WEIGHTS

    # 三个维度的匹配
    symbol_match = match_symbols(ocr_result, signature.symbol)
    step_match = match_steps(ocr_result, signature.step)
    example_match = match_examples(ocr_result, signature.example)

    # 加权综合
    overall = (
        weights["symbol"] * symbol_match.confidence
        + weights["step"] * step_match.confidence
        + weights["example"] * example_match.confidence
        + weights["visual"] * 0.0  # visual 暂不评分
    )

    # 证据收集
    evidence: List[str] = []
    if symbol_match.matched_markers:
        evidence.append(
            f"找到结论标记: {', '.join(symbol_match.matched_markers)}"
        )
    if symbol_match.matched_variables:
        evidence.append(
            f"找到偏好变量: {', '.join(symbol_match.matched_variables)}"
        )
    if step_match.matched_intros:
        evidence.append(
            f"找到引入语: {', '.join(step_match.matched_intros)}"
        )
    if step_match.matched_transitions:
        evidence.append(
            f"找到过渡词: {', '.join(step_match.matched_transitions)}"
        )
    if example_match.matched_problems:
        problem_ids = [m["problem_id"] for m in example_match.matched_problems]
        evidence.append(
            f"找到招牌例题: {', '.join(problem_ids)}"
        )

    # 判定
    if overall >= 0.75:
        verdict = "STRONG_MATCH"
    elif overall >= 0.55:
        verdict = "PROBABLE_MATCH"
    elif overall >= 0.35:
        verdict = "WEAK_MATCH"
    else:
        verdict = "NO_MATCH"

    return RecognitionResult(
        symbol_match=symbol_match,
        step_match=step_match,
        example_match=example_match,
        overall_similarity=overall,
        verdict=verdict,
        matched_signatures=[signature.teacher_id] if overall >= 0.55 else [],
        evidence=evidence,
        ocr_engine=ocr_result.engine,
    )


def recognize_multi(
    image: np.ndarray,
    signatures: List[MathSignature],
    ocr_engine: str = "auto",
) -> List[RecognitionResult]:
    """从一张图识别多个候选签名 - 找最匹配的"""
    eng = get_engine(ocr_engine)
    ocr_result = eng.recognize(image)
    return [
        recognize_from_ocr(ocr_result, sig)
        for sig in signatures
    ]
