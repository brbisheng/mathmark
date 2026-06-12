"""步骤结构匹配器

匹配数学老师特定的解题步骤结构:
- 引入语 ("设", "令", "记")
- 过渡词 ("化简得", "整理得")
- 结论格式 ("故 {result}", "Q.E.D.")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ..core.types import StepSignature
from .ocr import OCRResult


@dataclass
class StepMatch:
    matched_intros: List[str]
    matched_transitions: List[str]
    matched_conclusion_format: bool
    line_count: int
    has_structured_progression: bool
    confidence: float
    details: dict

    def to_dict(self) -> dict:
        return {
            "matched_intros": self.matched_intros,
            "matched_transitions": self.matched_transitions,
            "matched_conclusion_format": self.matched_conclusion_format,
            "line_count": self.line_count,
            "has_structured_progression": self.has_structured_progression,
            "confidence": self.confidence,
            "details": self.details,
        }


def _line_contains(line: str, phrases: List[str]) -> List[str]:
    """检查行中是否包含任何 phrase"""
    line_clean = line.strip()
    return [p for p in phrases if p in line_clean]


def _conclusion_format_match(full_text: str, fmt: str) -> bool:
    """检测结论格式是否匹配

    fmt 是模板, 如 "故 {result}", 提取关键模式
    """
    # 提取 {result} 之前的关键字
    m = re.match(r"^(.*?)\s*\{.*?\}", fmt)
    if not m:
        return False
    keyword = m.group(1).strip()
    if not keyword:
        return False
    # 检查文本中是否有以该关键字开头的句子
    return keyword in full_text


def match_steps(
    ocr_result: OCRResult,
    signature: StepSignature,
) -> StepMatch:
    """匹配步骤签名

    Args:
        ocr_result: OCR 识别结果
        signature: 教师的步骤签名配置

    Returns:
        StepMatch
    """
    lines = ocr_result.lines
    full_text = ocr_result.full_text

    matched_intros: List[str] = []
    matched_transitions: List[str] = []

    # 在每行中查找引入语
    for line in lines:
        for intro in signature.introduction_phrases:
            if line.strip().startswith(intro):
                if intro not in matched_intros:
                    matched_intros.append(intro)
                break

    # 在全文中查找过渡词
    for transition in signature.transition_words:
        if transition in full_text:
            matched_transitions.append(transition)

    has_conclusion = _conclusion_format_match(full_text, signature.conclusion_format)

    # 结构化进展检测: 引入语 -> 过渡 -> 结论 的顺序
    has_structured = False
    if matched_intros and (matched_transitions or has_conclusion):
        # 简单按行号顺序检查
        intro_line = -1
        conclusion_line = -1
        for i, line in enumerate(lines):
            if any(line.strip().startswith(p) for p in matched_intros) and intro_line == -1:
                intro_line = i
            if signature.conclusion_format and signature.conclusion_format.split("{")[0].strip() in line:
                conclusion_line = i
        if intro_line >= 0 and conclusion_line >= 0 and conclusion_line > intro_line:
            has_structured = True
        elif intro_line >= 0 and matched_transitions:
            has_structured = True

    # 置信度
    details = {
        "n_intros_matched": len(matched_intros),
        "n_intros_total": len(signature.introduction_phrases) or 1,
        "n_transitions_matched": len(matched_transitions),
        "n_transitions_total": len(signature.transition_words) or 1,
    }

    intro_score = details["n_intros_matched"] / details["n_intros_total"]
    transition_score = details["n_transitions_matched"] / details["n_transitions_total"]
    conclusion_score = 1.0 if has_conclusion else 0.0
    structure_score = 1.0 if has_structured else 0.0

    confidence = (
        0.35 * intro_score
        + 0.30 * transition_score
        + 0.20 * conclusion_score
        + 0.15 * structure_score
    )
    if len(matched_intros) >= 2:
        confidence = min(confidence + 0.05, 1.0)

    return StepMatch(
        matched_intros=matched_intros,
        matched_transitions=matched_transitions,
        matched_conclusion_format=has_conclusion,
        line_count=len(lines),
        has_structured_progression=has_structured,
        confidence=confidence,
        details=details,
    )
