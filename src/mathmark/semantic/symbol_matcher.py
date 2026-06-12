"""符号习惯匹配器

匹配数学老师特定的符号风格:
- 结论标记 (∴ Q.E.D. 故)
- 变量命名偏好
- 集合/向量/极限等特殊符号写法
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np

from ..core.types import SymbolSignature
from .ocr import OCRResult


@dataclass
class SymbolMatch:
    """符号匹配结果"""
    matched_markers: List[str]
    matched_variables: List[str]
    matched_notations: List[str]
    missing_markers: List[str]
    missing_variables: List[str]
    confidence: float
    details: dict

    def to_dict(self) -> dict:
        return {
            "matched_markers": self.matched_markers,
            "matched_variables": self.matched_variables,
            "matched_notations": self.matched_notations,
            "missing_markers": self.missing_markers,
            "missing_variables": self.missing_variables,
            "confidence": self.confidence,
            "details": self.details,
        }


def _normalize_text(text: str) -> str:
    """文本标准化 - 处理 OCR 误差"""
    # 移除空白
    text = text.strip()
    # 标准化全角/半角
    text = text.replace(" ", "").replace("　", "")
    # 标准化结论符变体
    text = text.replace(":", "：")
    return text


def _variable_substitutions(text: str) -> Set[str]:
    """从文本中提取可能的变量名"""
    import re
    # 单字母变量
    variables = set()
    # 匹配 a-z, A-Z, α-ω, 希腊字母
    for match in re.finditer(r"[a-zA-Zα-ωΑ-Ω]", text):
        variables.add(match.group())
    return variables


def _markers_in_text(text: str, markers: List[str]) -> List[str]:
    """检测哪些 marker 在文本中出现"""
    found = []
    for marker in markers:
        if marker in text:
            found.append(marker)
    return found


def _notation_match(text: str, notation: str) -> bool:
    """检测特定 notation 是否在文本中"""
    if not notation:
        return True
    # 多种表示都尝试匹配
    candidates = [notation]
    # 简化 LaTeX
    if notation.startswith("\\"):
        candidates.append(notation[1:].replace("{", "").replace("}", ""))
    # Unicode 表示
    if notation == "\\in":
        candidates.extend(["∈", "∊", "∈"])
    elif notation == "\\vec":
        candidates.extend(["→", "⃗"])
    elif notation == "\\therefore":
        candidates.append("∴")
    elif notation == "\\because":
        candidates.append("∵")
    elif notation == "\\Rightarrow":
        candidates.extend(["⇒", "⟹"])
    return any(c in text for c in candidates)


def match_symbols(
    ocr_result: OCRResult,
    signature: SymbolSignature,
) -> SymbolMatch:
    """匹配符号签名

    Args:
        ocr_result: OCR 识别结果
        signature: 教师的符号签名配置

    Returns:
        SymbolMatch 包含匹配详情和置信度
    """
    full_text = _normalize_text(ocr_result.full_text)
    full_text_lower = full_text.lower()

    matched_markers = _markers_in_text(full_text, signature.conclusion_markers)
    text_variables = _variable_substitutions(full_text)
    matched_variables = [v for v in signature.variable_primary if v in text_variables]
    missing_variables = [v for v in signature.variable_primary if v not in text_variables]

    # 检查 notation
    matched_notations = []
    if signature.set_notation and _notation_match(full_text, signature.set_notation):
        matched_notations.append("set_notation")
    if signature.vector_notation and _notation_match(full_text, signature.vector_notation):
        matched_notations.append("vector_notation")

    # 计算置信度
    details = {}

    # Marker 得分: 教师有 N 个偏好 marker, 找到 K 个
    n_markers = len(signature.conclusion_markers) or 1
    marker_score = min(len(matched_markers) / n_markers, 1.0)
    details["marker_score"] = marker_score
    details["n_markers_matched"] = len(matched_markers)
    details["n_markers_total"] = n_markers

    # Variable 得分
    n_vars = len(signature.variable_primary) or 1
    var_score = min(len(matched_variables) / n_vars, 1.0)
    details["variable_score"] = var_score
    details["n_vars_matched"] = len(matched_variables)
    details["n_vars_total"] = n_vars

    # Notation 得分
    n_notations = 2  # set + vector
    notation_score = len(matched_notations) / n_notations
    details["notation_score"] = notation_score

    # 综合: marker 40%, variable 30%, notation 30%
    confidence = 0.4 * marker_score + 0.3 * var_score + 0.3 * notation_score
    # Bonus: 找到多个 markers 提高置信度
    if len(matched_markers) >= 2:
        confidence = min(confidence + 0.1, 1.0)

    return SymbolMatch(
        matched_markers=matched_markers,
        matched_variables=matched_variables,
        matched_notations=matched_notations,
        missing_markers=[m for m in signature.conclusion_markers if m not in matched_markers],
        missing_variables=missing_variables,
        confidence=confidence,
        details=details,
    )
