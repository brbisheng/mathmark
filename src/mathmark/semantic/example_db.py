"""招牌例题库 - 哈希索引

数学老师有独特的"招牌例题"。
例如 "x²-5x+6=0" 因式分解 = (x-2)(x-3) 是数学老师常用的标志性例题。

本模块:
1. 计算例题的 SHA-256 哈希(规范化)
2. 在 OCR 文本中快速查找
3. 计算例题匹配相似度
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Set

from ..core.types import ExampleSignature, SignatureProblem
from .ocr import OCRResult


# 规范化规则 - 去除 OCR 误差
_NORMALIZE_RULES = [
    # 全角 -> 半角
    (r"[！-～]", lambda m: chr(ord(m.group()) - 0xFEE0)),
    # x², x^2, x ² 统一
    (r"\^?\s*2", "²"),
    (r"\^?\s*3", "³"),
    (r"\^?\s*4", "⁴"),
    (r"\^\s*\{?2\}?", "²"),
    (r"\^\s*\{?3\}?", "³"),
    # 空格
    (r"\s+", ""),
    # 常见 OCR 错误
    (r"[Oo]", "0"),  # OCR O -> 0 (数字)
    (r"[Il]", "1"),  # OCR I/l -> 1 (在数字上下文中)
]


def normalize_problem(text: str) -> str:
    """规范化数学题面, 用于哈希计算"""
    s = text
    for pattern, replacement in _NORMALIZE_RULES:
        s = re.sub(pattern, replacement, s)
    return s


def hash_problem(problem: str) -> str:
    """计算规范化题面的 SHA-256"""
    norm = normalize_problem(problem)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def index_signature_problems(example_sig: ExampleSignature) -> None:
    """为所有例题计算并设置 hash 字段"""
    for prob in example_sig.all_problems():
        if prob.hash is None:
            prob.hash = hash_problem(prob.problem)
        for variant in prob.variants:
            # 也可以为变体计算 hash, 但只存到详情里
            pass


@dataclass
class ExampleMatch:
    matched_problems: List[dict]
    confidence: float
    details: dict

    def to_dict(self) -> dict:
        return {
            "matched_problems": self.matched_problems,
            "confidence": self.confidence,
            "details": self.details,
        }


def _problem_in_text(problem: str, full_text: str) -> bool:
    """检测规范化题面是否在文本中"""
    norm_problem = normalize_problem(problem)
    norm_text = normalize_problem(full_text)
    return norm_problem in norm_text


def _hash_in_text(target_hash: str, all_hashes: Set[str]) -> Optional[str]:
    """在哈希集合中查找"""
    return target_hash if target_hash in all_hashes else None


def match_examples(
    ocr_result: OCRResult,
    signature: ExampleSignature,
) -> ExampleMatch:
    """匹配招牌例题

    Args:
        ocr_result: OCR 识别结果
        signature: 教师的例题签名

    Returns:
        ExampleMatch
    """
    full_text = ocr_result.full_text
    all_problems = signature.all_problems()

    # 索引化
    index_signature_problems(signature)
    all_hashes = {p.hash for p in all_problems if p.hash}

    matched: List[dict] = []

    for prob in all_problems:
        # 1. 完整题面匹配
        if _problem_in_text(prob.problem, full_text):
            matched.append({
                "problem_id": prob.id,
                "match_type": "exact",
                "matched_text": prob.problem,
            })
            continue

        # 2. 变体匹配
        for variant in prob.variants:
            if _problem_in_text(variant, full_text):
                matched.append({
                    "problem_id": prob.id,
                    "match_type": "variant",
                    "matched_text": variant,
                })
                break
            # 尝试变体的因式分解形式
            if prob.expected_factoring and _problem_in_text(prob.expected_factoring, full_text):
                matched.append({
                    "problem_id": prob.id,
                    "match_type": "factoring",
                    "matched_text": prob.expected_factoring,
                })
                break

    # 计算置信度
    n_total = len(all_problems) or 1
    confidence = min(len(matched) / n_total, 1.0)
    # 一个 exact match 已经很强
    if any(m["match_type"] == "exact" for m in matched):
        confidence = max(confidence, 0.9)

    return ExampleMatch(
        matched_problems=matched,
        confidence=confidence,
        details={
            "n_problems_total": n_total,
            "n_matched": len(matched),
            "match_types": [m["match_type"] for m in matched],
        },
    )
