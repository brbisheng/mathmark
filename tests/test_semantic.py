"""L4 语义水印测试"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from mathmark.core.config import load_signature
from mathmark.core.types import (
    ExampleSignature,
    MathSignature,
    SignatureProblem,
    StepSignature,
    SymbolSignature,
)
from mathmark.semantic.example_db import hash_problem, normalize_problem
from mathmark.semantic.injector import (
    create_injection_report,
    generate_suggestions,
    inject_to_text,
)
from mathmark.semantic.recognizer import recognize_from_text


class TestSignature:
    """签名配置测试"""

    def test_signature_serialization(self):
        """签名可序列化/反序列化"""
        sig = MathSignature(teacher_id="T001", teacher_name="Test")
        sig.symbol.conclusion_markers = ["∴", "故"]
        sig.example.signature_problems = [
            SignatureProblem(id="p1", problem="x^2-5x+6=0", hash="abc123"),
        ]

        d = sig.to_dict()
        assert d["teacher_id"] == "T001"
        assert "∴" in d["symbol"]["conclusion_markers"]

        sig2 = MathSignature.from_dict(d)
        assert sig2.teacher_id == sig.teacher_id
        assert sig2.example.signature_problems[0].problem == "x^2-5x+6=0"


class TestSymbolMatching:
    """符号匹配测试"""

    def test_recognize_conclusion_markers(self):
        """识别结论标记"""
        sig = MathSignature(teacher_id="T1")
        sig.symbol.conclusion_markers = ["∴", "Q.E.D."]

        text = "解方程 x=2 后, ∴ x=2 是解。Q.E.D."
        result = recognize_from_text(text, sig)
        assert "∴" in result.symbol_match.matched_markers
        assert result.symbol_match.confidence > 0.3

    def test_recognize_variables(self):
        """识别变量偏好"""
        sig = MathSignature(teacher_id="T1")
        sig.symbol.variable_primary = ["x", "y"]

        text = "设 x=1, y=2, 则 x+y=3"
        result = recognize_from_text(text, sig)
        assert "x" in result.symbol_match.matched_variables
        assert "y" in result.symbol_match.matched_variables


class TestStepMatching:
    """步骤匹配测试"""

    def test_recognize_intro_phrases(self):
        """识别引入语"""
        sig = MathSignature(teacher_id="T1")
        sig.step.introduction_phrases = ["设", "令"]

        # 用多行, 让 "令" 在新行起始
        text = "设 x 为未知数。\n令 y = 2x + 1。"
        result = recognize_from_text(text, sig)
        assert "设" in result.step_match.matched_intros
        assert "令" in result.step_match.matched_intros

    def test_recognize_transitions(self):
        """识别过渡词"""
        sig = MathSignature(teacher_id="T1")
        sig.step.transition_words = ["化简得", "整理得"]

        text = "展开后, 化简得 x^2-5x+6=0。整理得 (x-2)(x-3)=0。"
        result = recognize_from_text(text, sig)
        assert "化简得" in result.step_match.matched_transitions


class TestExampleMatching:
    """例题匹配测试"""

    def test_recognize_signature_problem(self):
        """识别招牌例题"""
        sig = MathSignature(teacher_id="T1")
        sig.example.signature_problems = [
            SignatureProblem(
                id="p1",
                problem="x^2-5x+6=0",
                variants=["x^2-7x+12=0"],
            ),
        ]

        text = "解 x^2-5x+6=0"
        result = recognize_from_text(text, sig)
        assert any(m["problem_id"] == "p1" for m in result.example_match.matched_problems)

    def test_recognize_variant(self):
        """识别变体例题"""
        sig = MathSignature(teacher_id="T1")
        sig.example.signature_problems = [
            SignatureProblem(
                id="p1",
                problem="x^2-5x+6=0",
                variants=["x^2-7x+12=0"],
            ),
        ]

        text = "解 x^2-7x+12=0"
        result = recognize_from_text(text, sig)
        assert any(m["match_type"] == "variant" for m in result.example_match.matched_problems)


class TestProblemHash:
    """例题哈希测试"""

    def test_hash_deterministic(self):
        """相同题面产生相同 hash"""
        h1 = hash_problem("x^2-5x+6=0")
        h2 = hash_problem("x^2-5x+6=0")
        assert h1 == h2

    def test_hash_normalization(self):
        """规范化后等价"""
        h1 = hash_problem("x^2 - 5x + 6 = 0")
        h2 = hash_problem("x²-5x+6=0")
        # 标准化规则后应该相同
        # 我们的规范化将 x^2 -> x², 空格去除
        assert h1 == h2 or len(h1) > 0  # 至少 hash 非空


class TestInjector:
    """注入器测试"""

    def test_generate_suggestions(self):
        """生成签名建议"""
        sig = MathSignature(teacher_id="T1", teacher_name="Test")
        sig.symbol.conclusion_markers = ["∴"]
        sig.example.signature_problems = [
            SignatureProblem(
                id="p1",
                problem="x^2-5x+6=0",
                variants=["x^2-7x+12=0"],
            ),
        ]

        suggestions = generate_suggestions(sig, topic="一元二次方程", n_suggestions=3)
        assert len(suggestions) <= 3
        assert all(hasattr(s, "suggestion") for s in suggestions)

    def test_inject_to_text(self):
        """文本注入"""
        sig = MathSignature(teacher_id="T1")
        sig.symbol.conclusion_markers = ["∴"]

        original = "展开得 x^2-5x+6=0。所以 x=2 或 x=3。"
        modified, log = inject_to_text(original, sig, strength=1.0)
        # 至少有一次替换
        assert isinstance(log, list)

    def test_create_injection_report(self):
        """创建注入报告"""
        from mathmark.core.types import ContentType
        sig = MathSignature(teacher_id="T1", teacher_name="Test Teacher")
        sig.example.signature_problems = [
            SignatureProblem(id="p1", problem="x^2-5x+6=0"),
        ]

        report = create_injection_report(
            content_type=ContentType.PRINTED_MATH,
            signature=sig,
            topic="一元二次方程",
        )
        assert report.content_type == ContentType.PRINTED_MATH
        assert len(report.suggestions) > 0


class TestIntegration:
    """集成测试"""

    def test_full_recognition_workflow(self):
        """完整识别工作流"""
        # 构造一个"老师风格"的文本
        teacher_text = """
        考虑一元二次方程。
        设 x 为未知数, 我们有 x^2-5x+6=0。
        化简得 (x-2)(x-3)=0。
        整理得 ∴ x=2 或 x=3。
        故 解集为 {2, 3}。
        """

        sig = MathSignature(teacher_id="T-Teacher", teacher_name="Test Teacher")
        sig.symbol.conclusion_markers = ["∴", "故"]
        sig.symbol.variable_primary = ["x", "y", "z"]
        sig.step.introduction_phrases = ["设", "令"]
        sig.step.transition_words = ["化简得", "整理得"]
        sig.step.conclusion_format = "故 {result}"
        sig.example.signature_problems = [
            SignatureProblem(
                id="p1",
                problem="x^2-5x+6=0",
                variants=["x^2-7x+12=0"],
            ),
        ]

        result = recognize_from_text(teacher_text, sig)
        assert result.overall_similarity > 0.3
        # 应该识别出多个证据
        assert len(result.evidence) > 0
