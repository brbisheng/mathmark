"""验证工作流示例 - 检测被偷的图"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from mathmark import (
    MathSignature,
    SignatureProblem,
    WatermarkConfig,
)
from mathmark.core.config import save_signature
from mathmark.verify.extractor import verify_image
from mathmark.verify.report import generate_legal_report


def main():
    """演示验证流程"""

    # 假设这是你的签名配置
    signature = MathSignature(
        teacher_id="T-VERIFY-001",
        teacher_name="数学张老师",
    )
    signature.symbol.conclusion_markers = ["∴", "故", "Q.E.D."]
    signature.symbol.variable_primary = ["x", "y", "z"]
    signature.step.introduction_phrases = ["设", "令", "记"]
    signature.example.signature_problems = [
        SignatureProblem(
            id="zhang-sig-001",
            problem="x^2 - 5x + 6 = 0",
            expected_factoring="(x-2)(x-3) = 0",
        ),
    ]

    # 配置
    config = WatermarkConfig(
        teacher_id=signature.teacher_id,
        teacher_name=signature.teacher_name,
    )
    config.semantic.signature = signature

    # 验证可疑图像
    suspicious_image = Path("path/to/suspicious.png")
    if not suspicious_image.exists():
        print(f"请先用 mathmark embed 处理一张图,然后改这个路径测试")
        return

    result = verify_image(suspicious_image, config)

    print(f"\n{'='*60}")
    print(f"验证结果")
    print(f"{'='*60}")
    print(f"图像: {result.image_path}")
    print(f"综合得分: {result.confidence:.3f}")
    print(f"判定: {result.verdict.value}")
    print(f"\n各层详情:")

    for layer, ev in result.layer_evidence.items():
        match_str = "✓ 匹配" if ev.matched else "✗ 不匹配"
        print(f"  {layer.value:30} {ev.confidence:6.3f}  {match_str}")
        if ev.details:
            for k, v in list(ev.details.items())[:2]:
                print(f"    - {k}: {str(v)[:80]}")

    # 生成证据
    if result.verdict.value in ("STRONG_MATCH", "PROBABLE_MATCH"):
        print(f"\n⚠️ 图像与您的签名高度匹配, 可能是您被盗用的内容")
        report_path = Path("evidence.pdf")
        generate_legal_report(result, report_path)
        print(f"✓ 法律证据报告: {report_path}")


if __name__ == "__main__":
    main()
