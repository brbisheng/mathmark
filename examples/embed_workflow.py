"""MathMark 完整工作流示例

演示从初始化到加水印到验证的完整流程。
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np

from mathmark import (
    LayerType,
    MathSignature,
    SignatureProblem,
    WatermarkConfig,
    WatermarkPipeline,
)
from mathmark.core.config import load_signature, save_signature
from mathmark.crypto.keys import generate_keypair, save_keypair
from mathmark.utils.image_io import load_image, save_image
from mathmark.verify.extractor import verify_image


def step1_setup_teacher():
    """步骤 1: 设置教师身份和签名"""
    print("=" * 60)
    print("步骤 1: 设置教师身份")
    print("=" * 60)

    work_dir = Path("/tmp/mathmark_demo")
    work_dir.mkdir(exist_ok=True)
    keys_dir = work_dir / "keys"
    keys_dir.mkdir(exist_ok=True)

    # 生成密钥
    priv_path = keys_dir / "private.pem"
    pub_path = keys_dir / "public.pem"
    if not priv_path.exists():
        kp = generate_keypair(algorithm="ec-p256")
        save_keypair(kp, priv_path, pub_path)
        print(f"✓ 私钥: {priv_path}")
        print(f"✓ 公钥: {pub_path}")
    else:
        print(f"✓ 密钥已存在")

    # 创建签名
    signature = MathSignature(
        teacher_id="T-DEMO-001",
        teacher_name="数学王老师",
    )
    signature.symbol.conclusion_markers = ["∴", "Q.E.D.", "故"]
    signature.symbol.variable_primary = ["x", "y", "z"]
    signature.step.introduction_phrases = ["设", "令", "记"]
    signature.step.transition_words = ["化简得", "整理得", "代入"]
    signature.step.conclusion_format = "故 {result}"
    signature.example.signature_problems = [
        SignatureProblem(
            id="wang-sig-001",
            problem="x^2 - 5x + 6 = 0",
            expected_factoring="(x-2)(x-3) = 0",
            variants=[
                "x^2 - 7x + 12 = 0",
                "x^2 - 9x + 20 = 0",
            ],
        ),
    ]
    signature.example.personal_problems = [
        SignatureProblem(
            id="wang-personal-001",
            problem="∫_0^∞ e^(-x²) dx = √π/2",
        ),
    ]

    sig_path = work_dir / "signatures" / "wang.json"
    sig_path.parent.mkdir(exist_ok=True)
    save_signature(signature, sig_path)
    print(f"✓ 签名配置: {sig_path}")

    return work_dir, signature, priv_path, pub_path


def step2_create_test_image(work_dir: Path) -> Path:
    """步骤 2: 创建一个模拟数学题目图"""
    print("\n" + "=" * 60)
    print("步骤 2: 创建测试图像")
    print("=" * 60)

    # 创建一个简单的数学题目图
    img = Image.new("RGB", (1024, 768), "white")
    draw = ImageDraw_safe(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    draw.text((50, 50), "Math Lesson: Quadratic Equations", fill="black", font=font)
    draw.text((50, 120), "Problem: x^2 - 5x + 6 = 0", fill="black", font=font)
    draw.text((50, 180), "Step 1: 设 x 为未知数, 我们有", fill="black", font=font_small)
    draw.text((50, 220), "        x^2 - 5x + 6 = 0", fill="black", font=font_small)
    draw.text((50, 280), "Step 2: 化简得 (x-2)(x-3) = 0", fill="black", font=font_small)
    draw.text((50, 320), "Step 3: 整理得 x = 2 或 x = 3", fill="black", font=font_small)
    draw.text((50, 380), "故 解集为 {2, 3}", fill="black", font=font)
    draw.text((50, 450), "Q.E.D.", fill="black", font=font)

    img_path = work_dir / "original.png"
    img.save(str(img_path))
    print(f"✓ 创建测试图: {img_path}")
    return img_path


class ImageDraw_safe:
    """简单的 ImageDraw 包装"""
    def __init__(self, img):
        from PIL import ImageDraw
        self._draw = ImageDraw.Draw(img)

    def text(self, xy, text, **kwargs):
        self._draw.text(xy, text, **kwargs)


def step3_embed(work_dir: Path, signature: MathSignature, priv_path: Path, pub_path: Path):
    """步骤 3: 加水印"""
    print("\n" + "=" * 60)
    print("步骤 3: 加水印")
    print("=" * 60)

    # 配置
    config = WatermarkConfig(
        teacher_id=signature.teacher_id,
        teacher_name=signature.teacher_name,
        teacher_private_key_path=priv_path,
        teacher_public_key_path=pub_path,
    )
    config.semantic.signature = signature
    config.metadata.copyright = f"© {signature.teacher_name} 2026"
    config.metadata.contact = signature.teacher_name

    # 加载图
    img_path = work_dir / "original.png"
    pil_img = load_image(img_path, mode="RGB")
    image = np.array(pil_img, dtype=np.uint8)

    # 处理
    pipeline = WatermarkPipeline(config)
    output_path = work_dir / "watermarked.png"
    manifest_path = work_dir / "watermarked.png.manifest.json"
    result = pipeline.process(
        image,
        output_path=output_path,
        manifest_path=manifest_path,
    )

    print(f"✓ 加水印耗时: {result.total_duration_ms:.0f}ms")
    for layer, report in result.layer_reports.items():
        status = "✓" if report.success else "✗"
        print(f"  {status} {layer.value}: {report.message[:60]}")
    print(f"✓ 输出: {output_path}")
    print(f"✓ Manifest: {manifest_path}")

    return output_path


def step4_simulate_attack(watermarked_path: Path):
    """步骤 4: 模拟被偷 - 社媒压缩"""
    print("\n" + "=" * 60)
    print("步骤 4: 模拟社媒压缩 (微信)")
    print("=" * 60)

    from mathmark.attacks import wechat_compress

    pil = load_image(watermarked_path, mode="RGB")
    image = np.array(pil, dtype=np.uint8)
    attacked = wechat_compress(image, quality=75)

    attacked_path = watermarked_path.parent / "after_wechat.png"
    save_image(attacked, attacked_path)
    print(f"✓ 模拟微信压缩后: {attacked_path}")
    return attacked_path


def step5_verify(work_dir: Path, signature: MathSignature, priv_path: Path, pub_path: Path):
    """步骤 5: 验证归属"""
    print("\n" + "=" * 60)
    print("步骤 5: 验证水印")
    print("=" * 60)

    # 配置
    config = WatermarkConfig(
        teacher_id=signature.teacher_id,
        teacher_name=signature.teacher_name,
        teacher_private_key_path=priv_path,
        teacher_public_key_path=pub_path,
    )
    config.semantic.signature = signature
    config.metadata.copyright = f"© {signature.teacher_name} 2026"

    # 验证原始带水印图
    print("\n--- 验证 1: 原始带水印图 ---")
    r1 = verify_image(work_dir / "watermarked.png", config)
    print(f"综合得分: {r1.confidence:.3f}")
    print(f"判定: {r1.verdict.value}")
    for layer, ev in r1.layer_evidence.items():
        print(f"  {layer.value}: {ev.confidence:.3f} (matched={ev.matched})")

    # 验证攻击后的图
    print("\n--- 验证 2: 微信压缩后 ---")
    r2 = verify_image(work_dir / "after_wechat.png", config)
    print(f"综合得分: {r2.confidence:.3f}")
    print(f"判定: {r2.verdict.value}")
    for layer, ev in r2.layer_evidence.items():
        print(f"  {layer.value}: {ev.confidence:.3f} (matched={ev.matched})")

    # 生成证据报告
    from mathmark.verify.report import generate_legal_report

    report_path = work_dir / "evidence.pdf"
    try:
        generate_legal_report(r1, report_path)
        print(f"\n✓ 证据报告: {report_path}")
    except Exception as e:
        print(f"\n报告生成失败: {e}")


def main():
    """主工作流"""
    work_dir, signature, priv_path, pub_path = step1_setup_teacher()
    step2_create_test_image(work_dir)
    watermarked = step3_embed(work_dir, signature, priv_path, pub_path)
    step4_simulate_attack(watermarked)
    step5_verify(work_dir, signature, priv_path, pub_path)

    print("\n" + "=" * 60)
    print("✓ 完整工作流完成!")
    print(f"所有文件: {work_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
