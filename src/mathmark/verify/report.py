"""PDF 证据报告生成

依赖:
    pip install reportlab

如果未安装, 生成文本/Markdown 报告作为降级。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from ..core.types import LayerType, VerificationResult

PathLike = Union[str, Path]


def generate_text_report(result: VerificationResult) -> str:
    """生成纯文本报告(降级方案)"""
    lines = [
        "=" * 70,
        "MathMark 取证报告 / Forensic Report",
        "=" * 70,
        f"图像: {result.image_path}",
        f"时间: {result.timestamp.isoformat()}",
        f"综合得分: {result.confidence:.3f}",
        f"判定: {result.verdict.value}",
        "",
        "-" * 70,
        "各层证据",
        "-" * 70,
    ]

    for layer, evidence in result.layer_evidence.items():
        lines.append(f"\n[{layer.value}]")
        lines.append(f"  置信度: {evidence.confidence:.3f}")
        lines.append(f"  匹配: {evidence.matched}")
        if evidence.details:
            for k, v in evidence.details.items():
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)[:100]
                lines.append(f"  {k}: {v}")

    if result.signer_info:
        lines.extend([
            "",
            "-" * 70,
            "签名人信息",
            "-" * 70,
            f"  ID: {result.signer_info.teacher_id}",
            f"  姓名: {result.signer_info.teacher_name}",
            f"  公钥指纹: {result.signer_info.public_key_fingerprint}",
            f"  签名时间: {result.signer_info.signing_time.isoformat()}",
        ])

    lines.extend([
        "",
        "=" * 70,
        "法律声明",
        "=" * 70,
        "本报告由 MathMark 自动生成, 用于内容版权归属举证。",
        "包含多层 (L1-L6) 防御栈的综合验证结果。",
        "综合判定 STRONG_MATCH/PROBABLE_MATCH 表明图像与指定教师高度关联。",
    ])

    return "\n".join(lines)


def generate_markdown_report(result: VerificationResult) -> str:
    """生成 Markdown 报告"""
    md = [
        "# MathMark 取证报告\n",
        f"**图像**: `{result.image_path}`  ",
        f"**时间**: {result.timestamp.isoformat()}  ",
        f"**综合得分**: **{result.confidence:.3f}**  ",
        f"**判定**: **`{result.verdict.value}`**\n",
        "## 各层证据\n",
    ]

    md.append("| 层级 | 置信度 | 匹配 | 详情 |")
    md.append("|------|--------|------|------|")
    for layer, evidence in result.layer_evidence.items():
        details_str = ""
        if evidence.details:
            details_str = "; ".join(
                f"{k}={v}" if not isinstance(v, (dict, list)) else f"{k}=..."
                for k, v in list(evidence.details.items())[:2]
            )
        matched = "✅" if evidence.matched else "❌"
        md.append(f"| {layer.value} | {evidence.confidence:.3f} | {matched} | {details_str} |")

    if result.signer_info:
        md.extend([
            "\n## 签名人信息\n",
            f"- **ID**: {result.signer_info.teacher_id}",
            f"- **姓名**: {result.signer_info.teacher_name}",
            f"- **公钥指纹**: `{result.signer_info.public_key_fingerprint}`",
            f"- **签名时间**: {result.signer_info.signing_time.isoformat()}",
        ])

    md.extend([
        "\n## 法律声明\n",
        "本报告由 MathMark 自动生成, 用于内容版权归属举证。",
    ])

    return "\n".join(md)


def generate_pdf_report(result: VerificationResult, output_path: PathLike) -> None:
    """生成 PDF 报告(使用 reportlab)"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        # 降级到 markdown
        md = generate_markdown_report(result)
        Path(output_path).with_suffix(".md").write_text(md, encoding="utf-8")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 尝试注册中文字体
    chinese_font = None
    for font_path in [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("ChineseFont", font_path))
                chinese_font = "ChineseFont"
                break
            except Exception:
                pass

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    if chinese_font:
        styles.add(ParagraphStyle(name="ChineseTitle", fontName=chinese_font, fontSize=20, spaceAfter=12))
        styles.add(ParagraphStyle(name="ChineseBody", fontName=chinese_font, fontSize=10, leading=14))
    else:
        styles.add(ParagraphStyle(name="ChineseTitle", fontSize=20, spaceAfter=12))

    elements = []
    elements.append(Paragraph("MathMark 取证报告", styles["ChineseTitle"]))
    elements.append(Spacer(1, 0.5*cm))

    # 概要
    summary_data = [
        ["图像", str(result.image_path)],
        ["时间", result.timestamp.isoformat()],
        ["综合得分", f"{result.confidence:.3f}"],
        ["判定", result.verdict.value],
    ]
    t = Table(summary_data, colWidths=[3*cm, 14*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), chinese_font or "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5*cm))

    elements.append(Paragraph("各层证据", styles["ChineseBody"]))
    elements.append(Spacer(1, 0.3*cm))

    evidence_data = [["层级", "置信度", "匹配", "详情摘要"]]
    for layer, ev in result.layer_evidence.items():
        details_str = ""
        if ev.details:
            details_str = "; ".join(
                f"{k}={v}" if not isinstance(v, (dict, list)) else f"{k}=..."
                for k, v in list(ev.details.items())[:2]
            )
        evidence_data.append([
            layer.value,
            f"{ev.confidence:.3f}",
            "是" if ev.matched else "否",
            details_str[:80],
        ])

    t2 = Table(evidence_data, colWidths=[4*cm, 2*cm, 1.5*cm, 9*cm])
    t2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), chinese_font or "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(t2)

    if result.signer_info:
        elements.append(Spacer(1, 0.5*cm))
        elements.append(Paragraph("签名人信息", styles["ChineseBody"]))
        signer_data = [
            ["ID", result.signer_info.teacher_id],
            ["姓名", result.signer_info.teacher_name],
            ["公钥指纹", result.signer_info.public_key_fingerprint[:64] + "..."],
            ["签名时间", result.signer_info.signing_time.isoformat()],
        ]
        t3 = Table(signer_data, colWidths=[3*cm, 14*cm])
        t3.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), chinese_font or "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("BOX", (0, 0), (-1, -1), 1, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(t3)

    elements.append(Spacer(1, 1*cm))
    elements.append(Paragraph("法律声明", styles["ChineseBody"]))
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph(
        "本报告由 MathMark 自动生成, 用于内容版权归属举证。"
        "包含 L1-L6 防御栈的综合验证结果。"
        "STRONG_MATCH / PROBABLE_MATCH 判定表明图像与指定教师高度关联。",
        styles["ChineseBody"],
    ))

    doc.build(elements)


def generate_legal_report(result: VerificationResult, output_path: PathLike) -> None:
    """生成法律可用报告(优先 PDF, 失败则 Markdown)"""
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    if suffix == ".pdf":
        try:
            generate_pdf_report(result, output_path)
        except Exception as e:
            # 降级
            md = generate_markdown_report(result)
            output_path.with_suffix(".md").write_text(md, encoding="utf-8")
    elif suffix in (".md", ".markdown"):
        md = generate_markdown_report(result)
        output_path.write_text(md, encoding="utf-8")
    else:
        txt = generate_text_report(result)
        output_path.write_text(txt, encoding="utf-8")
