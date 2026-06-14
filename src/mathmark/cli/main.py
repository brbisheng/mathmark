"""MathMark CLI - 命令行界面

子命令:
    init       初始化教师身份
    sign       配置/编辑签名
    embed      嵌入水印
    verify     验证归属
    extract    提取水印bits
    benchmark  性能测试
    doctor     环境检查
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .. import __version__
from ..core.config import (
    create_default_config,
    load_config,
    load_signature,
    save_config,
    save_signature,
)
from ..core.pipeline import WatermarkPipeline
from ..core.types import (
    ContentType,
    LayerType,
    Verdict,
    WatermarkConfig,
)
from ..crypto.keys import (
    generate_keypair,
    load_public_key,
    save_keypair,
)
from ..semantic.example_db import hash_problem
from ..utils.image_io import is_image_file, load_image, load_image_rgb
from ..verify.extractor import extract_all, verify_image
from ..verify.report import generate_legal_report

app = typer.Typer(
    name="mathmark",
    help="MathMark - 多层抗AI去水印工具",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold cyan]mathmark[/bold cyan] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=_version_callback,
        is_eager=True,
        help="显示版本",
    ),
) -> None:
    """MathMark - 数学图文内容版权保护工具"""
    pass


# ============================================================
# init
# ============================================================

@app.command()
def init(
    name: str = typer.Option(..., "--name", "-n", help="教师姓名"),
    teacher_id: Optional[str] = typer.Option(None, "--id", help="教师 ID (默认: 随机)"),
    output_dir: Path = typer.Option(
        Path.home() / ".mathmark",
        "--output", "-o",
        help="配置输出目录",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config", help="配置文件路径"),
) -> None:
    """初始化教师身份和密钥"""
    output_dir.mkdir(parents=True, exist_ok=True)
    keys_dir = output_dir / "keys"
    keys_dir.mkdir(exist_ok=True)

    teacher_id = teacher_id or f"T{datetime.now().strftime('%Y%m%d%H%M%S')}"
    priv_path = keys_dir / "private.pem"
    pub_path = keys_dir / "public.pem"

    if not priv_path.exists():
        console.print(f"[yellow]生成新密钥对...[/yellow]")
        keypair = generate_keypair(algorithm="ec-p256")
        save_keypair(keypair, priv_path, pub_path)
        console.print(f"[green]✓[/green] 私钥: {priv_path}")
        console.print(f"[green]✓[/green] 公钥: {pub_path}")
    else:
        console.print(f"[blue]i[/blue] 密钥已存在, 跳过生成")

    cfg_path = config_path or (output_dir / "config.yaml")
    cfg = create_default_config(teacher_id=teacher_id, teacher_name=name)
    # 设置密钥路径
    cfg.teacher_public_key_path = pub_path
    cfg.teacher_private_key_path = priv_path
    cfg.metadata.copyright = f"© {name} {datetime.now().year}"
    cfg.metadata.contact = name
    save_config(cfg, cfg_path)
    console.print(f"[green]✓[/green] 配置: {cfg_path}")
    console.print()
    console.print("[bold green]初始化完成![/bold green]")
    console.print(f"教师: {name} (ID: {teacher_id})")
    console.print(f"下一步: [cyan]mathmark sign --problem 'x^2-5x+6=0'[/cyan]")


# ============================================================
# sign
# ============================================================

@app.command()
def sign(
    problem: Optional[str] = typer.Option(None, "--problem", "-p", help="招牌例题"),
    variant: Optional[List[str]] = typer.Option(None, "--variant", "-v", help="变体(可多次指定)"),
    marker: Optional[List[str]] = typer.Option(None, "--marker", "-m", help="结论标记"),
    config: Path = typer.Option(
        Path.home() / ".mathmark" / "config.yaml",
        "--config", "-c",
        help="配置文件路径",
    ),
    signature: Path = typer.Option(
        Path.home() / ".mathmark" / "signatures" / "default.json",
        "--signature", "-s",
        help="签名文件路径",
    ),
) -> None:
    """配置/编辑签名"""
    signature.parent.mkdir(parents=True, exist_ok=True)

    if signature.exists():
        sig = load_signature(signature)
    else:
        from ..core.types import MathSignature
        cfg_teacher_id = "unknown"
        if config.exists():
            try:
                cfg = load_config(config)
                cfg_teacher_id = cfg.teacher_id
            except Exception:
                pass
        sig = MathSignature(teacher_id=cfg_teacher_id)

    if marker:
        sig.symbol.conclusion_markers = list(set(sig.symbol.conclusion_markers + marker))
        console.print(f"[green]✓[/green] 已添加结论标记: {marker}")

    if problem:
        from ..core.types import SignatureProblem
        problem_hash = hash_problem(problem)
        sp = SignatureProblem(
            id=f"sig-{len(sig.example.signature_problems) + 1:03d}",
            problem=problem,
            variants=list(variant) if variant else [],
            hash=problem_hash,
        )
        sig.example.signature_problems.append(sp)
        console.print(f"[green]✓[/green] 已添加招牌例题: {problem} (hash: {problem_hash[:12]}...)")
        if variant:
            console.print(f"  [blue]i[/blue] 变体: {', '.join(variant)}")

    save_signature(sig, signature)
    console.print(f"[green]✓[/green] 签名已保存: {signature}")


# ============================================================
# embed
# ============================================================

@app.command()
def embed(
    input_path: Path = typer.Argument(..., help="输入图像或目录"),
    output_path: Path = typer.Argument(..., help="输出路径(文件或目录)"),
    config: Path = typer.Option(
        Path.home() / ".mathmark" / "config.yaml",
        "--config", "-c",
    ),
    layers: str = typer.Option(
        "all",
        "--layers", "-l",
        help="要启用的层(逗号分隔, 'all', 或 'L1,L2,...')",
    ),
    signature: Optional[Path] = typer.Option(None, "--signature", "-s", help="签名文件"),
    batch: bool = typer.Option(False, "--batch", "-b", help="批处理模式(输入是目录)"),
) -> None:
    """嵌入水印到图像"""
    if not config.exists():
        console.print(f"[red]✗[/red] 配置文件不存在: {config}")
        console.print(f"请先运行: [cyan]mathmark init --name '你的名字'[/cyan]")
        raise typer.Exit(1)

    cfg = load_config(config)

    # 解析 layers
    from ..core.config import _parse_layers
    if layers != "all":
        cfg.enabled_layers = _parse_layers(layers)
    if signature and signature.exists():
        from ..core.config import load_signature
        sig = load_signature(signature)
        cfg.semantic.signature = sig
        # 同时更新 L4 配置
        from ..core.types import SemanticSettings
        cfg.semantic = SemanticSettings(signature=sig, injection_strength=cfg.semantic.injection_strength)
    cfg.semantic.auto_suggest = True  # 默认开

    # 收集图像
    if batch or input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        if not output_path.is_dir():
            console.print(f"[red]✗[/red] 批处理模式下输出必须是目录")
            raise typer.Exit(1)
        image_files = sorted([p for p in input_path.glob("*") if is_image_file(p)])
    else:
        if not is_image_file(input_path):
            console.print(f"[red]✗[/red] 不支持的图像格式: {input_path}")
            raise typer.Exit(1)
        image_files = [input_path]
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_files:
        console.print(f"[red]✗[/red] 未找到图像文件")
        raise typer.Exit(1)

    console.print(f"[cyan]→[/cyan] 找到 {len(image_files)} 张图像")
    console.print(f"[cyan]→[/cyan] 启用的层: {', '.join(lt.value for lt in sorted(cfg.enabled_layers, key=lambda x: x.value))}")

    pipeline = WatermarkPipeline(cfg)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("处理中...", total=len(image_files))

        for img_path in image_files:
            try:
                image = load_image_rgb(img_path)
                out_file = output_path / img_path.name if (batch or input_path.is_dir()) else output_path
                manifest_file = out_file.with_suffix(".manifest.json")
                result = pipeline.process(
                    image,
                    output_path=out_file,
                    manifest_path=manifest_file,
                )
                progress.update(task, advance=1, description=f"已处理: {img_path.name}")
            except Exception as e:
                console.print(f"[red]✗[/red] {img_path.name}: {e}")

    console.print(f"\n[bold green]✓ 完成![/bold green] 输出: {output_path}")


# ============================================================
# verify
# ============================================================

@app.command()
def verify(
    image_path: Path = typer.Argument(..., help="待验证图像路径"),
    config: Path = typer.Option(
        Path.home() / ".mathmark" / "config.yaml",
        "--config", "-c",
    ),
    public_key: Optional[Path] = typer.Option(None, "--pubkey", help="公钥路径(用于 C2PA 验签)"),
    report: Optional[Path] = typer.Option(None, "--report", help="生成 PDF 报告"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="相似度阈值"),
) -> None:
    """验证水印/归属"""
    if not image_path.exists():
        console.print(f"[red]✗[/red] 文件不存在: {image_path}")
        raise typer.Exit(1)

    if not config.exists():
        console.print(f"[red]✗[/red] 配置文件不存在: {config}")
        raise typer.Exit(1)

    cfg = load_config(config)
    pub_key = None
    if public_key and public_key.exists():
        pub_key = load_public_key(public_key)
    elif cfg.teacher_public_key_path and cfg.teacher_public_key_path.exists():
        try:
            pub_key = load_public_key(cfg.teacher_public_key_path)
        except Exception:
            pass

    console.print(f"[cyan]→[/cyan] 验证: {image_path}")
    result = verify_image(image_path, cfg, public_key=pub_key, threshold=threshold)

    # 表格展示
    table = Table(title="验证结果", show_header=True, header_style="bold magenta")
    table.add_column("层", style="cyan")
    table.add_column("置信度", justify="right")
    table.add_column("匹配", justify="center")

    for layer, evidence in result.layer_evidence.items():
        match_str = "[green]✓[/green]" if evidence.matched else "[red]✗[/red]"
        table.add_row(
            layer.value,
            f"{evidence.confidence:.3f}",
            match_str,
        )

    table.add_section()
    table.add_row(
        "[bold]综合得分[/bold]",
        f"[bold]{result.confidence:.3f}[/bold]",
        f"[bold]{result.verdict.value}[/bold]",
    )
    console.print(table)

    if result.signer_info:
        console.print(f"\n[green]签名人[/green]: {result.signer_info.teacher_name or result.signer_info.teacher_id}")
        console.print(f"  公钥指纹: {result.signer_info.public_key_fingerprint[:32]}...")
        console.print(f"  签名时间: {result.signer_info.signing_time}")

    if report:
        try:
            generate_legal_report(result, report)
            console.print(f"\n[green]✓[/green] 报告已生成: {report}")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] 报告生成失败: {e}")


# ============================================================
# extract
# ============================================================

@app.command()
def extract(
    image_path: Path = typer.Argument(...),
    config: Path = typer.Option(
        Path.home() / ".mathmark" / "config.yaml", "--config", "-c",
    ),
) -> None:
    """提取所有水印 bits"""
    if not config.exists():
        console.print(f"[red]✗[/red] 配置文件不存在: {config}")
        raise typer.Exit(1)
    cfg = load_config(config)
    result = extract_all(image_path, cfg)

    table = Table(title="提取结果", show_header=True, header_style="bold magenta")
    table.add_column("层", style="cyan")
    table.add_column("Bits (hex)", overflow="fold")
    table.add_column("状态", justify="center")

    for layer, bits in result.items():
        if bits is None:
            table.add_row(layer.value, "[dim]N/A[/dim]", "[red]✗[/red]")
        else:
            table.add_row(layer.value, bits.hex()[:64], "[green]✓[/green]")

    console.print(table)


# ============================================================
# benchmark
# ============================================================

@app.command()
def benchmark(
    size: int = typer.Option(1024, "--size", "-s", help="图像尺寸"),
    iterations: int = typer.Option(3, "--iterations", "-n"),
    config: Path = typer.Option(
        Path.home() / ".mathmark" / "config.yaml", "--config", "-c",
    ),
) -> None:
    """性能基准测试"""
    import numpy as np

    if not config.exists():
        # 使用默认配置
        cfg = WatermarkConfig(teacher_id="bench")
    else:
        cfg = load_config(config)

    console.print(f"[cyan]→[/cyan] 生成测试图像 ({size}x{size})...")
    test_image = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)

    console.print(f"[cyan]→[/cyan] 运行 {iterations} 次...")
    pipeline = WatermarkPipeline(cfg)
    result = pipeline.benchmark(test_image, n_iterations=iterations)

    console.print()
    console.print(Panel(result.to_str(), title="基准测试结果", border_style="green"))


# ============================================================
# doctor
# ============================================================

@app.command()
def doctor() -> None:
    """环境检查"""
    from rich.panel import Panel
    from rich.table import Table

    table = Table(title="环境检查", show_header=True, header_style="bold magenta")
    table.add_column("组件", style="cyan")
    table.add_column("状态", justify="center")
    table.add_column("详情")

    # 核心依赖
    deps = [
        ("numpy", "基础数组"),
        ("PIL", "图像处理"),
        ("cv2", "OpenCV"),
        ("scipy", "科学计算"),
        ("piexif", "EXIF 读写"),
        ("imagehash", "感知哈希"),
        ("pywt", "PyWavelets"),
    ]
    optional_deps = [
        ("trustmark", "Adobe TrustMark"),
        ("onnxruntime", "ONNX 推理"),
        ("imwatermark", "DWT-DCT-SVD"),
        ("pytesseract", "Tesseract OCR"),
        ("paddleocr", "PaddleOCR"),
        ("c2pa", "C2PA 标准"),
        ("pptx", "PPT 处理"),
        ("reportlab", "PDF 报告"),
    ]

    for mod, desc in deps:
        try:
            __import__(mod)
            table.add_row(f"{mod}", "[green]✓[/green]", desc)
        except ImportError:
            table.add_row(f"{mod}", "[red]✗[/red]", f"{desc} (必需)")

    for mod, desc in optional_deps:
        try:
            __import__(mod)
            table.add_row(f"{mod}", "[green]✓[/green]", desc)
        except ImportError:
            table.add_row(f"{mod}", "[yellow]—[/yellow]", f"{desc} (可选)")

    console.print(table)

    # 配置检查
    cfg_path = Path.home() / ".mathmark" / "config.yaml"
    if cfg_path.exists():
        console.print(f"\n[green]✓[/green] 配置: {cfg_path}")
    else:
        console.print(f"\n[yellow]⚠[/yellow] 未找到配置: {cfg_path}")
        console.print(f"  运行 [cyan]mathmark init --name '你的名字'[/cyan] 初始化")


# ============================================================
# 辅助
# ============================================================

from datetime import datetime  # noqa: E402
from rich.panel import Panel  # noqa: E402


if __name__ == "__main__":
    app()
