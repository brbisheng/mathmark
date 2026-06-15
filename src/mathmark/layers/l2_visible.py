"""L2 可见水印层

设计目标:
1. 抗 SOTA WOA-GAN/W-OA-CLIP 泛化攻击: 使用多样化水印样式
2. 抗 LaMa/MAT 修复攻击: 全图均匀扰动 (不是局部 logo)
3. 抗 GradCAM 局部模糊攻击: 扰动覆盖整个图像

实现策略:
- 半透明文字/Logo (用户可见的所有权宣示)
- 全图均匀高频扰动 (对抗性扰动, 抗局部攻击)
- 抗 SOTA 攻击的关键: 扰动与图像内容**深度融合**, 而非贴片式 logo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from ..core.types import LayerReport, LayerType, VisibleSettings
from ..utils.perf import measure_time

PathLike = Union[str, Path]


# ============================================================
# 抗攻击扰动
# ============================================================

def generate_perturbation(
    image: np.ndarray,
    secret_key: bytes,
    strength: float = 0.02,
) -> np.ndarray:
    """生成抗攻击的全图扰动

    Args:
        image: 输入图像 (H, W, C), uint8
        secret_key: 用于生成 PRNG 种子 (保证可重现)
        strength: 扰动强度 (0~1)

    Returns:
        扰动图 (H, W, C), uint8,与原图同形状
    """
    h, w = image.shape[:2]

    # 用密钥生成确定性 PRNG
    seed = int.from_bytes(secret_key[:4], "big") if len(secret_key) >= 4 else 42
    rng = np.random.default_rng(seed)

    # 1. 高频纹理扰动 - 抗局部修复
    #    使用 sin/cos 叠加产生不可分离的纹理
    x = np.linspace(0, 8 * np.pi, w, dtype=np.float32)
    y = np.linspace(0, 8 * np.pi, h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    # 多个不同频率的波形叠加
    texture = np.zeros((h, w), dtype=np.float32)
    for freq in [3.0, 7.0, 13.0, 23.0]:
        phase = rng.uniform(0, 2 * np.pi)
        texture += np.sin(freq * X + phase) * np.cos(freq * Y + phase * 0.7)
    texture = texture / (np.abs(texture).max() + 1e-8)  # 归一化到 [-1, 1]

    # 2. 扩展到 RGB 三通道 (每通道用不同偏移, 增加复杂度)
    perturbation = np.stack([
        texture,
        np.roll(texture, shift=h // 7, axis=0),
        np.roll(texture, shift=w // 11, axis=1),
    ], axis=-1)

    # 3. 调制扰动到视觉不可见但 AI 修复会破坏
    perturbation = perturbation * strength * 255.0

    return perturbation.astype(np.float32)


def apply_adversarial_perturbation(
    image: np.ndarray,
    secret_key: bytes,
    strength: float = 0.02,
) -> np.ndarray:
    """将对抗扰动应用到图像

    与原图相加, 然后 clip 到合法范围。
    """
    perturbation = generate_perturbation(image, secret_key, strength)
    result = image.astype(np.float32) + perturbation
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# 可见水印
# ============================================================

def _get_chinese_font(size: int) -> ImageFont.FreeTypeFont:
    """获取中文字体 - 跨平台查找"""
    candidates = [
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                return font
            except Exception:
                continue
    # fallback: 找一个能加载的字体
    fallbacks = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in fallbacks:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _font_supports_cjk(font) -> bool:
    """快速检查字体是否支持中文字符"""
    try:
        from PIL import Image, ImageDraw
        test_img = Image.new("1", (100, 100), 0)
        test_draw = ImageDraw.Draw(test_img)
        # '数' 编码位
        bbox = test_draw.textbbox((0, 0), "数", font=font)
        return (bbox[2] - bbox[0]) > 5  # CJK 字符通常宽 > 10
    except Exception:
        return False


def render_visible_watermark(
    image: np.ndarray,
    text: str,
    position: str = "bottom-right",
    opacity: float = 0.20,
    font_size_ratio: float = 0.03,
    color: Tuple[int, int, int] = (160, 160, 160),
) -> np.ndarray:
    """在图像上叠加半透明文字水印

    Args:
        image: 输入图像 (H, W, 3), uint8
        text: 水印文字
        position: 位置 ("bottom-right", "bottom-left", "top-right", "top-left", "center", "tiled")
        opacity: 透明度 (0~1)
        font_size_ratio: 字体大小相对图像宽度
        color: 文字颜色, 默认浅灰配合 multiply 混合
    """
    h, w = image.shape[:2]
    font_size = max(int(w * font_size_ratio), 16)

    pil_img = Image.fromarray(image)
    font = _get_chinese_font(font_size)

    if position == "tiled":
        # 全图平铺 - 抗裁切, 用 multiply 混合保护正文
        # 灰度 mask: 文字像素 = color 灰度, 背景 = 255 (白) → multiply 不影响白底
        # 正文是黑色 (≈0), multiply 后仍是 ≈0 → 水印在正文下不可见, 不盖字
        gray = sum(color) // 3
        mask = Image.new("L", pil_img.size, 255)
        mask_draw = ImageDraw.Draw(mask)
        _draw_tiled(mask_draw, text, font, w, h, gray)
        result = ImageChops.multiply(pil_img.convert("RGB"), mask.convert("RGB"))
    else:
        # 角落文字 - 保留 alpha 混合 (角落通常不在正文区)
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        _draw_corner(draw, text, font, w, h, position, opacity, color)
        result = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")

    return np.array(result, dtype=np.uint8)


def _draw_corner(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    w: int,
    h: int,
    position: str,
    opacity: float,
    color: Tuple[int, int, int],
) -> None:
    """在某个角落画水印"""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    margin = max(int(min(w, h) * 0.02), 10)

    positions = {
        "bottom-right": (w - tw - margin, h - th - margin),
        "bottom-left": (margin, h - th - margin),
        "top-right": (w - tw - margin, margin),
        "top-left": (margin, margin),
        "center": ((w - tw) // 2, (h - th) // 2),
    }
    x, y = positions.get(position, positions["bottom-right"])

    # 阴影: 与 color 互补色, 保证对比
    shadow_color = _shadow_color(color)
    shadow_offset = max(2, font.size // 20)
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(*shadow_color, int(opacity * 220)))
    # 主体
    draw.text((x, y), text, font=font, fill=(*color, int(opacity * 255)))


def _draw_tiled(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    w: int,
    h: int,
    gray: int,
) -> None:
    """平铺水印 - 抗裁切的关键

    在灰度 mask 上绘制: 文字像素 = gray, 背景 = 255.
    与原图 multiply 后: 白底不变 (255*gray/255=gray), 黑字区不变 (0*gray/255=0).
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # 平铺间距 - 拉大间距避免密铺盖住正文
    spacing_x = int(tw * 1.4)
    spacing_y = int(th * 2.2)

    for y in range(-spacing_y, h + spacing_y, spacing_y):
        for x in range(-spacing_x, w + spacing_x, spacing_x):
            # 奇数行偏移, 形成交错
            offset = (spacing_x // 2) if (y // spacing_y) % 2 == 1 else 0
            cx, cy = x + offset, y
            # 主体 (灰度值, 0=黑, 255=白; 与 multiply mask 配合)
            draw.text((cx, cy), text, font=font, fill=gray)


def _shadow_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """取 color 的对比色作为描边/阴影 — 浅色字用深色描边, 深色字用浅色描边"""
    avg = sum(color) / 3
    return (240, 240, 240) if avg < 128 else (16, 16, 16)


# ============================================================
# Layer 接口实现
# ============================================================

def process(
    image: np.ndarray,
    settings: VisibleSettings,
    secret_key: bytes = b"mathmark-default-key-2026",
    teacher_id: str = "",
    teacher_name: str = "",
    output_path: Optional[PathLike] = None,
) -> Tuple[np.ndarray, LayerReport]:
    """L2 可见水印处理

    1. 添加半透明文字水印 (宣示所有权)
    2. 应用对抗扰动 (抗 AI 修复攻击)
    """
    # 把 settings.text 里的占位符替换成实际 ID + 名字, 让截图就能溯源
    # 用 replace 不用 format: 防止 teacher_id 中包含 {__class__} 等格式注入
    resolved_text = settings.text
    resolved_text = resolved_text.replace("{teacher_id}", teacher_id or "TEACHER")
    resolved_text = resolved_text.replace("{teacher_name}", teacher_name or "")
    resolved_text = resolved_text.strip()

    with measure_time("L2_visible") as timer:
        try:
            result = image.copy()

            # 步骤1: 可见文字水印
            result = render_visible_watermark(
                result,
                text=resolved_text,
                position=settings.position,
                opacity=settings.opacity,
                font_size_ratio=settings.font_size_ratio,
                color=settings.color,
            )

            # 步骤2: 对抗扰动 (如果启用)
            perturbation_applied = False
            if settings.enable_perturbation and settings.perturbation_strength > 0:
                result = apply_adversarial_perturbation(
                    result,
                    secret_key=secret_key,
                    strength=settings.perturbation_strength,
                )
                perturbation_applied = True

            report = LayerReport(
                layer=LayerType.VISIBLE,
                success=True,
                duration_ms=timer.duration_ms,
                message=f"visible text='{resolved_text[:30]}', position={settings.position}, "
                        f"perturbation={'on' if perturbation_applied else 'off'}",
                metadata={
                    "text": settings.text,
                    "position": settings.position,
                    "opacity": settings.opacity,
                    "perturbation_applied": perturbation_applied,
                    "perturbation_strength": settings.perturbation_strength,
                },
            )
        except Exception as e:
            result = image
            report = LayerReport(
                layer=LayerType.VISIBLE,
                success=False,
                duration_ms=timer.duration_ms,
                message=f"Failed: {e}",
            )

    if output_path:
        from ..utils.image_io import save_image
        save_image(result, output_path)

    return result, report


def is_perturbation_present(
    image1: np.ndarray,
    image2: np.ndarray,
    threshold: float = 0.005,
) -> bool:
    """检测两张图像之间是否有 L2 对抗扰动

    用于验证。True 表示扰动仍然存在。
    """
    diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
    return diff.mean() > threshold * 255
