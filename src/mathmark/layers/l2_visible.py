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
    position: str = "diagonal_scatter",
    opacity: float = 0.30,
    font_size_ratio: float = 0.04,
    color: Tuple[int, int, int] = (160, 160, 160),
    scatter_count_x: int = 3,
    scatter_count_y: int = 2,
    scatter_angle: float = -30.0,
) -> np.ndarray:
    """在图像上叠加半透明文字水印

    Args:
        image: 输入图像 (H, W, 3), uint8
        text: 水印文字
        position: 位置 ("diagonal_scatter", "tiled", "bottom-right", "bottom-left",
                           "top-right", "top-left", "center")
        opacity: 透明度 (0~1) — multiply 模式下用作 mask 灰度 (color × opacity)
        font_size_ratio: 字体大小相对图像宽度
        color: 文字颜色, 默认浅灰配合 multiply 混合
        scatter_count_x/y: diagonal_scatter 模式下的网格数
        scatter_angle: diagonal_scatter 模式下每个实例的倾斜角 (度)
    """
    h, w = image.shape[:2]
    font_size = max(int(w * font_size_ratio), 16)

    pil_img = Image.fromarray(image)
    font = _get_chinese_font(font_size)

    if position in ("tiled", "diagonal_scatter"):
        # 散布 / 平铺模式 - 用 multiply 混合保护正文
        # 灰度 mask: 文字像素 = color 灰度 × opacity (越透明越白), 背景 = 255
        # 与原图 multiply 后: 白底变浅灰 (水印可见), 黑字不变 (数学不被遮挡)
        gray = int(sum(color) // 3 * opacity)  # 灰度按 opacity 缩放
        gray = max(min(gray, 255), 0)
        mask = Image.new("L", pil_img.size, 255)
        mask_draw = ImageDraw.Draw(mask)
        if position == "diagonal_scatter":
            _draw_diagonal_scatter(
                mask_draw, text, font, w, h, gray,
                count_x=scatter_count_x, count_y=scatter_count_y, angle=scatter_angle,
            )
        else:  # tiled (back-compat)
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


def _draw_diagonal_scatter(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    w: int,
    h: int,
    gray: int,
    count_x: int = 3,
    count_y: int = 2,
    angle: float = -30.0,
) -> None:
    """对角散布水印 - 路透/法新风格 (折衷方案)

    在 count_x × count_y 网格上散布 count_x*count_y 个文字实例,
    奇偶行/列偏移半个 spacing 形成砖墙式覆盖。
    每个实例独立旋转 angle 度, 看起来像盖章而不是平铺。

    灰度 mask 模式: 文字像素 = gray, 背景 = 255.
    关键: 用 RGBA + alpha 旋转避免 BICUBIC 在 L 图上产生黑角。
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # 网格间距 (基于 count 而非密铺)
    spacing_x = w / (count_x + 0.5)
    spacing_y = h / (count_y + 0.5)

    # 用大画布容纳旋转后的文本 (旋转 expand=True 后, 边长按 cos+sin 比例放大)
    # 1.4 倍对 -30°~30° 都够用
    pad = int(max(tw, th) * 0.4)
    canvas_w = tw + pad * 2
    canvas_h = th + pad * 2

    # 拿到主 mask 图像
    main_mask = draw._image  # type: ignore[attr-defined]

    for row in range(count_y):
        for col in range(count_x):
            # 砖墙式偏移
            x_off = (spacing_x / 2) if (row % 2 == 1) else 0
            y_off = (spacing_y / 2) if (col % 2 == 1) else 0

            cx = int(spacing_x * (col + 0.5) + x_off)
            cy = int(spacing_y * (row + 0.5) + y_off)

            # RGBA 透明画布: 文字=不透明灰, 背景=全透明
            # 这样旋转时 alpha=0 的区域不会被插值成黑
            stamp = Image.new("RGBA", (canvas_w, canvas_h), (gray, gray, gray, 0))
            stamp_draw = ImageDraw.Draw(stamp)
            stamp_draw.text((pad, pad), text, font=font, fill=(gray, gray, gray, 255))

            # 旋转, expand=True 让画布放大到能容纳完整旋转内容
            stamp = stamp.rotate(angle, resample=Image.BICUBIC, expand=True)

            # 计算粘贴位置 (居中于 cx, cy)
            sw, sh = stamp.size
            px = cx - sw // 2
            py = cy - sh // 2

            # 完全在画布外 → 跳过
            if px + sw <= 0 or py + sh <= 0 or px >= w or py >= h:
                continue

            # 裁剪到主 mask 范围
            src_x0 = max(0, -px)
            src_y0 = max(0, -py)
            src_x1 = min(sw, w - px)
            src_y1 = min(sh, h - py)
            dst_x0 = max(0, px)
            dst_y0 = max(0, py)

            stamp_cropped = stamp.crop((src_x0, src_y0, src_x1, src_y1))
            # 提取 alpha 作为 mask, 保证透明区域不覆盖主 mask 的 255
            stamp_alpha = stamp_cropped.split()[3]
            stamp_l = stamp_cropped.convert("L")

            main_mask.paste(stamp_l, (dst_x0, dst_y0), mask=stamp_alpha)


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
