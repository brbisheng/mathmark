"""图像 I/O 工具

统一处理 numpy array 与 PIL Image 之间的转换,
保留 EXIF 信息,处理不同的图像格式。
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def load_image(
    path: PathLike,
    mode: str = "RGB",
    max_size: int | None = None,
) -> Image.Image:
    """加载图像为 PIL Image

    Args:
        path: 文件路径
        mode: 目标颜色模式 (RGB, RGBA, L)
        max_size: 最大边长,超过则等比例缩小
    """
    img = Image.open(path)

    # 保留 EXIF
    exif_data = img.info.get("exif", None)
    icc_profile = img.info.get("icc_profile", None)

    if mode and img.mode != mode:
        if mode == "RGBA" and img.mode == "RGB":
            # 转换 RGB->RGBA 时加不透明 alpha
            img = img.convert("RGBA")
        elif mode == "RGB" and img.mode == "RGBA":
            # RGBA->RGB 时用白底
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert(mode)

    if max_size and max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    # 还原 EXIF
    if exif_data and mode == "RGB":
        img.info["exif"] = exif_data
    if icc_profile:
        img.info["icc_profile"] = icc_profile

    return img


def load_image_rgb(path: PathLike, max_size: int | None = None) -> np.ndarray:
    """加载为 RGB numpy 数组 (H, W, 3), uint8"""
    img = load_image(path, mode="RGB", max_size=max_size)
    return np.array(img, dtype=np.uint8)


def save_image(
    img: Union[np.ndarray, Image.Image],
    path: PathLike,
    quality: int = 95,
    preserve_exif: bool = True,
    exif_bytes: Optional[bytes] = None,
    xmp_bytes: Optional[bytes] = None,
    **kwargs,
) -> None:
    """保存图像

    Args:
        img: numpy array 或 PIL Image
        path: 输出路径
        quality: JPEG/WebP 质量
        preserve_exif: 是否保留 EXIF
        exif_bytes: 显式提供的 EXIF bytes (优先于 img.info["exif"])
        xmp_bytes: 显式提供的 XMP packet bytes. JPEG: 写入 APP1 (XMP 段),
            PNG: 写入 iTXt chunk (keyword=XML:com.adobe.xmp). WebP/TIFF/BMP: 跳过.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            pil_img = Image.fromarray(img, mode="L")
        elif img.ndim == 3:
            if img.shape[2] == 3:
                pil_img = Image.fromarray(img, mode="RGB")
            elif img.shape[2] == 4:
                pil_img = Image.fromarray(img, mode="RGBA")
            else:
                raise ValueError(f"Unsupported channel count: {img.shape[2]}")
        else:
            raise ValueError(f"Unsupported array shape: {img.shape}")
    else:
        pil_img = img

    # 默认参数
    save_kwargs = {}
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif suffix == ".webp":
        save_kwargs["quality"] = quality
        save_kwargs["lossless"] = False
    elif suffix == ".png":
        save_kwargs["optimize"] = True
    save_kwargs.update(kwargs)

    # 把 EXIF 真正写进文件 (PNG 需要显式传 exif=, JPEG 会自动保留)
    if preserve_exif:
        eb = exif_bytes or pil_img.info.get("exif")
        if eb:
            save_kwargs["exif"] = eb

    # 写 XMP - audit B12: 之前 XMP 被截断后塞进 EXIF ImageDescription,
    # 不是真正的 XMP. 现在按容器分别写 APP1 (JPEG) / iTXt (PNG).
    if xmp_bytes:
        if suffix in (".jpg", ".jpeg"):
            # JPEG APP1 with XMP namespace signature
            pil_img.info["xmp"] = b"http://ns.adobe.com/xap/1.0/\x00" + xmp_bytes
        elif suffix == ".png":
            from PIL.PngImagePlugin import PngInfo
            pnginfo = save_kwargs.get("pnginfo")
            if pnginfo is None:
                pnginfo = PngInfo()
            pnginfo.add_text("XML:com.adobe.xmp", xmp_bytes.decode("utf-8"))
            save_kwargs["pnginfo"] = pnginfo

    pil_img.save(str(path), **save_kwargs)


def bytes_to_pil(data: bytes) -> Image.Image:
    """bytes 转 PIL Image"""
    return Image.open(BytesIO(data))


def pil_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """PIL Image 转 bytes"""
    buf = BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def hamming_distance(hash1: str, hash2: str) -> int:
    """计算两个 hash 字符串的汉明距离"""
    if len(hash1) != len(hash2):
        return max(len(hash1), len(hash2))
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def hamming_similarity(hash1: str, hash2: str) -> float:
    """计算两个 hash 字符串的相似度 (0~1)"""
    if not hash1 or not hash2:
        return 0.0
    dist = hamming_distance(hash1, hash2)
    return 1.0 - dist / len(hash1)


def is_image_file(path: PathLike) -> bool:
    """判断是否为支持的图像文件"""
    path = Path(path)
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def get_image_info(path: PathLike) -> dict:
    """获取图像元信息"""
    path = Path(path)
    img = Image.open(path)
    return {
        "path": str(path),
        "size": img.size,
        "mode": img.mode,
        "format": img.format,
        "file_size": path.stat().st_size,
        "has_exif": bool(img.info.get("exif")),
    }
