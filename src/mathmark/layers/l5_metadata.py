"""L5 元数据层 - EXIF / XMP / 自定义字段

写入:
- EXIF: Copyright, Artist, ImageDescription, DateTime
- XMP: dc:rights, dc:creator, xmp:CreatorTool
- 自定义: 老师 ID, 签名 hash, 处理时间戳

虽然社交媒体会剥离元数据, 但对于本地存储和部分平台仍然有效。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import piexif
from PIL import Image

from ..core.types import LayerReport, LayerType, MetadataSettings
from ..utils.perf import measure_time

PathLike = Union[str, Path]


# ============================================================
# EXIF 操作
# ============================================================

def _build_exif_dict(
    settings: MetadataSettings,
    teacher_id: str,
    signature_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """构造 EXIF dict"""
    exif_dict = {
        "0th": {},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }

    # 0th IFD - 图像基本信息
    if settings.copyright:
        exif_dict["0th"][piexif.ImageIFD.Copyright] = settings.copyright.encode("utf-8")
    if settings.contact:
        exif_dict["0th"][piexif.ImageIFD.Artist] = settings.contact.encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = f"MathMark-signed by {teacher_id}".encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.Software] = "MathMark v0.1.0".encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.DateTime] = datetime.now().strftime("%Y:%m:%d %H:%M:%S").encode("utf-8")

    # 自定义字段
    user_comment = json.dumps({
        "mathmark_teacher_id": teacher_id,
        "mathmark_signature_hash": signature_hash or "",
        "mathmark_timestamp": datetime.now().isoformat(),
        **settings.custom_fields,
    }, ensure_ascii=False)
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment.encode("utf-8")

    return exif_dict


def write_exif(
    image: Image.Image,
    settings: MetadataSettings,
    teacher_id: str,
    signature_hash: Optional[str] = None,
) -> Image.Image:
    """向 PIL Image 写入 EXIF"""
    exif_dict = _build_exif_dict(settings, teacher_id, signature_hash)
    exif_bytes = piexif.dump(exif_dict)
    image.info["exif"] = exif_bytes
    return image


def read_exif(image: Image.Image) -> Dict[str, Any]:
    """从 PIL Image 读取 EXIF"""
    exif_bytes = image.info.get("exif", b"")
    if not exif_bytes:
        return {}
    try:
        exif_dict = piexif.load(exif_bytes)
    except Exception:
        return {}

    result: Dict[str, Any] = {}

    # 0th IFD
    zeroth = exif_dict.get("0th", {})
    for tag in [piexif.ImageIFD.Copyright, piexif.ImageIFD.Artist, piexif.ImageIFD.ImageDescription]:
        if tag in zeroth:
            key = {piexif.ImageIFD.Copyright: "copyright",
                   piexif.ImageIFD.Artist: "artist",
                   piexif.ImageIFD.ImageDescription: "description"}[tag]
            value = zeroth[tag]
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            result[key] = value

    # Exif IFD
    exif_ifd = exif_dict.get("Exif", {})
    if piexif.ExifIFD.UserComment in exif_ifd:
        try:
            user_comment = exif_ifd[piexif.ExifIFD.UserComment]
            if isinstance(user_comment, bytes):
                # 跳过 prefix 字节
                if user_comment.startswith(b"ASCII\0\0\0"):
                    user_comment = user_comment[8:]
                elif user_comment.startswith(b"UNICODE\0"):
                    user_comment = user_comment[8:]
                user_comment = user_comment.decode("utf-8", errors="ignore")
            user_data = json.loads(user_comment)
            result["mathmark"] = user_data
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    return result


# ============================================================
# XMP 操作 (简化版 - 嵌入到 EXIF UserComment)
# ============================================================

def build_xmp_packet(
    settings: MetadataSettings,
    teacher_id: str,
    signature_hash: Optional[str] = None,
) -> str:
    """构造 XMP packet (作为字符串)"""
    now = datetime.now().isoformat()
    xmp = f"""<?xpacket begin='﻿' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:mathmark="http://mathmark.org/ns/1.0/">
    <dc:rights>
        <rdf:Alt>
            <rdf:li xml:lang="x-default">{settings.copyright or teacher_id}</rdf:li>
        </rdf:Alt>
    </dc:rights>
    <dc:creator>
        <rdf:Seq>
            <rdf:li>{settings.contact or teacher_id}</rdf:li>
        </rdf:Seq>
    </dc:creator>
    <xmp:CreatorTool>MathMark v0.1.0</xmp:CreatorTool>
    <xmp:CreateDate>{now}</xmp:CreateDate>
    <mathmark:TeacherId>{teacher_id}</mathmark:TeacherId>
    <mathmark:SignatureHash>{signature_hash or ''}</mathmark:SignatureHash>
</rdf:Description>
</rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""
    return xmp


# ============================================================
# Layer 接口
# ============================================================

def process(
    image: np.ndarray,
    settings: MetadataSettings,
    teacher_id: str,
    signature_hash: Optional[str] = None,
    output_path: Optional[PathLike] = None,
) -> Tuple[np.ndarray, LayerReport, dict]:
    """L5 元数据处理

    注意: 这一层不修改像素, 但需要保存图像才能持久化元数据。
    如果 output_path 为 None, 则返回原图(不持久化)。
    """
    with measure_time("L5_metadata") as timer:
        result: dict = {"exif": {}, "xmp": ""}
        try:
            pil_img = Image.fromarray(image.astype(np.uint8))
            if pil_img.mode == "RGBA":
                # EXIF 必须是 RGB
                bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                bg.paste(pil_img, mask=pil_img.split()[3])
                pil_img = bg
            elif pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # 写入 EXIF
            if settings.write_exif:
                pil_img = write_exif(pil_img, settings, teacher_id, signature_hash)
                result["exif_written"] = True
                result["exif"] = read_exif(pil_img)

            # 构造 XMP packet
            if settings.write_xmp:
                xmp = build_xmp_packet(settings, teacher_id, signature_hash)
                result["xmp"] = xmp
                result["xmp_written"] = True
                # XMP 嵌入到 EXIF 1st IFD
                try:
                    exif_bytes = pil_img.info.get("exif", piexif.dump({"0th": {}, "Exif": {}}))
                    exif_dict = piexif.load(exif_bytes)
                    # 注意: piexif 不直接支持 XMP, 这里把 XMP 嵌入 UserComment
                    # 实际 XMP 应作为 APP1 段写入 JPEG, 这里简化处理
                    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = (
                        f"MathMark XMP embedded. Raw XMP:\n{xmp[:200]}..."
                    ).encode("utf-8")
                    pil_img.info["exif"] = piexif.dump(exif_dict)
                except Exception as e:
                    result["xmp_warning"] = str(e)

            result_img = np.array(pil_img, dtype=np.uint8)

            report = LayerReport(
                layer=LayerType.METADATA,
                success=True,
                duration_ms=timer.duration_ms,
                message=f"EXIF: copyright={settings.copyright[:20]!r}, contact={settings.contact[:20]!r}",
                metadata={
                    "exif_written": result.get("exif_written", False),
                    "xmp_written": result.get("xmp_written", False),
                    "teacher_id": teacher_id,
                    "signature_hash": signature_hash or "",
                },
            )
        except Exception as e:
            result_img = image
            report = LayerReport(
                layer=LayerType.METADATA,
                success=False,
                duration_ms=timer.duration_ms,
                message=f"Failed: {e}",
            )

    if output_path:
        from ..utils.image_io import save_image
        save_image(result_img, output_path, quality=95)

    return result_img, report, result


def extract(image: Image.Image) -> Dict[str, Any]:
    """从图像中提取元数据(用于验证)"""
    return read_exif(image)
