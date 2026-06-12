"""L1 指纹层 - 感知哈希 (pHash + dHash)

功能:
- 计算图像的 pHash (perceptual hash) 和 dHash (difference hash)
- 用于快速筛查被修改的副本 (社交媒体再传播)
- 与数据库中的 pHash 比对, 找到相似副本
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import imagehash
import numpy as np
from PIL import Image

from ..core.types import LayerReport, LayerType
from ..utils.image_io import hamming_distance, hamming_similarity
from ..utils.perf import measure_time

PathLike = Union[str, Path]


@dataclass
class Fingerprint:
    """图像指纹"""
    phash: str
    dhash: str
    whash: Optional[str] = None
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return {
            "phash": self.phash,
            "dhash": self.dhash,
            "whash": self.whash,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fingerprint":
        return cls(
            phash=d["phash"],
            dhash=d["dhash"],
            whash=d.get("whash"),
            width=d.get("width", 0),
            height=d.get("height", 0),
        )


def compute_fingerprint(
    image: Union[Image.Image, np.ndarray, PathLike],
    include_whash: bool = True,
) -> Fingerprint:
    """计算图像的感知指纹

    Args:
        image: PIL Image / numpy array (RGB) / 文件路径
        include_whash: 是否计算 whash (wavelet-based, 更鲁棒)

    Returns:
        Fingerprint 包含 phash, dhash, 可选 whash
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8))
    else:
        img = image

    # 确保是 RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    phash = str(imagehash.phash(img, hash_size=16))
    dhash = str(imagehash.dhash(img, hash_size=16))
    whash = str(imagehash.whash(img, hash_size=16)) if include_whash else None

    return Fingerprint(
        phash=phash,
        dhash=dhash,
        whash=whash,
        width=img.width,
        height=img.height,
    )


def fingerprint_similarity(fp1: Fingerprint, fp2: Fingerprint) -> float:
    """计算两个指纹的综合相似度 (0~1)

    综合 pHash 和 dHash 的汉明距离。
    """
    sim_phash = hamming_similarity(fp1.phash, fp2.phash)
    sim_dhash = hamming_similarity(fp1.dhash, fp2.dhash)
    # 加权平均
    sim = 0.6 * sim_phash + 0.4 * sim_dhash

    if fp1.whash and fp2.whash:
        sim_whash = hamming_similarity(fp1.whash, fp2.whash)
        sim = 0.5 * sim + 0.5 * sim_whash

    return sim


@dataclass
class FingerprintDatabase:
    """指纹数据库 - 用于快速查找相似图像"""
    entries: dict[str, Fingerprint]  # key -> Fingerprint

    def add(self, key: str, image: Union[Image.Image, np.ndarray, PathLike]) -> None:
        self.entries[key] = compute_fingerprint(image)

    def add_fingerprint(self, key: str, fp: Fingerprint) -> None:
        self.entries[key] = fp

    def find_similar(
        self,
        query: Union[Fingerprint, Image.Image, np.ndarray, PathLike],
        threshold: float = 0.85,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """查找最相似的 top_k 个条目

        Returns:
            [(key, similarity), ...] 按相似度降序
        """
        if not isinstance(query, Fingerprint):
            query_fp = compute_fingerprint(query)
        else:
            query_fp = query

        results = []
        for key, fp in self.entries.items():
            sim = fingerprint_similarity(query_fp, fp)
            if sim >= threshold:
                results.append((key, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, path: PathLike) -> None:
        """持久化到 JSON"""
        import json
        Path(path).write_text(
            json.dumps({k: v.to_dict() for k, v in self.entries.items()}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: PathLike) -> "FingerprintDatabase":
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            entries={k: Fingerprint.from_dict(v) for k, v in data.items()}
        )


# ============================================================
# Layer 接口实现
# ============================================================

def process(
    image: np.ndarray,
    output_path: Optional[PathLike] = None,
) -> tuple[np.ndarray, LayerReport, Fingerprint]:
    """L1 指纹层的处理入口

    这一层是"零修改"层 - 不修改图像,只计算指纹。
    """
    with measure_time("L1_fingerprint") as timer:
        try:
            pil_img = Image.fromarray(image.astype(np.uint8))
            fp = compute_fingerprint(pil_img)

            report = LayerReport(
                layer=LayerType.FINGERPRINT,
                success=True,
                duration_ms=timer.duration_ms,
                message=f"phash={fp.phash[:16]}..., dhash={fp.dhash[:16]}...",
                metadata={
                    "phash": fp.phash,
                    "dhash": fp.dhash,
                    "whash": fp.whash or "",
                    "width": fp.width,
                    "height": fp.height,
                },
            )
        except Exception as e:
            report = LayerReport(
                layer=LayerType.FINGERPRINT,
                success=False,
                duration_ms=timer.duration_ms,
                message=f"Failed: {e}",
            )
            fp = Fingerprint(phash="", dhash="")

    return image, report, fp


def extract(image: np.ndarray) -> Optional[Fingerprint]:
    """从图像中提取指纹 (用于验证)"""
    try:
        return compute_fingerprint(image)
    except Exception:
        return None
