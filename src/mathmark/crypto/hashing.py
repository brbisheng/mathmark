"""哈希工具"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_bytes_raw(data: bytes) -> bytes:
    """返回原始 32 字节 hash"""
    return hashlib.sha256(data).digest()


def sha256_file(path: PathLike, chunk_size: int = 65536) -> str:
    """计算文件 SHA-256"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_image(image: np.ndarray) -> str:
    """计算图像数组的 SHA-256"""
    return sha256_bytes(image.tobytes())


def hmac_sha256(key: bytes, data: bytes) -> str:
    """HMAC-SHA256"""
    return hmac.new(key, data, hashlib.sha256).hexdigest()
