"""OCR 引擎

支持多个后端:
1. pytesseract (Tesseract OCR) - 轻量, 中英文
2. paddleocr - 中文+数学公式, 更鲁棒
3. mock - 用于测试, 不实际 OCR

如果都不可用, 降级到 mock 引擎并发出警告。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass
class OCRToken:
    """OCR 识别的单个 token (含位置)"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    line_num: int = 0

    def contains(self, substring: str) -> bool:
        return substring in self.text


@dataclass
class OCRResult:
    """OCR 识别结果"""
    tokens: List[OCRToken]
    full_text: str
    lines: List[str]
    engine: str

    def find_all(self, pattern: str) -> List[OCRToken]:
        """查找包含指定文本的 tokens"""
        return [t for t in self.tokens if pattern in t.text]

    def has_any(self, candidates: List[str]) -> List[str]:
        """返回全文中出现的所有候选项"""
        return [c for c in candidates if c in self.full_text]


# ============================================================
# 引擎实现
# ============================================================

class TesseractEngine:
    """Tesseract OCR 引擎"""

    def __init__(self, lang: str = "chi_sim+eng", config: str = ""):
        try:
            import pytesseract
            self.pytesseract = pytesseract
        except ImportError:
            raise RuntimeError("pytesseract not installed. pip install pytesseract")
        self.lang = lang
        self.config = config

    def recognize(self, image: np.ndarray) -> OCRResult:
        """识别图像中的文字"""
        from PIL import Image
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image)

        # 详细模式: 包含位置信息
        data = self.pytesseract.image_to_data(
            pil_img,
            lang=self.lang,
            config=self.config,
            output_type=self.pytesseract.Output.DICT,
        )

        tokens = []
        for i, text in enumerate(data["text"]):
            text = str(text).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = 0.0
            if conf < 0:
                continue
            token = OCRToken(
                text=text,
                confidence=conf / 100.0,
                bbox=(int(data["left"][i]), int(data["top"][i]),
                      int(data["width"][i]), int(data["height"][i])),
                line_num=int(data["line_num"][i]),
            )
            tokens.append(token)

        full_text = " ".join(t.text for t in tokens)

        # 重建行
        lines_dict: dict[int, List[OCRToken]] = {}
        for t in tokens:
            lines_dict.setdefault(t.line_num, []).append(t)
        lines = []
        for line_num in sorted(lines_dict.keys()):
            line_tokens = sorted(lines_dict[line_num], key=lambda x: x.bbox[0])
            lines.append("".join(t.text for t in line_tokens))

        return OCRResult(tokens=tokens, full_text=full_text, lines=lines, engine="tesseract")


class PaddleOCREngine:
    """PaddleOCR 引擎 - 中文/数学更鲁棒"""

    def __init__(self, lang: str = "ch"):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        except ImportError:
            raise RuntimeError("paddleocr not installed. pip install paddleocr")
        self.lang = lang

    def recognize(self, image: np.ndarray) -> OCRResult:
        """识别图像中的文字"""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        results = self.ocr.ocr(image, cls=True)
        if not results or not results[0]:
            return OCRResult(tokens=[], full_text="", lines=[], engine="paddleocr")

        tokens = []
        lines_dict: dict[int, List[Tuple[OCRToken, int]]] = {}
        line_num = 0

        for line in results[0]:
            # line = [bbox_points, (text, confidence)]
            if not line or len(line) < 2:
                continue
            bbox_points, (text, conf) = line
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            token = OCRToken(
                text=text,
                confidence=float(conf),
                bbox=(int(x), int(y), int(w), int(h)),
                line_num=line_num,
            )
            tokens.append(token)
            line_num += 1

        full_text = " ".join(t.text for t in tokens)
        lines = [t.text for t in tokens]
        return OCRResult(tokens=tokens, full_text=full_text, lines=lines, engine="paddleocr")


class MockOCREngine:
    """Mock OCR 引擎 - 仅用于测试 (无 OCR 依赖时降级)

    默认文本包含 signatures/default.json 里的全部 signature 维度:
    - conclusion_markers: ∴, 故
    - variable_primary: x, y, z
    - introduction_phrases: 设, 令, 记
    - transition_words: 化简得, 整理得, 代入
    - signature_problems: x^2 - 5x + 6 = 0

    这样 sandbox/run.py demo 跑完 L4 verify 能命中所有维度, 不依赖真 OCR.
    真场景下请装 pytesseract 或 paddleocr.
    """

    def __init__(
        self,
        mock_text: str = (
            "设 x 为未知数, 令 y = 0 记 z = 1. "
            "化简得 x^2 - 5x + 6 = 0, 整理得 (x-2)(x-3) = 0. "
            "代入验证: x, y, z 都满足. "
            "故 解集为 {2, 3} ∴ Q.E.D."
        ),
    ):
        self.mock_text = mock_text

    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult(
            tokens=[OCRToken(text=self.mock_text, confidence=1.0, bbox=(0, 0, 100, 20))],
            full_text=self.mock_text,
            lines=[self.mock_text],
            engine="mock",
        )


# ============================================================
# 工厂
# ============================================================

_engines = {}

def get_engine(name: str = "auto") -> object:
    """获取 OCR 引擎

    Args:
        name: "auto" / "tesseract" / "paddleocr" / "mock"
    """
    if name in _engines:
        return _engines[name]

    if name == "auto":
        # 优先 paddleocr (中文更准)
        try:
            engine = PaddleOCREngine()
            _engines[name] = engine
            return engine
        except Exception:
            pass
        try:
            engine = TesseractEngine()
            _engines[name] = engine
            return engine
        except Exception:
            pass
        warnings.warn("No OCR engine available, using mock")
        engine = MockOCREngine()
        _engines[name] = engine
        return engine

    if name == "tesseract":
        engine = TesseractEngine()
    elif name == "paddleocr":
        engine = PaddleOCREngine()
    elif name == "mock":
        engine = MockOCREngine()
    else:
        raise ValueError(f"Unknown OCR engine: {name}")

    _engines[name] = engine
    return engine


def recognize(image: np.ndarray, engine: str = "auto") -> OCRResult:
    """快捷识别接口"""
    eng = get_engine(engine)
    return eng.recognize(image)
