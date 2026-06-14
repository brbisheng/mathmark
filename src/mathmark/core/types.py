"""MathMark 核心类型定义

集中定义所有跨模块使用的数据类，便于类型检查和重构。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ============================================================
# 枚举
# ============================================================

class LayerType(str, enum.Enum):
    """6 层防御栈的层级"""
    FINGERPRINT = "L1_fingerprint"  # pHash
    VISIBLE = "L2_visible"          # 可见水印
    INVISIBLE_TRUSTMARK = "L3a_trustmark"     # TrustMark
    INVISIBLE_DWT = "L3b_dwt_dct_svd"          # DWT-DCT-SVD
    INVISIBLE_COX = "L3c_cox_spread"           # Cox 扩频
    SEMANTIC = "L4_semantic"        # 数学语义水印 (核心)
    METADATA = "L5_metadata"        # EXIF/XMP
    C2PA = "L6_c2pa"                # C2PA manifest


class Verdict(str, enum.Enum):
    """验证结论"""
    STRONG_MATCH = "STRONG_MATCH"     # 强归属 (>0.85)
    PROBABLE_MATCH = "PROBABLE_MATCH" # 可能归属 (>0.65)
    WEAK_INDICATION = "WEAK"          # 弱关联 (>0.4)
    NO_MATCH = "NO_MATCH"             # 不匹配


class ContentType(str, enum.Enum):
    """内容类型 - 决定 L4 注入策略"""
    PPT_EXPORT = "ppt_export"        # PPT 导出
    PRINTED_MATH = "printed_math"    # 印刷体数学题
    HANDWRITTEN = "handwritten"      # 手写板书
    MIXED = "mixed"                  # 混合


# ============================================================
# 签名配置 (L4)
# ============================================================

@dataclass
class SymbolSignature:
    """A. 符号习惯"""
    conclusion_markers: list[str] = field(default_factory=lambda: ["∴", "故"])
    variable_primary: list[str] = field(default_factory=lambda: ["x", "y", "z"])
    set_notation: str = "∈"
    vector_notation: str = "\\vec{AB}"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepSignature:
    """B. 步骤结构"""
    introduction_phrases: list[str] = field(default_factory=lambda: ["设", "令", "记"])
    transition_words: list[str] = field(default_factory=lambda: ["化简得", "整理得", "代入"])
    conclusion_format: str = "故 {result}"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignatureProblem:
    """招牌例题 - 数学老师的标志性题目"""
    id: str
    problem: str                              # 题面
    expected_factoring: Optional[str] = None  # 预期因式分解
    variants: list[str] = field(default_factory=list)
    recurrence_pattern: Optional[str] = None
    hash: Optional[str] = None                # 自动计算的 SHA-256


@dataclass
class ExampleSignature:
    """C. 招牌例题库"""
    signature_problems: list[SignatureProblem] = field(default_factory=list)
    personal_problems: list[SignatureProblem] = field(default_factory=list)

    def all_problems(self) -> list[SignatureProblem]:
        return self.signature_problems + self.personal_problems


@dataclass
class VisualSignature:
    """D. 视觉记号"""
    arrow_style: str = "⟹"
    underline_style: str = "wavy"  # wavy | straight | dashed
    emphasis_style: str = "red-box"
    color_scheme: list[str] = field(default_factory=lambda: ["#FF6B6B", "#4ECDC4"])


@dataclass
class MathSignature:
    """数学老师的完整签名 - L4 层核心"""
    teacher_id: str
    teacher_name: str = ""
    symbol: SymbolSignature = field(default_factory=SymbolSignature)
    step: StepSignature = field(default_factory=StepSignature)
    example: ExampleSignature = field(default_factory=ExampleSignature)
    visual: VisualSignature = field(default_factory=VisualSignature)

    def to_dict(self) -> dict[str, Any]:
        """转字典(用于 JSON 序列化)"""
        return {
            "teacher_id": self.teacher_id,
            "teacher_name": self.teacher_name,
            "symbol": self.symbol.__dict__,
            "step": self.step.__dict__,
            "example": {
                "signature_problems": [p.__dict__ for p in self.example.signature_problems],
                "personal_problems": [p.__dict__ for p in self.example.personal_problems],
            },
            "visual": self.visual.__dict__,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MathSignature:
        """从字典构造"""
        example_d = d.get("example", {})
        sig_problems = [SignatureProblem(**p) for p in example_d.get("signature_problems", [])]
        pers_problems = [SignatureProblem(**p) for p in example_d.get("personal_problems", [])]
        return cls(
            teacher_id=d["teacher_id"],
            teacher_name=d.get("teacher_name", ""),
            symbol=SymbolSignature(**d.get("symbol", {})),
            step=StepSignature(**d.get("step", {})),
            example=ExampleSignature(
                signature_problems=sig_problems,
                personal_problems=pers_problems,
            ),
            visual=VisualSignature(**d.get("visual", {})),
        )


# ============================================================
# 各层配置
# ============================================================

@dataclass
class VisibleSettings:
    """L2 可见水印配置"""
    # 默认带机器可读 ID 段, 截图 OCR 出来能直接溯源
    text: str = "© {teacher_id} {teacher_name}"
    position: str = "tiled"  # tiled | bottom-right | center
    opacity: float = 0.18
    font_size_ratio: float = 0.04   # 相对图像宽度的字体大小
    color: tuple[int, int, int] = (255, 255, 255)
    perturbation_strength: float = 0.02  # 对抗扰动强度
    enable_perturbation: bool = True


@dataclass
class TrustMarkSettings:
    """L3a TrustMark 配置"""
    model_path: Optional[Path] = None     # ONNX 模型路径
    secret_bits: Optional[bytes] = None   # 100-bit 密钥
    encoder_model: str = "trustmark"
    use_quantized: bool = True            # int8 量化


@dataclass
class DWTSettings:
    """L3b DWT-DCT-SVD 配置"""
    alpha: float = 0.2
    block_size: int = 4


@dataclass
class CoxSettings:
    """L3c Cox 扩频配置"""
    strength: float = 0.05
    seed: int = 42


@dataclass
class SemanticSettings:
    """L4 语义水印配置"""
    signature: Optional[MathSignature] = None
    injection_strength: float = 0.5       # 注入强度 0~1
    ocr_engines: list[str] = field(default_factory=lambda: ["tesseract", "paddleocr"])
    auto_suggest: bool = True             # 自动建议签名变体


@dataclass
class MetadataSettings:
    """L5 元数据配置"""
    write_exif: bool = True
    write_xmp: bool = True
    copyright: str = ""
    contact: str = ""
    custom_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class C2PASettings:
    """L6 C2PA 配置"""
    enable: bool = True
    private_key_path: Optional[Path] = None
    certificate_path: Optional[Path] = None
    algorithm: str = "ES256"  # ES256 | EdDSA
    tsa_url: Optional[str] = None  # 时间戳机构


# ============================================================
# 顶层配置
# ============================================================

@dataclass
class WatermarkConfig:
    """完整的水印配置"""
    teacher_id: str
    teacher_name: str = ""
    teacher_public_key_path: Optional[Path] = None
    teacher_private_key_path: Optional[Path] = None

    enabled_layers: set[LayerType] = field(default_factory=lambda: {
        LayerType.FINGERPRINT,
        LayerType.VISIBLE,
        LayerType.INVISIBLE_TRUSTMARK,
        LayerType.INVISIBLE_DWT,
        LayerType.INVISIBLE_COX,
        LayerType.SEMANTIC,
        LayerType.METADATA,
        LayerType.C2PA,
    })

    visible: VisibleSettings = field(default_factory=VisibleSettings)
    trustmark: TrustMarkSettings = field(default_factory=TrustMarkSettings)
    dwt: DWTSettings = field(default_factory=DWTSettings)
    cox: CoxSettings = field(default_factory=CoxSettings)
    semantic: SemanticSettings = field(default_factory=SemanticSettings)
    metadata: MetadataSettings = field(default_factory=MetadataSettings)
    c2pa: C2PASettings = field(default_factory=C2PASettings)

    # 验证配置
    similarity_threshold: float = 0.75
    legal_export: bool = True

    def is_enabled(self, layer: LayerType) -> bool:
        return layer in self.enabled_layers


# ============================================================
# 处理结果
# ============================================================

@dataclass
class LayerReport:
    """单层处理报告"""
    layer: LayerType
    success: bool
    duration_ms: float
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WatermarkResult:
    """加水印处理的完整结果"""
    image: np.ndarray
    teacher_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # 各层结果
    layer_reports: dict[LayerType, LayerReport] = field(default_factory=dict)
    extracted_bits: dict[LayerType, bytes] = field(default_factory=dict)

    # 元数据
    phash: Optional[str] = None
    semantic_injection_log: list[dict[str, Any]] = field(default_factory=list)
    c2pa_manifest_path: Optional[Path] = None
    exif_bytes: Optional[bytes] = None

    # 性能
    total_duration_ms: float = 0.0

    def save(self, output_path: Path) -> None:
        """保存结果到 JSON"""
        import json
        data = {
            "teacher_id": self.teacher_id,
            "timestamp": self.timestamp.isoformat(),
            "phash": self.phash,
            "total_duration_ms": self.total_duration_ms,
            "layer_reports": {
                k.value: {
                    "success": v.success,
                    "duration_ms": v.duration_ms,
                    "message": v.message,
                    "metadata": v.metadata,
                }
                for k, v in self.layer_reports.items()
            },
            "semantic_injection_log": self.semantic_injection_log,
            "c2pa_manifest_path": str(self.c2pa_manifest_path) if self.c2pa_manifest_path else None,
        }
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ============================================================
# 验证结果
# ============================================================

@dataclass
class Evidence:
    """单层取证证据"""
    layer: LayerType
    confidence: float         # 0~1
    details: dict[str, Any] = field(default_factory=dict)
    matched: bool = False


@dataclass
class SignerInfo:
    """签名人信息"""
    teacher_id: str
    teacher_name: str
    public_key_fingerprint: str
    signing_time: datetime


@dataclass
class VerificationResult:
    """验证/取证的完整结果"""
    image_path: Path
    verdict: Verdict
    confidence: float          # 0~1 综合得分
    layer_evidence: dict[LayerType, Evidence] = field(default_factory=dict)
    signer_info: Optional[SignerInfo] = None
    c2pa_manifest_path: Optional[Path] = None
    semantic_similarity: float = 0.0
    bit_consensus: Optional[bytes] = None
    phash_match: Optional[dict[str, Any]] = None
    chain_of_custody: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "semantic_similarity": self.semantic_similarity,
            "layer_evidence": {
                k.value: {
                    "confidence": v.confidence,
                    "matched": v.matched,
                    "details": v.details,
                }
                for k, v in self.layer_evidence.items()
            },
            "signer_info": self.signer_info.__dict__ if self.signer_info else None,
            "c2pa_manifest_path": str(self.c2pa_manifest_path) if self.c2pa_manifest_path else None,
            "phash_match": self.phash_match,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BenchmarkResult:
    """性能基准测试结果"""
    image_size: tuple[int, int]
    n_iterations: int
    mean_duration_ms: float
    std_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    memory_peak_mb: float
    layer_breakdown: dict[LayerType, float] = field(default_factory=dict)

    def to_str(self) -> str:
        return (
            f"Image {self.image_size[0]}x{self.image_size[1]}: "
            f"{self.mean_duration_ms:.0f}ms ± {self.std_duration_ms:.0f}ms "
            f"(min={self.min_duration_ms:.0f}ms, max={self.max_duration_ms:.0f}ms, "
            f"n={self.n_iterations})\n"
            f"Memory peak: {self.memory_peak_mb:.0f}MB\n"
            f"Layer breakdown:\n" +
            "\n".join(f"  {k.value}: {v:.0f}ms" for k, v in self.layer_breakdown.items())
        )
