"""MathMark - 多层抗AI去水印工具，专为数学教学内容版权保护设计"""

__version__ = "0.1.0"
__author__ = "MathMark Project"

from .core.config import (
    create_default_config,
    load_config,
    load_signature,
    save_config,
    save_signature,
)
from .core.types import (
    BenchmarkResult,
    C2PASettings,
    ContentType,
    CoxSettings,
    DWTSettings,
    Evidence,
    ExampleSignature,
    LayerReport,
    LayerType,
    MathSignature,
    MetadataSettings,
    SemanticSettings,
    SignerInfo,
    SignatureProblem,
    StepSignature,
    SymbolSignature,
    TrustMarkSettings,
    Verdict,
    VerificationResult,
    VisibleSettings,
    VisualSignature,
    WatermarkConfig,
    WatermarkResult,
)


_LAZY_EXPORTS = {
    "WatermarkPipeline": (".core.pipeline", "WatermarkPipeline"),
    "extract_all": (".verify.extractor", "extract_all"),
    "verify_image": (".verify.extractor", "verify_image"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BenchmarkResult",
    "C2PASettings",
    "ContentType",
    "CoxSettings",
    "DWTSettings",
    "Evidence",
    "ExampleSignature",
    "LayerReport",
    "LayerType",
    "MathSignature",
    "MetadataSettings",
    "SemanticSettings",
    "SignerInfo",
    "SignatureProblem",
    "StepSignature",
    "SymbolSignature",
    "TrustMarkSettings",
    "Verdict",
    "VerificationResult",
    "VisibleSettings",
    "VisualSignature",
    "WatermarkConfig",
    "WatermarkPipeline",
    "WatermarkResult",
    "__version__",
    "__author__",
    "create_default_config",
    "extract_all",
    "load_config",
    "load_signature",
    "save_config",
    "save_signature",
    "verify_image",
]
