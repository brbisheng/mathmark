"""Semantic module - 数学语义水印 (L4 核心)"""
from .example_db import (
    ExampleMatch,
    hash_problem,
    index_signature_problems,
    match_examples,
    normalize_problem,
)
from .injector import (
    InjectionReport,
    InjectionSuggestion,
    create_injection_report,
    generate_suggestions,
    inject_to_latex,
    inject_to_pptx,
    inject_to_text,
    save_injection_report,
)
from .ocr import (
    OCRResult,
    OCRToken,
    get_engine,
    recognize,
)
from .recognizer import (
    RecognitionResult,
    recognize_from_image,
    recognize_from_ocr,
    recognize_from_text,
    recognize_multi,
)
from .step_matcher import StepMatch, match_steps
from .symbol_matcher import SymbolMatch, match_symbols

__all__ = [
    "ExampleMatch",
    "InjectionReport",
    "InjectionSuggestion",
    "OCRResult",
    "OCRToken",
    "RecognitionResult",
    "StepMatch",
    "SymbolMatch",
    "create_injection_report",
    "generate_suggestions",
    "get_engine",
    "hash_problem",
    "index_signature_problems",
    "inject_to_latex",
    "inject_to_pptx",
    "inject_to_text",
    "match_examples",
    "match_steps",
    "match_symbols",
    "normalize_problem",
    "recognize",
    "recognize_from_image",
    "recognize_from_ocr",
    "recognize_from_text",
    "recognize_multi",
    "save_injection_report",
]
