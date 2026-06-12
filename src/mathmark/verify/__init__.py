"""Verify module - 水印提取、验证、报告生成"""
from .consensus import ConsensusResult, bit_voting, consensus_against_reference
from .extractor import extract_all, verify_image
from .report import (
    generate_legal_report,
    generate_markdown_report,
    generate_pdf_report,
    generate_text_report,
)

__all__ = [
    "ConsensusResult",
    "bit_voting",
    "consensus_against_reference",
    "extract_all",
    "extract_all",
    "generate_legal_report",
    "generate_markdown_report",
    "generate_pdf_report",
    "generate_text_report",
    "verify_image",
]
