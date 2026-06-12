"""Bit consensus - 多层水印 bit 提取的一致性检查

当 TrustMark + DWT + Cox 三种方法都提取到 bits,
通过投票/对比找出最可信的 bits。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConsensusResult:
    """bit 一致性检查结果"""
    consensus_bits: np.ndarray
    confidence: float
    methods: list[str]
    agreement_matrix: Optional[np.ndarray] = None  # (n_bits, n_methods)

    def to_dict(self) -> dict:
        return {
            "consensus_bits": self.consensus_bits.tolist()[:64] + (["..."] if len(self.consensus_bits) > 64 else []),
            "n_bits": len(self.consensus_bits),
            "confidence": self.confidence,
            "methods": self.methods,
        }


def bit_voting(
    bits_dict: dict[str, np.ndarray],
    target_teacher_id: str,
) -> ConsensusResult:
    """多方法 bit 投票

    Args:
        bits_dict: {"trustmark": bits1, "dwt": bits2, "cox": bits3}
        target_teacher_id: 目标教师 ID (用于参考)
    """
    if not bits_dict:
        return ConsensusResult(
            consensus_bits=np.array([], dtype=np.uint8),
            confidence=0.0,
            methods=[],
        )

    # 规范化到相同长度 (用最短的)
    min_len = min(len(b) for b in bits_dict.values() if len(b) > 0)
    if min_len == 0:
        return ConsensusResult(
            consensus_bits=np.array([], dtype=np.uint8),
            confidence=0.0,
            methods=list(bits_dict.keys()),
        )

    methods = []
    bits_list = []
    for name, bits in bits_dict.items():
        if len(bits) >= min_len:
            methods.append(name)
            bits_list.append(bits[:min_len])

    if not bits_list:
        return ConsensusResult(
            consensus_bits=np.array([], dtype=np.uint8),
            confidence=0.0,
            methods=list(bits_dict.keys()),
        )

    bits_array = np.array(bits_list)  # (n_methods, min_len)

    # 投票: 对每个 bit, 取多数
    consensus = (np.sum(bits_array, axis=0) > bits_array.shape[0] / 2).astype(np.uint8)

    # 计算一致性: 平均 pairwise agreement
    n_methods = bits_array.shape[0]
    agreements = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            agreements.append(np.mean(bits_array[i] == bits_array[j]))
    confidence = np.mean(agreements) if agreements else 0.0

    return ConsensusResult(
        consensus_bits=consensus,
        confidence=float(confidence),
        methods=methods,
        agreement_matrix=bits_array,
    )


def consensus_against_reference(
    consensus: np.ndarray,
    reference: np.ndarray,
) -> float:
    """consensus 与参考 bits 的相似度"""
    n = min(len(consensus), len(reference))
    if n == 0:
        return 0.0
    return float(np.mean(consensus[:n] == reference[:n]))
