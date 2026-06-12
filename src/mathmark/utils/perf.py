"""性能监控工具

提供 context manager 用于测量代码块耗时,
以及内存使用监控。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class TimingResult:
    duration_ms: float


@contextmanager
def measure_time(name: str = "", verbose: bool = False) -> Iterator[TimingResult]:
    """测量代码块耗时的 context manager

    Usage:
        with measure_time("my_op") as t:
            do_something()
        print(t.duration_ms)
    """
    result = TimingResult(duration_ms=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        end = time.perf_counter()
        result.duration_ms = (end - start) * 1000.0
        if verbose:
            print(f"[{name}] {result.duration_ms:.2f}ms")


def get_memory_mb() -> float:
    """获取当前进程内存使用 (MB)

    跨平台:尝试使用 psutil, 失败则用 resource (仅 Linux/Mac)。
    """
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS: ru_maxrss in bytes; Linux: ru_maxrss in KB
        if hasattr(resource, "RUSAGE_SELF"):
            # 跨平台
            return usage.ru_maxrss / 1024 / 1024
    except (ImportError, ValueError):
        pass

    return 0.0
