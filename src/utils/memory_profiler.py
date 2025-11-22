"""
Memory monitoring and profiling utilities.

Provides:
- Memory usage tracking
- Memory limit enforcement
- Memory-based circuit breaker
"""

import functools
import logging
from typing import Callable, Dict

import psutil

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and limit memory usage."""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current process memory usage.

        Returns:
            Dict with memory metrics in MB and percent
        """
        process = psutil.Process()
        mem_info = process.memory_info()

        return {
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),     # % of system memory
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    @staticmethod
    def get_system_memory() -> Dict[str, float]:
        """
        Get system-wide memory usage.

        Returns:
            Dict with system memory metrics
        """
        mem = psutil.virtual_memory()

        return {
            "total_mb": mem.total / 1024 / 1024,
            "available_mb": mem.available / 1024 / 1024,
            "used_mb": mem.used / 1024 / 1024,
            "percent": mem.percent,
            "free_mb": mem.free / 1024 / 1024,
        }

    @staticmethod
    def memory_limit(max_mb: int = 512):
        """
        Decorator to enforce memory limits on functions.

        Args:
            max_mb: Maximum memory increase allowed (in MB)

        Example:
            @MemoryMonitor.memory_limit(max_mb=256)
            def expensive_operation():
                # Operation that might use a lot of memory
                pass
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                mem_before = MemoryMonitor.get_memory_usage()

                result = func(*args, **kwargs)

                mem_after = MemoryMonitor.get_memory_usage()
                delta = mem_after["rss_mb"] - mem_before["rss_mb"]

                if delta > max_mb:
                    logger.warning(
                        "memory_limit_exceeded",
                        extra={
                            "function": func.__name__,
                            "delta_mb": delta,
                            "limit_mb": max_mb,
                        }
                    )
                    raise MemoryError(
                        f"Function '{func.__name__}' used {delta:.1f}MB, "
                        f"limit is {max_mb}MB"
                    )

                return result
            return wrapper
        return decorator

    @staticmethod
    def log_memory_usage(label: str = "memory_check") -> None:
        """
        Log current memory usage.

        Args:
            label: Label for the log entry
        """
        mem = MemoryMonitor.get_memory_usage()
        logger.info(
            label,
            extra={
                "rss_mb": round(mem["rss_mb"], 1),
                "percent": round(mem["percent"], 1),
                "available_mb": round(mem["available_mb"], 1),
            }
        )


def check_memory_threshold(threshold_percent: float = 85.0) -> bool:
    """
    Check if system memory usage exceeds threshold.

    Args:
        threshold_percent: Memory usage threshold (0-100)

    Returns:
        True if memory usage is below threshold, False otherwise
    """
    mem = psutil.virtual_memory()
    return mem.percent < threshold_percent
