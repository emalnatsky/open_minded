"""
cri_memory package — phase-scoped memory access for the CRI dialogue.

Public surface:
    from cri_memory import MemoryAccess
"""

from .access import MemoryAccess as _BaseMemoryAccess


class MemoryAccess(_BaseMemoryAccess):
    """Memory access with child-facing role_model normalization."""


__all__ = ["MemoryAccess"]
