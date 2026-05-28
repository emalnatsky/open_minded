"""
cri_memory package — phase-scoped memory access for the CRI dialogue.

Public surface:
    from cri_memory import MemoryAccess
"""

from .access import MemoryAccess as _BaseMemoryAccess


class MemoryAccess(_BaseMemoryAccess):
    """Memory access with child-facing role_model normalization."""

    def is_child_facing_memory_field(self, field: str) -> bool:
        if field == "role_model":
            value = self.d.last_um_preview.get(field)
            if not self.d.um.is_meaningful_role_model(value):
                return False
        return super().is_child_facing_memory_field(field)


__all__ = ["MemoryAccess"]
