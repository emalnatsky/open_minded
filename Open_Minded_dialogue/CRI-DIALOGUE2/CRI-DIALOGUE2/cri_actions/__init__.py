"""
cri_actions package — intent-to-action routing for the CRI dialogue.

Public surface:
    from cri_actions import ActionHandler, NudgeManager
"""

from .handler import ActionHandler
from .nudge import NudgeManager

__all__ = ["ActionHandler", "NudgeManager"]
