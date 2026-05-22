"""
cri_logger package — conversation logging + resume for the CRI dialogue.

Owned by the dialogue class but kept here so the 4500-line monolith doesn't
also have to host ~400 lines of logging boilerplate.

Public surface:
    from cri_logger import ConversationLogger, ResumeHelper
"""

from .conversation_logger import ConversationLogger
from .resume import ResumeHelper

__all__ = ["ConversationLogger", "ResumeHelper"]
