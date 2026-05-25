"""
classifier package — intent classification for the CRI dialogue.

Public surface (import from here, not from sub-modules):

    from classifier import IntentResult, REPEAT_SENTINEL, VALID_INTENTS
    from classifier import StubIntentClassifier, GPTIntentClassifier
"""

from .intent_result import IntentResult, REPEAT_SENTINEL, VALID_INTENTS
from .stub import StubIntentClassifier
from .gpt import GPTIntentClassifier

__all__ = [
    "IntentResult",
    "REPEAT_SENTINEL",
    "VALID_INTENTS",
    "StubIntentClassifier",
    "GPTIntentClassifier",
]
