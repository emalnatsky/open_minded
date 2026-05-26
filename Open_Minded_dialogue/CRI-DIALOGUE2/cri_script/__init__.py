"""
cri_script package — content plan + script-builder for the CRI dialogue.

The 9-phase script (Greeting, Tutorial, Leo mini-story, Correct hobby bridge,
Topic 1, Mistake 1, Topic 2, Mistake 2, Nudge) lives here, plus the layered
content-plan machinery (L1, L2-slot, L2-pregen, sequence, render).

Public surface:
    from cri_script import ContentPlan, Segments, ScriptBuilder

The dialogue keeps thin pass-through wrappers so existing call sites
(self.build_script(), self.turn_text(turn), self.l1(...), etc.) stay
identical.
"""

from .content_plan import ContentPlan
from .segments import Segments
from .builder import ScriptBuilder

__all__ = ["ContentPlan", "Segments", "ScriptBuilder"]
