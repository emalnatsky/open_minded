"""
cri_um package — UM (User Model) I/O for the CRI dialogue.

All HTTP calls to Eunike's FastAPI live here. Read fields, bulk-pull
profiles, pull pregenerated utterances, write/delete changes.

Public surface:
    from cri_um import UMClient
"""

from .client import UMClient

__all__ = ["UMClient"]
