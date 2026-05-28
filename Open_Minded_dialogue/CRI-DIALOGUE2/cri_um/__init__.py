"""
cri_um package — UM (User Model) I/O for the CRI dialogue.

All HTTP calls to Eunike's FastAPI live here. Read fields, bulk-pull
profiles, pull pregenerated utterances, write/delete changes.

Public surface:
    from cri_um import UMClient
"""

from .client import UMClient as _BaseUMClient


class UMClient(_BaseUMClient):
    """UM client with Dialogue 2 child-facing normalization."""

    NO_ROLE_MODEL_VALUES = {
        "geen",
        "nee",
        "niemand",
        "weet ik niet",
        "ik weet het niet",
        "ik weet niet",
        "geen idee",
        "niet",
        "nvt",
        "n.v.t.",
        "niet van toepassing",
        "geen rolmodel",
        "geen role model",
        "geen voorbeeld",
    }

    def is_meaningful_role_model(self, value: str) -> bool:
        """Return True only when role_model names an actual person/example."""
        if not self.is_known(value):
            return False
        clean = " ".join(str(value).strip().lower().split())
        compact = clean.replace(".", "").replace("-", "").replace("/", "")
        return clean not in self.NO_ROLE_MODEL_VALUES and compact not in self.NO_ROLE_MODEL_VALUES

    def meaningful_role_model(self, um: dict, fallback: str = "") -> str:
        value = um.get("role_model", self.d.UNKNOWN_VALUE)
        return value if self.is_meaningful_role_model(value) else fallback

    def known(self, um: dict, field: str, fallback: str = "") -> str:
        if field == "role_model":
            return self.meaningful_role_model(um, fallback)
        return super().known(um, field, fallback)


__all__ = ["UMClient"]
