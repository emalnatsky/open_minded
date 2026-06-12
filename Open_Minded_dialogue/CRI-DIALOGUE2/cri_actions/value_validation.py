# -*- coding: utf-8 -*-
"""Snap a (possibly mis-heard) STT value to the nearest known valid value.

Whisper mangles open-ended Dutch words ("spelling" -> "speel in",
"gamen" -> "game in"). For fields where the set of sensible answers is small
and known (school subjects, foods, hobbies, sports), we keep a canonical list
plus common mis-hear aliases, and snap the transcript onto it before storing.
"""
import re
import unicodedata
from difflib import SequenceMatcher

# Fields that have a canonical list — anything that doesn't snap to this list
# is blocked by is_valid_field_value().
VALIDATED_FIELDS = {
    "fav_subject", "school_strength", "fav_food", "hobby_fav", "sports_fav_play",
}

# Canonical valid values per field. Extend freely.
VALID_VALUES = {
    "fav_subject": [
        "rekenen", "taal", "spelling", "lezen", "begrijpend lezen", "schrijven",
        "geschiedenis", "aardrijkskunde", "biologie", "natuur",
        "wereldorientatie", "gym", "muziek", "tekenen", "knutselen",
        "handvaardigheid", "engels", "frans", "duits", "verkeer",
        "natuur en techniek", "wetenschap", "techniek",
    ],
    # school_strength shares the same domain as fav_subject
    "school_strength": [
        "rekenen", "taal", "spelling", "lezen", "begrijpend lezen", "schrijven",
        "geschiedenis", "aardrijkskunde", "biologie", "natuur",
        "wereldorientatie", "gym", "muziek", "tekenen", "knutselen",
        "handvaardigheid", "engels", "frans", "duits", "verkeer",
        "natuur en techniek", "wetenschap", "techniek",
    ],
    "fav_food": [
        "pizza", "pasta", "spaghetti", "patat", "friet", "frietjes",
        "broccoli", "spruitjes", "groente",
        "pannenkoeken", "boterham", "brood",
        "appel", "banaan", "fruit",
        "sushi", "rijst", "noodles",
        "macaroni", "lasagne",
        "soep", "kaas", "snoep", "ijs",
        "kip", "vis", "vlees",
        "hamburger", "hagelslag", "pindakaas",
        "yoghurt", "smoothie",
        "taart", "cake", "koekjes", "chocolade",
        "stamppot", "hutspot", "erwtensoep",
    ],
    "hobby_fav": [
        "gamen", "computerspelletjes", "spelletjes",
        "voetbal", "voetballen",
        "hockey", "hockeyen",
        "tennis", "tennissen",
        "zwemmen",
        "turnen",
        "dansen", "ballet",
        "paardrijden", "paarden",
        "fietsen", "skateboarden", "skeeleren",
        "tekenen", "schilderen",
        "lezen", "boeken",
        "knutselen", "bouwen", "lego",
        "zingen", "muziek maken",
        "koken", "bakken",
        "klimmen",
        "schaatsen", "skieen",
        "toneel", "theater",
        "fotograferen",
    ],
    "sports_fav_play": [
        "voetbal", "voetballen",
        "hockey", "hockeyen",
        "tennis", "tennissen",
        "zwemmen",
        "turnen",
        "basketbal", "basketballen",
        "volleybal",
        "handbal",
        "atletiek", "hardlopen",
        "fietsen", "wielrennen",
        "judo", "karate", "taekwondo",
        "dansen", "ballet",
        "paardrijden",
        "schaatsen",
        "skieen",
        "rugby",
        "golf",
    ],
}

# Explicit mis-hear -> canonical, for the ones fuzzy matching alone may miss.
MISHEAR_ALIASES = {
    "fav_subject": {
        "speel in": "spelling", "speelin": "spelling", "spelen": "spelling",
        "speling": "spelling", "reken": "rekenen",
        "aard rijkskunde": "aardrijkskunde", "aardrijks kunde": "aardrijkskunde",
        "gym les": "gym", "gymles": "gym",
        "begrijpend": "begrijpend lezen",
    },
    "school_strength": {
        "speel in": "spelling", "speelin": "spelling", "spelen": "spelling",
        "speling": "spelling", "reken": "rekenen",
        "aard rijkskunde": "aardrijkskunde", "aardrijks kunde": "aardrijkskunde",
        "gym les": "gym", "gymles": "gym",
        "begrijpend": "begrijpend lezen",
    },
    "hobby_fav": {
        "game in": "gamen", "gaming": "gamen", "gamern": "gamen",
        "gemen": "gamen", "gamel": "gamen",
        "voetbal len": "voetballen",
    },
    "fav_food": {
        "broccolli": "broccoli", "brocoli": "broccoli",
        "panakoeken": "pannenkoeken", "pannekoeken": "pannenkoeken",
        "pata": "patat", "pattat": "patat",
        "spagetti": "spaghetti", "spaghetie": "spaghetti",
    },
    "sports_fav_play": {
        "voetbal len": "voetballen",
        "hard lopen": "hardlopen",
    },
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", text).strip()


def _despace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def snap_to_valid_value(field: str, value: str, cutoff: float = 0.72):
    """Return (canonical_value, matched_bool).

    Tries in order:
      1. Explicit mishear alias (exact or de-spaced)
      2. Exact canonical match
      3. Fuzzy ratio on de-spaced normalised forms

    If no list exists for the field, or no confident match is found,
    returns the original value and False — never forces a wrong snap.
    """
    options = VALID_VALUES.get(field)
    if not options or not value:
        return value, False

    norm = _norm(value)
    aliases = MISHEAR_ALIASES.get(field, {})

    # 1) explicit alias — exact or de-spaced
    if norm in aliases:
        return aliases[norm], True
    despace_alias_map = {_despace(_norm(k)): v for k, v in aliases.items()}
    if _despace(norm) in despace_alias_map:
        return despace_alias_map[_despace(norm)], True

    # 2) exact canonical match
    norm_options = {_norm(o): o for o in options}
    if norm in norm_options:
        return norm_options[norm], True

    # 3) fuzzy: best ratio on de-spaced normalised forms
    best, best_ratio = None, 0.0
    nd = _despace(norm)
    for opt in options:
        r = SequenceMatcher(None, nd, _despace(_norm(opt))).ratio()
        if r > best_ratio:
            best, best_ratio = opt, r
    if best is not None and best_ratio >= cutoff:
        return best, True

    return value, False


def is_valid_field_value(field: str, value: str) -> bool:
    """Return True if value is acceptable to store for this field.

    For fields in VALIDATED_FIELDS: only values that snap confidently to the
    canonical list are accepted. For all other fields: always True (no filter).

    This is the guard that prevents junk like "heel leuk" or "dat vind ik"
    from being stored as a fav_subject or fav_food.
    """
    if field not in VALIDATED_FIELDS:
        return True
    if not value:
        return False
    _, matched = snap_to_valid_value(field, value)
    return matched


if __name__ == "__main__":
    snap_tests = [
        ("fav_subject", "speel in"),
        ("fav_subject", "spelling"),
        ("fav_subject", "reekenen"),
        ("fav_subject", "aard rijkskunde"),
        ("fav_subject", "aardrijkskunde"),
        ("fav_subject", "voetbal"),        # not a subject -> no snap
        ("fav_subject", "heel erg leuk"),  # junk -> no snap
        ("school_strength", "reken"),
        ("school_strength", "gym les"),
        ("hobby_fav", "game in"),
        ("hobby_fav", "gamen"),
        ("hobby_fav", "voetballe"),
        ("fav_food", "brocolli"),
        ("fav_food", "pizza"),
        ("fav_food", "iets heel anders xyz"),  # nonsense -> no snap
        ("sports_fav_play", "voetbal len"),
    ]
    print("=== snap_to_valid_value ===")
    for f, v in snap_tests:
        out, matched = snap_to_valid_value(f, v)
        print(f"  {f:18} {v!r:26} -> {out!r:22} matched={matched}")

    print("\n=== is_valid_field_value ===")
    guard_tests = [
        ("fav_subject", "rekenen"),
        ("fav_subject", "speel in"),
        ("fav_subject", "heel leuk"),
        ("fav_food", "pizza"),
        ("fav_food", "juist niet lekker"),
        ("hobby_fav", "gamen"),
        ("hobby_fav", "dat vind ik"),
        ("aspiration", "whatever"),   # not validated -> always True
    ]
    for f, v in guard_tests:
        print(f"  {f:18} {v!r:26} -> {is_valid_field_value(f, v)}")
