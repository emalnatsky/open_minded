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

CLOSED_LIST_FIELDS = {
    "fav_subject", "school_strength", "sports_fav_play",
}

OPEN_VALUE_FIELDS = {
    "hobbies", "hobby_fav", "freetime_fav", "fav_food", "books_fav_title",
    "animal_fav", "pet_type", "pet_name", "interest", "aspiration",
    "role_model", "name", "age",
}

YES_NO_MAYBE_FIELDS = {
    "sports_enjoys", "books_enjoys", "music_enjoys", "animals_enjoys",
    "has_pet", "has_best_friend",
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
        "spelen", "buiten spelen", "met vrienden spelen",
        "lego bouwen", "playmobiel bouwen",
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


def _clean_value(value: str) -> str:
    clean = str(value or "").strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip(" .,!?;:")


def _split_candidate_values(value: str) -> list[str]:
    clean = _clean_value(value)
    if not clean:
        return []
    parts = re.split(r"\s*(?:,|/|&|\ben\b|\band\b)\s*", clean, flags=re.IGNORECASE)
    return [_clean_value(part) for part in parts if _clean_value(part)]


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


def _validate_yes_no_maybe_value(value: str) -> dict:
    clean = _clean_value(value)
    norm = _norm(clean)
    yes_values = {
        "ja", "ja hoor", "jawel", "zeker", "jazeker", "ja zeker",
        "klopt", "dat klopt", "inderdaad", "ja inderdaad",
    }
    no_values = {
        "nee", "nee hoor", "nee joh", "niet", "niet echt", "helemaal niet",
        "absoluut niet", "zeker niet", "nooit", "ja nee",
    }
    maybe_values = {
        "misschien", "een beetje", "beetje", "soms", "mwah", "mwa",
    }
    if norm in yes_values:
        return {"accepted": True, "normalized_value": "ja", "reason": "yes_no_yes", "needs_gpt": False}
    if norm in no_values:
        return {"accepted": True, "normalized_value": "nee", "reason": "yes_no_no", "needs_gpt": False}
    if norm in maybe_values:
        normalized = "een beetje" if norm in {"een beetje", "beetje"} else norm
        return {"accepted": True, "normalized_value": normalized, "reason": "yes_no_maybe", "needs_gpt": False}
    return {"accepted": False, "normalized_value": clean, "reason": "not_yes_no_maybe", "needs_gpt": False}


def _open_value_junk_reason(value: str) -> str:
    clean = _clean_value(value)
    norm = _norm(clean)
    if not clean:
        return "empty"
    if len(clean) > 80:
        return "too_long"
    if re.search(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9 ,.'-]", clean):
        return "symbol_heavy"
    if norm in {
        "ja", "ja hoor", "jawel", "klopt", "dat klopt", "nee", "nee hoor",
        "niet echt", "weet ik niet", "ik weet het niet", "geen idee",
        "misschien", "mwah", "mwa", "niks", "niets", "geen",
    }:
        return "non_value_phrase"
    if norm in {
        "hobby", "eten", "lievelingseten", "lievelingsvak", "vak",
        "beroep", "inspiratie", "sport", "school",
    }:
        return "field_name_not_value"
    vague_phrases = (
        "dat vind ik", "dat is leuk", "heel leuk", "best leuk", "niet leuk",
        "ik vind", "ik hou", "ik houd", "dat doe ik", "geen idee",
    )
    if any(phrase in norm for phrase in vague_phrases):
        return "vague_sentence"
    return ""


def validate_memory_value_local(field: str, value: str, transcript: str = "", turn: dict = None) -> dict:
    """Field-aware local validation for a proposed UM value."""
    clean = _clean_value(value)
    if not clean:
        return {"accepted": False, "normalized_value": clean, "reason": "empty", "needs_gpt": False}

    snapped, matched = snap_to_valid_value(field, clean)
    if field in CLOSED_LIST_FIELDS:
        if field in {"fav_subject", "school_strength"} and _norm(clean) == "spelen":
            return {
                "accepted": False,
                "normalized_value": clean,
                "reason": "closed_list_no_match",
                "needs_gpt": False,
            }
        parts = _split_candidate_values(clean)
        if len(parts) > 1:
            normalized_parts = []
            for part in parts:
                if field in {"fav_subject", "school_strength"} and _norm(part) == "spelen":
                    return {
                        "accepted": False,
                        "normalized_value": clean,
                        "reason": "closed_list_no_match",
                        "needs_gpt": False,
                    }
                part_snapped, part_matched = snap_to_valid_value(field, part)
                if not part_matched:
                    return {
                        "accepted": False,
                        "normalized_value": clean,
                        "reason": "closed_list_no_match",
                        "needs_gpt": False,
                    }
                normalized_parts.append(part if _norm(part_snapped) == _norm(part) else part_snapped)
            return {
                "accepted": True,
                "normalized_value": " en ".join(dict.fromkeys(normalized_parts)),
                "reason": "closed_list_multi_match",
                "needs_gpt": False,
            }
        normalized = clean if matched and _norm(snapped) == _norm(clean) else snapped
        return {
            "accepted": bool(matched),
            "normalized_value": normalized if matched else clean,
            "reason": "closed_list_match" if matched else "closed_list_no_match",
            "needs_gpt": False,
        }

    if field in YES_NO_MAYBE_FIELDS:
        return _validate_yes_no_maybe_value(clean)

    open_snapped, open_matched = snap_to_valid_value(field, clean, cutoff=0.9)
    if open_matched:
        return {
            "accepted": True,
            "normalized_value": open_snapped,
            "reason": "known_open_value_match",
            "needs_gpt": False,
        }

    if field in OPEN_VALUE_FIELDS or field not in CLOSED_LIST_FIELDS:
        junk_reason = _open_value_junk_reason(clean)
        if junk_reason:
            return {
                "accepted": False,
                "normalized_value": clean,
                "reason": junk_reason,
                "needs_gpt": False,
            }
        word_count = len(_norm(clean).split())
        if word_count <= 5 and len(clean) <= 45:
            return {
                "accepted": True,
                "normalized_value": clean,
                "reason": "clean_open_value",
                "needs_gpt": False,
            }
        return {
            "accepted": False,
            "normalized_value": clean,
            "reason": "ambiguous_open_value",
            "needs_gpt": True,
        }

    return {"accepted": True, "normalized_value": clean, "reason": "unvalidated_field", "needs_gpt": False}


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
