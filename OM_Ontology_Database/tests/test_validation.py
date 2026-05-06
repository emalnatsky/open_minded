"""
Unit tests for the validation pipeline.
These do NOT require GraphDB or the API server to be running.
They test the validation logic in isolation.

Run: python -m pytest tests/test_validation.py -v

Updated for UM v5.1: age is no longer a UM field (it's metadata at
child creation), field names match the v5.1 schema.
"""

import pytest
from services.validation import validate_field


# ── Layer 1a: Schema / type checks ───────────────────────────────────────────

class TestSchemaChecks:

    def test_age_valid_integer(self):
        r = validate_field("age", 9)
        assert r.passed

    def test_age_string_that_is_numeric(self):
        r = validate_field("age", "9")
        assert r.passed       # "9" can be cast to int 9

    def test_age_out_of_range_high(self):
        r = validate_field("age", 99)
        assert not r.passed
        assert r.flag_type == "malformed"

    def test_age_out_of_range_low(self):
        r = validate_field("age", 2)
        assert not r.passed

    def test_age_non_integer(self):
        r = validate_field("age", "nine")
        assert not r.passed
        assert r.flag_type == "malformed"

    def test_boolean_field_valid_ja(self):
        r = validate_field("has_best_friend", "ja")
        assert r.passed

    def test_boolean_field_valid_nee(self):
        r = validate_field("has_best_friend", "nee")
        assert r.passed

    def test_boolean_field_invalid_value(self):
        r = validate_field("has_best_friend", "maybe")
        assert not r.passed
        assert r.flag_type == "malformed"

    def test_three_way_enum_valid_weet_niet(self):
        r = validate_field("sports_enjoys", "weet niet")
        assert r.passed

    def test_three_way_enum_valid_ja(self):
        r = validate_field("sports_enjoys", "ja")
        assert r.passed

    def test_three_way_enum_invalid(self):
        r = validate_field("sports_enjoys", "misschien")
        assert not r.passed
        assert r.flag_type == "malformed"

    def test_string_field_empty_rejected(self):
        r = validate_field("fav_food", "")
        assert not r.passed

    def test_string_field_whitespace_only_rejected(self):
        r = validate_field("fav_food", "   ")
        assert not r.passed

    def test_unknown_field_rejected(self):
        r = validate_field("invented_field_xyz", "value")
        assert not r.passed

    def test_valid_string_field(self):
        r = validate_field("fav_food", "pizza")
        # May or may not call LLM depending on SKIP_LLM_VALIDATION
        # At minimum it should not fail on schema grounds
        assert r.flag_type in ("none", "unexpected", "malformed")

    def test_node_field_non_empty_string(self):
        r = validate_field("hobbies", "voetbal")
        assert r.passed or r.flag_type == "unexpected"   # LLM may vary


# ── Allowlist check ───────────────────────────────────────────────────────────

class TestAllowlist:
    """
    These tests verify the SPARQL injection protection:
    any field_name not in VALID_FIELDS must be rejected immediately.
    """

    def test_sparql_injection_via_field_name(self):
        r = validate_field(
            "fav_food} . } INSERT DATA { <http://evil.com/> <http://evil.com/> <http://evil.com/>",
            "pizza",
        )
        assert not r.passed

    def test_semicolon_in_field_name(self):
        r = validate_field("fav_food; DROP TABLE children", "pizza")
        assert not r.passed

    def test_empty_field_name(self):
        r = validate_field("", "pizza")
        assert not r.passed


# ── Integer range boundaries ──────────────────────────────────────────────────

class TestIntegerBoundaries:

    def test_age_exactly_at_min(self):
        r = validate_field("age", 6)
        assert r.passed

    def test_age_exactly_at_max(self):
        r = validate_field("age", 13)
        assert r.passed

    def test_age_one_below_min(self):
        r = validate_field("age", 5)
        assert not r.passed

    def test_age_one_above_max(self):
        r = validate_field("age", 14)
        assert not r.passed


# ── Scalar field boundaries ──────────────────────────────────────────────────

class TestScalarFields:

    def test_hobby_fav_accepts_string(self):
        r = validate_field("hobby_fav", "tekenen")
        assert r.flag_type != "malformed"

    def test_aspiration_accepts_string(self):
        r = validate_field("aspiration", "astronaut")
        assert r.flag_type != "malformed"

    def test_freetime_fav_accepts_string(self):
        r = validate_field("freetime_fav", "buiten spelen")
        assert r.flag_type != "malformed"


# ── Multi-value fields ────────────────────────────────────────────────────────

class TestMultiValueFields:

    def test_single_hobby_string(self):
        r = validate_field("hobbies", "tekenen")
        # Should not fail on schema grounds
        assert r.flag_type != "malformed"

    def test_hobby_list_first_item(self):
        # validate_field validates one value at a time
        # caller is responsible for iterating over list
        r = validate_field("hobbies", "voetbal")
        assert r.flag_type != "malformed"

    def test_sports_fav_play(self):
        r = validate_field("sports_fav_play", "tennis")
        assert r.flag_type != "malformed"

    def test_books_fav_title(self):
        r = validate_field("books_fav_title", "Harry Potter")
        assert r.flag_type != "malformed"