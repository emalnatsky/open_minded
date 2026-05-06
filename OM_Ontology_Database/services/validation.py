"""
Two-layer validation pipeline.

Layer 1a — Schema check (deterministic, always runs):
  - Field name must be in VALID_FIELDS (enforced before this is called).
  - Value must match the declared type (integer, boolean enum, string).
  - Integer values must be within min/max range.
  - Enum values must be in the allowed_values list.
  - String values must be non-empty after stripping.

Layer 1b — SHACL check (structural, runs if shapes file exists):
  - Constructs a tiny rdflib graph with just the new triple(s).
  - For scalar fields: adds a datatype property on a dummy Child node.
  - For node fields: adds the relationship + a typed target node with
    the correct node_property — matching how graphdb_client.py stores them.
  - Validates against um_shapes.ttl using pyshacl.
  - If shapes file not found: logs a warning and skips.

Layer 2 — LLM semantic check (runs only if llm_validate=True for the field):
  - Calls Ollama via llm_client.validate_field_value().
  - Catches unexpected/malformed values (joke answers, gibberish, etc.).
  - If Ollama unavailable: skips and logs a warning (fail open).

A ValidationResult is returned with all relevant information.
Validation does NOT write to GraphDB — that is the caller's responsibility.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, XSD

from config import UM_PREFIX, SHACL_SHAPES_PATH, OWL_ONTOLOGY_PATH
from models.um_fields import VALID_FIELDS
from services import llm_client

logger = logging.getLogger(__name__)

UM = Namespace(UM_PREFIX)


@dataclass
class ValidationResult:
    passed: bool
    field_name: str
    value: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    flag_type: str = "none"       # "none" | "malformed" | "unexpected"
    llm_skipped: bool = False


def validate_field(
    field_name: str,
    value,
    extra_props: dict | None = None,
) -> ValidationResult:
    """
    Run the full validation pipeline for a single field + value.
    Returns a ValidationResult. The caller decides what to do with it.

    extra_props: e.g. {"motivation": "because it's fun"} for node fields
    """
    result = ValidationResult(passed=True, field_name=field_name, value=str(value))

    if field_name not in VALID_FIELDS:
        result.passed = False
        result.errors.append(f"Unknown field '{field_name}'. Not in schema allowlist.")
        return result

    field_def = VALID_FIELDS[field_name]

    # ── Layer 1a: Schema / type check ────────────────────────────────────────
    schema_error = _check_schema(field_name, value, field_def)
    if schema_error:
        result.passed = False
        result.flag_type = "malformed"
        result.errors.append(schema_error)
        return result   # no point running further checks on malformed input

    # ── Layer 1b: SHACL structural check ─────────────────────────────────────
    shacl_error = _check_shacl(field_name, value, field_def)
    if shacl_error:
        result.passed = False
        result.flag_type = "malformed"
        result.errors.append(shacl_error)
        return result

    # ── Layer 2: LLM semantic check ───────────────────────────────────────────
    if field_def.get("llm_validate", False):
        llm_result = llm_client.validate_field_value(
            field_name=field_name,
            value=str(value),
            description=field_def.get("description", field_name),
        )
        result.llm_skipped = llm_result.get("skipped", False)

        if not result.llm_skipped and not llm_result["valid"]:
            result.passed = False
            result.flag_type = llm_result.get("flag_type", "unexpected")
            result.errors.append(
                f"LLM flagged value as {result.flag_type}: {llm_result['reason']}"
            )
        elif result.llm_skipped:
            result.warnings.append(llm_result["reason"])

    return result


# ── Layer 1a: schema check ────────────────────────────────────────────────────

def _check_schema(field_name: str, value, field_def: dict) -> Optional[str]:
    """
    Deterministic structural validation.
    Returns an error message string, or None if the check passes.
    """
    ftype = field_def["type"]

    if ftype == "integer":
        try:
            cast = int(value)
        except (ValueError, TypeError):
            return f"Field '{field_name}' expects an integer, got: {repr(value)}"
        mn = field_def.get("min")
        mx = field_def.get("max")
        if mn is not None and cast < mn:
            return f"Field '{field_name}': value {cast} is below minimum {mn}"
        if mx is not None and cast > mx:
            return f"Field '{field_name}': value {cast} is above maximum {mx}"
        return None

    if ftype in ("boolean", "enum"):
        str_val = str(value).strip().lower()
        allowed = field_def.get("allowed_values", [])
        if allowed and str_val not in allowed:
            return (
                f"Field '{field_name}': value '{str_val}' is not in "
                f"allowed values: {allowed}"
            )
        return None

    # String type: check non-empty
    str_val = str(value).strip()
    if not str_val:
        return f"Field '{field_name}': value must not be empty"

    return None


# ── Layer 1b: SHACL check ────────────────────────────────────────────────────

def _check_shacl(field_name: str, value, field_def: dict) -> Optional[str]:
    """
    Validate the proposed data against the SHACL shapes file.

    For SCALAR fields: creates a dummy Child with the datatype property.
    For NODE fields: creates a dummy Child with the relationship pointing
    to a typed target node — matching how graphdb_client.py stores data.

    Returns an error message if violation found, or None if valid.
    Silently skips and logs a warning if shapes file is not yet present.
    """
    if not os.path.exists(SHACL_SHAPES_PATH):
        logger.warning(
            "SHACL shapes file not found at '%s'. "
            "Skipping SHACL validation for field '%s'. "
            "Build um_shapes.ttl in Protégé and place it in ontology/.",
            SHACL_SHAPES_PATH, field_name,
        )
        return None

    # Lazy import: pyshacl may not be installed in all environments
    try:
        from pyshacl import validate as shacl_validate
    except ImportError:
        logger.warning("pyshacl not installed. pip install pyshacl. Skipping SHACL check.")
        return None

    # Build a tiny in-memory RDF graph with the proposed triple(s)
    data_graph = Graph()
    data_graph.bind("um", UM)

    child_subject = UM["child/validation_subject"]
    data_graph.add((child_subject, RDF.type, UM["Child"]))
    # Add mandatory identity properties so the ChildShape doesn't fail
    # on minCount requirements unrelated to the field being validated
    data_graph.add((child_subject, UM["childId"], Literal("validation_subject", datatype=XSD.string)))
    data_graph.add((child_subject, UM["createdAt"], Literal("2026-01-01T00:00:00Z", datatype=XSD.dateTime)))

    if field_def["storage"] == "scalar":
        # Scalar field: add as a datatype property on Child
        predicate = UM[field_name]
        if field_def["type"] == "integer":
            obj = Literal(int(value), datatype=XSD.integer)
        else:
            obj = Literal(str(value).strip().lower() if field_def["type"] in ("boolean", "enum") else str(value).strip(), datatype=XSD.string)
        data_graph.add((child_subject, predicate, obj))

    else:
        # Node field: add the relationship to a typed target node
        # This mirrors how graphdb_client._write_node() creates triples:
        #   <child> um:LIKES_HOBBY <node>
        #   <node>  rdf:type       um:Hobby
        #   <node>  um:name        "voetbal"
        relationship = field_def["relationship"]
        target_class = field_def["target_class"]
        node_prop    = field_def["node_property"]

        node_subject = UM[f"node/validation/{target_class}/test"]
        data_graph.add((child_subject, UM[relationship], node_subject))
        data_graph.add((node_subject, RDF.type, UM[target_class]))
        data_graph.add((node_subject, UM[node_prop], Literal(str(value).strip(), datatype=XSD.string)))

    shapes_graph = Graph()
    shapes_graph.parse(SHACL_SHAPES_PATH, format="turtle")

    # Optionally load the OWL ontology for richer inference during validation
    ont_graph = None
    if os.path.exists(OWL_ONTOLOGY_PATH):
        ont_graph = Graph()
        ont_graph.parse(OWL_ONTOLOGY_PATH, format="turtle")

    conforms, _, report_text = shacl_validate(
        data_graph,
        shacl_graph=shapes_graph,
        ont_graph=ont_graph,
        inference="none",
        abort_on_first=True,
    )

    if not conforms:
        violation_summary = _extract_shacl_message(report_text)
        return f"SHACL violation for '{field_name}': {violation_summary}"

    return None


def _extract_shacl_message(report_text: str) -> str:
    """Pull the first sh:resultMessage from the SHACL report string."""
    for line in report_text.splitlines():
        if "Constraint Violation" in line or "resultMessage" in line:
            return line.strip()
    return report_text[:200].strip()