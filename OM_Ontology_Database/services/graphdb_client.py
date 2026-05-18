"""
All GraphDB I/O operations. No FastAPI code lives here.

Design decisions:
- Field names are validated against VALID_FIELDS before any SPARQL construction.
- Values are escaped with _escape_sparql_string() before embedding in queries.
- Integers and booleans are type-cast and stored with correct xsd: datatypes.
- Scalar updates: DELETE old + provenance, log HistoryEntry, INSERT new.
- Node updates: for single-value node fields, mark old node as superseded
  then insert new node. For multi-value fields, append by default.
- All operations use chained SPARQL Update statements for closest-to-atomic behaviour.
- sparql_ask() is used for existence checks (never SELECT for this).
"""

import uuid
import re
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from fastapi import HTTPException

from config import (
    SPARQL_QUERY_URL, SPARQL_UPDATE_URL,
    UM_PREFIX, RDF_PREFIX, XSD_PREFIX,
)
from models.um_fields import VALID_FIELDS


# ── Derived constants from schema ────────────────────────────────────────────
# Used in get_child_profile() to separate primary node properties from extras.
# Derived at import time so it stays in sync with um_fields.py automatically.

_NODE_PRIMARY_PROPS = {
    fdef["node_property"]
    for fdef in VALID_FIELDS.values()
    if fdef["storage"] == "node"
}

_NODE_PROVENANCE_PROPS = {"active", "source", "timestamp", "sessionId", "childId"}

_NODE_STANDARD_PROPS = _NODE_PRIMARY_PROPS | _NODE_PROVENANCE_PROPS


# ── Low-level SPARQL helpers ──────────────────────────────────────────────────

def sparql_query(query: str) -> dict:
    """Send SPARQL SELECT to GraphDB. Returns parsed JSON results."""
    resp = requests.post(
        SPARQL_QUERY_URL,
        data={"query": query},
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(500, f"GraphDB SELECT failed ({resp.status_code}): {resp.text}")
    return resp.json()


def sparql_update(update: str) -> None:
    """Send SPARQL UPDATE (INSERT/DELETE) to GraphDB."""
    resp = requests.post(
        SPARQL_UPDATE_URL,
        data={"update": update},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code not in (200, 204):
        raise HTTPException(500, f"GraphDB UPDATE failed ({resp.status_code}): {resp.text}")


def sparql_ask(query: str) -> bool:
    """Send SPARQL ASK query. Returns True or False."""
    resp = requests.post(
        SPARQL_QUERY_URL,
        data={"query": query},
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=10,
    )
    if resp.status_code != 200:
        raise HTTPException(500, f"GraphDB ASK failed ({resp.status_code}): {resp.text}")
    return resp.json().get("boolean", False)


# ── String utilities ──────────────────────────────────────────────────────────

def _escape_sparql_string(value: str) -> str:
    """
    Escape a string value for safe embedding in a SPARQL literal.
    Handles backslash, double-quote, and newlines.
    This does NOT protect against field-name injection — use VALID_FIELDS for that.
    """
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    value = value.replace("\n", "\\n")
    value = value.replace("\r", "\\r")
    # Escape non-ASCII as SPARQL \uXXXX — avoids HTTP encoding ambiguity
    result = ""
    for ch in value:
        if ord(ch) > 127:
            result += f"\\u{ord(ch):04X}"
        else:
            result += ch
    return result


def _sanitize_for_uri(text: str) -> str:
    """
    Make a string safe to embed in a URI component.
    Lowercases, replaces spaces and non-alphanumeric chars with underscores.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_\-]", "_", text)
    text = re.sub(r"_+", "_", text)   # collapse multiple underscores
    if len(text) > 60:
        import hashlib
        short_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return text[:50] + "_" + short_hash
    return text                # cap length for readable URIs


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _child_uri(child_id: str) -> str:
    return f"{UM_PREFIX}child/{_sanitize_for_uri(child_id)}"


def _node_uri(target_class: str, child_id: str, value: str, field_name: str = "") -> str:
    field_part = f"{_sanitize_for_uri(field_name)}/" if field_name else ""
    return (
        f"{UM_PREFIX}node/"
        f"{target_class}/"
        f"{field_part}"
        f"{_sanitize_for_uri(child_id)}/"
        f"{_sanitize_for_uri(value)}"
    )


def _history_uri(child_id: str, field_name: str) -> str:
    uid = str(uuid.uuid4()).replace("-", "")[:12]
    return f"{UM_PREFIX}history/{_sanitize_for_uri(child_id)}/{field_name}/{uid}"


# ── Type casting ──────────────────────────────────────────────────────────────

def _normalize_value(value: Any, field_def: dict) -> tuple[Any, str]:
    """
    Cast value to the correct Python type and return (cast_value, xsd_type_string).
    Raises ValueError with a descriptive message on type mismatch.

    Returns:
        (cast_value, xsd_type) e.g. (9, "xsd:integer") or ("voetbal", "xsd:string")
    """
    ftype = field_def["type"]
    xsd   = field_def.get("xsd_type", "xsd:string")

    if ftype == "integer":
        try:
            cast = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"Expected integer, got: {repr(value)}")
        mn = field_def.get("min")
        mx = field_def.get("max")
        if mn is not None and cast < mn:
            raise ValueError(f"Value {cast} is below minimum {mn}")
        if mx is not None and cast > mx:
            raise ValueError(f"Value {cast} is above maximum {mx}")
        return cast, xsd

    if ftype in ("boolean", "enum"):
        # Normalise to lowercase string
        str_val = str(value).strip().lower()
        allowed = field_def.get("allowed_values", [])
        if allowed and str_val not in allowed:
            raise ValueError(
                f"Value '{str_val}' not in allowed values: {allowed}"
            )
        return str_val, xsd

    # Default: string
    str_val = str(value).strip()
    if not str_val:
        raise ValueError("Value must not be empty")
    return str_val, xsd


# ── SPARQL PREFIX block ───────────────────────────────────────────────────────

_PREFIXES = f"""
PREFIX um:   <{UM_PREFIX}>
PREFIX rdf:  <{RDF_PREFIX}>
PREFIX xsd:  <{XSD_PREFIX}>
"""


# ── Child existence ───────────────────────────────────────────────────────────

def child_exists(child_id: str) -> bool:
    uri = _child_uri(child_id)
    return sparql_ask(f"""
        {_PREFIXES}
        ASK {{ <{uri}> rdf:type um:Child . }}
    """)


# ── Create child ──────────────────────────────────────────────────────────────

def create_child(
    child_id: str,
    age: Optional[int] = None,
    grade: Optional[str] = None,
    session_id: str = "unknown",
) -> None:
    """
    Insert a new Child node. Raises 409 if child already exists.
    Age and grade can be set here at creation or via update_fields later.
    """
    if child_exists(child_id):
        raise HTTPException(409, f"Child '{child_id}' already exists.")

    uri = _child_uri(child_id)
    now = _now_iso()

    # Build the INSERT block with mandatory triples
    triples = f"""
        <{uri}> rdf:type um:Child ;
                um:childId "{_escape_sparql_string(child_id)}" ;
                um:createdAt "{now}"^^xsd:dateTime ;
                um:createdBySession "{_escape_sparql_string(session_id)}" .
    """

    # Optionally add age and grade directly at creation
    if age is not None:
        triples += f'    <{uri}> um:age {int(age)} ;\n'
        triples += f'           um:age_source "initial_create" ;\n'
        triples += f'           um:age_timestamp "{now}"^^xsd:dateTime ;\n'
        triples += f'           um:age_sessionId "{_escape_sparql_string(session_id)}" .\n'
    if grade is not None:
        safe_grade = _escape_sparql_string(str(grade))
        triples += f'    <{uri}> um:grade "{safe_grade}"^^xsd:string ;\n'
        triples += f'           um:grade_source "initial_create" ;\n'
        triples += f'           um:grade_timestamp "{now}"^^xsd:dateTime ;\n'
        triples += f'           um:grade_sessionId "{_escape_sparql_string(session_id)}" .\n'

    sparql_update(f"{_PREFIXES} INSERT DATA {{ {triples} }}")


# ── Read full profile ──────────────────────────────────────────────────────────

def get_child_profile(child_id: str) -> dict:
    """
    Return the full user model as a dict with two sections:
      scalars: { field_name: { value, source, timestamp, session_id } }
      nodes:   { field_name: [ { value, extra_props, source, timestamp } ] }

    Raises 404 if child does not exist.
    """
    if not child_exists(child_id):
        raise HTTPException(404, f"Child '{child_id}' not found.")

    uri = _child_uri(child_id)

    # ── 1. Query all scalar properties on Child node ──────────────────────────
    scalar_q = f"""
    {_PREFIXES}
    SELECT ?prop ?val WHERE {{
        <{uri}> ?prop ?val .
        FILTER(?prop != rdf:type)
        FILTER(!STRSTARTS(STR(?prop), "{UM_PREFIX}hasHistory"))
    }}
    """
    raw = sparql_query(scalar_q)["results"]["bindings"]

    # Group triples: separate main field values from companion _source/_timestamp/_sessionId
    scalar_vals: dict[str, str]  = {}
    companion: dict[str, dict]   = {}   # e.g. {"age": {"source": ..., "timestamp": ...}}

    for row in raw:
        prop_full = row["prop"]["value"]
        val       = row["val"]["value"]
        prop      = prop_full.replace(UM_PREFIX, "")  # strip namespace

        if prop.endswith("_source"):
            base = prop[: -len("_source")]
            companion.setdefault(base, {})["source"] = val
        elif prop.endswith("_timestamp"):
            base = prop[: -len("_timestamp")]
            companion.setdefault(base, {})["timestamp"] = val
        elif prop.endswith("_sessionId"):
            base = prop[: -len("_sessionId")]
            companion.setdefault(base, {})["session_id"] = val
        else:
            # Only include recognised UM fields or base child metadata
            if prop in VALID_FIELDS and VALID_FIELDS[prop]["storage"] == "scalar":
                scalar_vals[prop] = val
            # Always include createdAt, childId for info
            elif prop in ("childId", "createdAt", "createdBySession"):
                scalar_vals[prop] = val

    scalars: dict[str, dict] = {}
    for field, val in scalar_vals.items():
        meta = companion.get(field, {})
        scalars[field] = {
            "value":      val,
            "source":     meta.get("source", "unknown"),
            "timestamp":  meta.get("timestamp", "unknown"),
            "session_id": meta.get("session_id", "unknown"),
        }

    # ── 2. Query all named nodes connected to this child ─────────────────────
    node_q = f"""
    {_PREFIXES}
    SELECT ?rel ?nodeUri ?nodeProp ?nodeVal WHERE {{
        <{uri}> ?rel ?nodeUri .
        ?nodeUri rdf:type ?nodeClass .
        FILTER(STRSTARTS(STR(?nodeClass), "{UM_PREFIX}"))
        FILTER(?nodeClass != um:Child)
        FILTER(?nodeClass != um:HistoryEntry)
        FILTER(?nodeUri != <{uri}>)
        ?nodeUri ?nodeProp ?nodeVal .
        FILTER(?nodeProp != rdf:type)
    }}
    """
    node_raw = sparql_query(node_q)["results"]["bindings"]

    # Build: { relationship_local_name: { nodeUri: { props } } }
    node_map: dict[str, dict[str, dict]] = {}
    for row in node_raw:
        rel_full  = row["rel"]["value"].replace(UM_PREFIX, "")
        node_uri  = row["nodeUri"]["value"]
        prop_full = row["nodeProp"]["value"].replace(UM_PREFIX, "")
        val       = row["nodeVal"]["value"]
        node_map.setdefault(rel_full, {}).setdefault(node_uri, {})[prop_full] = val

    # Map relationship names back to UM field names
    rel_to_field = {
        fdef["relationship"]: fname
        for fname, fdef in VALID_FIELDS.items()
        if fdef["storage"] == "node"
    }
    nodes: dict[str, list[dict]] = {}
    for rel, node_dict in node_map.items():
        field_name = rel_to_field.get(rel, rel)
        entries = []
        for n_uri, props in node_dict.items():
            # Skip superseded nodes
            if props.get("active", "true") == "false":
                continue
            entries.append({
                "node_uri":   n_uri,
                "value":      props.get(VALID_FIELDS.get(field_name, {}).get("node_property", "name"), ""),
                "extra_props": {
                    k: v for k, v in props.items()
                    if k not in _NODE_STANDARD_PROPS
                },
                "source":     props.get("source", "unknown"),
                "timestamp":  props.get("timestamp", "unknown"),
                "session_id": props.get("sessionId", "unknown"),
            })
        if entries:
            nodes[field_name] = entries

    return {"scalars": scalars, "nodes": nodes}


# ── History logging ───────────────────────────────────────────────────────────

def _log_history(
    child_id: str,
    field_name: str,
    old_value: Optional[str],
    new_value: str,
    source: str,
    session_id: str,
) -> None:
    """
    Create a HistoryEntry node linked to the child.
    Called before every write that replaces an existing value.
    """
    if old_value is None:
        return   # nothing to record — this is initial creation, not an update

    child_uri_str = _child_uri(child_id)
    h_uri         = _history_uri(child_id, field_name)
    now           = _now_iso()

    sparql_update(f"""
    {_PREFIXES}
    INSERT DATA {{
        <{h_uri}> rdf:type um:HistoryEntry ;
                  um:forChild "{_escape_sparql_string(child_id)}" ;
                  um:field "{_escape_sparql_string(field_name)}" ;
                  um:previousValue "{_escape_sparql_string(str(old_value))}" ;
                  um:newValue "{_escape_sparql_string(str(new_value))}" ;
                  um:changedAt "{now}"^^xsd:dateTime ;
                  um:changedBy "{_escape_sparql_string(source)}" ;
                  um:sessionId "{_escape_sparql_string(session_id)}" .
        <{child_uri_str}> um:hasHistory <{h_uri}> .
    }}
    """)


def _log_reaffirmed(
    child_id: str,
    field_name: str,
    value: str,
    source: str,
    session_id: str,
) -> None:
    """
    Log that a field value was submitted again with the same value.
    Lighter than a full HistoryEntry — uses um:changeType "reaffirmed"
    so it's easy to filter in queries.

    The main value + provenance on the Child node is NOT overwritten.
    """
    child_uri_str = _child_uri(child_id)
    h_uri         = _history_uri(child_id, field_name)
    now           = _now_iso()

    sparql_update(f"""
    {_PREFIXES}
    INSERT DATA {{
        <{h_uri}> rdf:type um:HistoryEntry ;
                  um:forChild "{_escape_sparql_string(child_id)}" ;
                  um:field "{_escape_sparql_string(field_name)}" ;
                  um:previousValue "{_escape_sparql_string(value)}" ;
                  um:newValue "{_escape_sparql_string(value)}" ;
                  um:changeType "reaffirmed" ;
                  um:changedAt "{now}"^^xsd:dateTime ;
                  um:changedBy "{_escape_sparql_string(source)}" ;
                  um:sessionId "{_escape_sparql_string(session_id)}" .
        <{child_uri_str}> um:hasHistory <{h_uri}> .
    }}
    """)


# ── Write scalar field ────────────────────────────────────────────────────────

def _write_scalar(
    child_id: str,
    field_name: str,
    value: Any,
    source: str,
    session_id: str,
) -> None:
    """
    Write a scalar field to the Child node.
    Steps:
      1. Read current value (for history log).
      2. Log HistoryEntry if previous value exists.
      3. DELETE old value + companion triples.
      4. INSERT new value + companion triples.
    All in one SPARQL Update request (chained with semicolons).
    """
    field_def     = VALID_FIELDS[field_name]
    cast_val, xsd = _normalize_value(value, field_def)
    child_uri_str = _child_uri(child_id)
    now           = _now_iso()
    safe_src      = _escape_sparql_string(source)
    safe_sess     = _escape_sparql_string(session_id)

    # Read current value for history
    old_val = _read_scalar_raw(child_id, field_name)

    # If value is unchanged: keep original provenance, log reaffirmation only
    if old_val is not None and str(old_val) == str(cast_val):
        _log_reaffirmed(child_id, field_name, str(cast_val), source, session_id)
        return

    _log_history(child_id, field_name, old_val, str(cast_val), source, session_id)

    # Format the literal correctly for SPARQL
    if xsd == "xsd:integer":
        literal = f"{cast_val}"   # bare integer — valid SPARQL 1.1 shorthand for xsd:integer
    else:
        literal = f'"{_escape_sparql_string(str(cast_val))}"^^xsd:string'

    # Chained DELETE + INSERT in one update request
    sparql_update(f"""
    {_PREFIXES}
    DELETE {{
        <{child_uri_str}> um:{field_name} ?v .
        <{child_uri_str}> um:{field_name}_source ?s .
        <{child_uri_str}> um:{field_name}_timestamp ?t .
        <{child_uri_str}> um:{field_name}_sessionId ?sid .
    }}
    WHERE {{
        OPTIONAL {{ <{child_uri_str}> um:{field_name} ?v }}
        OPTIONAL {{ <{child_uri_str}> um:{field_name}_source ?s }}
        OPTIONAL {{ <{child_uri_str}> um:{field_name}_timestamp ?t }}
        OPTIONAL {{ <{child_uri_str}> um:{field_name}_sessionId ?sid }}
    }};
    INSERT DATA {{
        <{child_uri_str}> um:{field_name} {literal} ;
                          um:{field_name}_source "{safe_src}" ;
                          um:{field_name}_timestamp "{now}"^^xsd:dateTime ;
                          um:{field_name}_sessionId "{safe_sess}" .
    }}
    """)


def _read_scalar_raw(child_id: str, field_name: str) -> Optional[str]:
    """Read a scalar field value as a raw string. Returns None if not set."""
    uri = _child_uri(child_id)
    r = sparql_query(f"""
    {_PREFIXES}
    SELECT ?v WHERE {{ <{uri}> um:{field_name} ?v }}
    """)
    bindings = r["results"]["bindings"]
    return bindings[0]["v"]["value"] if bindings else None


# ── Write node field ──────────────────────────────────────────────────────────

def _write_node(
    child_id: str,
    field_name: str,
    value: str,
    source: str,
    session_id: str,
    extra_props: dict[str, str] | None = None,
) -> None:
    """
    Write a node-type field.

    For single-value fields: mark any existing node as superseded, then insert new.
    For multi-value fields: always append a new node (do not remove old ones).

    The node URI is deterministic: um:node/{Class}/{child_id}/{sanitized_value}
    This means writing the same value twice is idempotent — the second write
    updates provenance metadata but does not create a duplicate node.
    """
    if extra_props is None:
        extra_props = {}

    field_def     = VALID_FIELDS[field_name]
    multi_value   = field_def["multi_value"]
    rel           = field_def["relationship"]
    target_class  = field_def["target_class"]
    node_prop     = field_def["node_property"]

    cast_val, _   = _normalize_value(value, field_def)
    str_val       = str(cast_val)
    child_uri_str = _child_uri(child_id)
    new_node_uri = _node_uri(target_class, child_id, str_val, field_name)
    now           = _now_iso()
    safe_src      = _escape_sparql_string(source)
    safe_sess     = _escape_sparql_string(session_id)
    safe_val      = _escape_sparql_string(str_val)

    # Check if this exact node already exists and is active — if so, it's
    # a reaffirmation, not a new write. Log it lightly and skip overwrite.
    if sparql_ask(f"""
        {_PREFIXES}
        ASK {{
            <{child_uri_str}> um:{rel} <{new_node_uri}> .
            <{new_node_uri}> um:active "true"^^xsd:string
        }}
        """):
        _log_reaffirmed(child_id, field_name, str_val, source, session_id)
        return

    # For single-value: mark existing node as superseded
    if not multi_value:
        old_node_uri = _get_current_node_uri(child_id, field_name)
        if old_node_uri and old_node_uri != new_node_uri:
            _log_history(
                child_id, field_name,
                old_value=_get_node_prop_value(old_node_uri, node_prop),
                new_value=str_val,
                source=source,
                session_id=session_id,
            )
            # Mark old node as superseded and remove the relationship
            sparql_update(f"""
                        {_PREFIXES}
                        DELETE {{ <{child_uri_str}> um:{rel} <{old_node_uri}> }}
                        WHERE  {{ <{child_uri_str}> um:{rel} <{old_node_uri}> }};
                        DELETE {{ <{old_node_uri}> um:active ?oldActive }}
                        WHERE  {{ <{old_node_uri}> um:active ?oldActive }};
                        INSERT DATA {{
                            <{old_node_uri}> um:active "false"^^xsd:string .
                            <{old_node_uri}> um:SUPERSEDED_BY <{new_node_uri}> .
                        }}
                        """)

    # Build extra property triples for the new node
    extra_triples = ""
    for prop_name, prop_val in extra_props.items():
        safe_prop_val = _escape_sparql_string(str(prop_val))
        extra_triples += f'        <{new_node_uri}> um:{prop_name} "{safe_prop_val}"^^xsd:string .\n'

    # INSERT the new node (idempotent: DELETE+INSERT so re-writing same value updates provenance)
    sparql_update(f"""
    {_PREFIXES}
    DELETE {{
        <{new_node_uri}> um:{node_prop} ?oldProp .
        <{new_node_uri}> um:source ?oldSrc .
        <{new_node_uri}> um:timestamp ?oldTs .
        <{new_node_uri}> um:sessionId ?oldSid .
    }}
    WHERE {{
        OPTIONAL {{ <{new_node_uri}> um:{node_prop} ?oldProp }}
        OPTIONAL {{ <{new_node_uri}> um:source ?oldSrc }}
        OPTIONAL {{ <{new_node_uri}> um:timestamp ?oldTs }}
        OPTIONAL {{ <{new_node_uri}> um:sessionId ?oldSid }}
    }};
    INSERT DATA {{
        <{new_node_uri}> rdf:type um:{target_class} ;
                         um:{node_prop} "{safe_val}"^^xsd:string ;
                         um:active "true"^^xsd:string ;
                         um:source "{safe_src}" ;
                         um:timestamp "{now}"^^xsd:dateTime ;
                         um:sessionId "{safe_sess}" ;
                         um:childId "{_escape_sparql_string(child_id)}" .
{extra_triples}
        <{child_uri_str}> um:{rel} <{new_node_uri}> .
    }}
    """)


def _get_current_node_uri(child_id: str, field_name: str) -> Optional[str]:
    """Get the URI of the currently active node for a single-value node field."""
    uri      = _child_uri(child_id)
    rel      = VALID_FIELDS[field_name]["relationship"]
    r = sparql_query(f"""
    {_PREFIXES}
    SELECT ?nodeUri WHERE {{
        <{uri}> um:{rel} ?nodeUri .
        ?nodeUri um:active "true"^^xsd:string .
    }} LIMIT 1
    """)
    bindings = r["results"]["bindings"]
    return bindings[0]["nodeUri"]["value"] if bindings else None


def _get_node_prop_value(node_uri: str, node_prop: str) -> Optional[str]:
    """Read a single property from a node URI."""
    r = sparql_query(f"""
    {_PREFIXES}
    SELECT ?v WHERE {{ <{node_uri}> um:{node_prop} ?v }} LIMIT 1
    """)
    bindings = r["results"]["bindings"]
    return bindings[0]["v"]["value"] if bindings else None


# ── Public write dispatcher ───────────────────────────────────────────────────

def write_field(
    child_id: str,
    field_name: str,
    value: Any,
    source: str,
    session_id: str,
    extra_props: dict[str, str] | None = None,
) -> None:
    """
    Write any UM field, dispatching to scalar or node handler.

    field_name MUST be in VALID_FIELDS — enforced by the caller (main.py),
    but we double-check here as a safety net.

    For multi-value node fields, value can be a single string or a list.
    Each list item is written as a separate node.
    """
    if field_name not in VALID_FIELDS:
        raise HTTPException(400, f"Unknown field: '{field_name}'. Not in schema.")

    if not child_exists(child_id):
        raise HTTPException(404, f"Child '{child_id}' not found.")

    field_def = VALID_FIELDS[field_name]

    if field_def["storage"] == "scalar":
        _write_scalar(child_id, field_name, value, source, session_id)

    else:  # node
        # Accept list or single value
        values = value if isinstance(value, list) else [value]
        for v in values:
            _write_node(
                child_id, field_name, str(v).strip(),
                source, session_id, extra_props or {},
            )


# ── Delete field ──────────────────────────────────────────────────────────────

def delete_scalar_field(child_id: str, field_name: str) -> None:
    """
    Delete a scalar field. Logs a HistoryEntry noting the deletion.
    Does not raise if field is already absent — idempotent.
    """
    child_uri_str = _child_uri(child_id)
    old_val = _read_scalar_raw(child_id, field_name)
    if old_val is not None:
        _log_history(
            child_id, field_name,
            old_value=old_val,
            new_value="[DELETED]",
            source="child_requested_delete",
            session_id="gui",
        )
    sparql_update(f"""
    {_PREFIXES}
    DELETE {{
        <{child_uri_str}> um:{field_name} ?v .
        <{child_uri_str}> um:{field_name}_source ?s .
        <{child_uri_str}> um:{field_name}_timestamp ?t .
        <{child_uri_str}> um:{field_name}_sessionId ?sid .
    }}
    WHERE {{
        OPTIONAL {{ <{child_uri_str}> um:{field_name} ?v }}
        OPTIONAL {{ <{child_uri_str}> um:{field_name}_source ?s }}
        OPTIONAL {{ <{child_uri_str}> um:{field_name}_timestamp ?t }}
        OPTIONAL {{ <{child_uri_str}> um:{field_name}_sessionId ?sid }}
    }}
    """)


def delete_node_field_value(child_id: str, field_name: str, value: str) -> None:
    """
    Remove a specific value from a node field.
    For multi-value fields, removes just that one value.
    Marks the node as active=false rather than hard-deleting (preserves history).
    """
    field_def     = VALID_FIELDS[field_name]
    rel           = field_def["relationship"]
    target_class  = field_def["target_class"]
    child_uri_str = _child_uri(child_id)
    node_uri = _node_uri(target_class, child_id, value, field_name)

    old_val = _get_node_prop_value(node_uri, field_def["node_property"])
    if old_val is not None:
        _log_history(
            child_id, field_name,
            old_value=old_val,
            new_value="[DELETED]",
            source="child_requested_delete",
            session_id="gui",
        )

    sparql_update(f"""
        {_PREFIXES}
        DELETE {{ <{child_uri_str}> um:{rel} <{node_uri}> }}
        WHERE  {{ <{child_uri_str}> um:{rel} <{node_uri}> }};
        DELETE {{ <{node_uri}> um:active ?oldActive }}
        WHERE  {{ <{node_uri}> um:active ?oldActive }};
        INSERT DATA {{ <{node_uri}> um:active "false"^^xsd:string . }}
        """)


def delete_child(child_id: str) -> None:
    """
    Hard-delete ALL triples where child is the subject,
    AND all HistoryEntry nodes belonging to this child,
    AND all node triples for this child's nodes.
    GDPR-compliant full erasure.
    """
    child_uri_str = _child_uri(child_id)

    # 1. Delete all child-uri triples
    sparql_update(f"""
    {_PREFIXES}
    DELETE WHERE {{ <{child_uri_str}> ?p ?o }}
    """)

    # 2. Delete all history entries for this child
    sparql_update(f"""
    {_PREFIXES}
    DELETE WHERE {{
        ?h rdf:type um:HistoryEntry ;
           um:forChild "{_escape_sparql_string(child_id)}" .
        ?h ?p ?o .
    }}
    """)

    # 3. Delete all named nodes belonging to this child
    sparql_update(f"""
    {_PREFIXES}
    DELETE WHERE {{
        ?node um:childId "{_escape_sparql_string(child_id)}" .
        ?node ?p ?o .
    }}
    """)


# ── History query ─────────────────────────────────────────────────────────────

def get_history(child_id: str, field_name: Optional[str] = None) -> list[dict]:
    """
    Return history entries for a child, optionally filtered to one field.
    Ordered by changedAt descending (most recent first).
    """
    uri   = _child_uri(child_id)
    filter_clause = (
        f'FILTER(?field = "{_escape_sparql_string(field_name)}"^^xsd:string)'
        if field_name else ""
    )
    r = sparql_query(f"""
    {_PREFIXES}
    SELECT ?field ?oldVal ?newVal ?changedAt ?changedBy ?sessionId ?changeType WHERE {{
        <{uri}> um:hasHistory ?h .
        ?h um:field ?field ;
           um:previousValue ?oldVal ;
           um:newValue ?newVal ;
           um:changedAt ?changedAt ;
           um:changedBy ?changedBy .
        OPTIONAL {{ ?h um:sessionId ?sessionId }}
        OPTIONAL {{ ?h um:changeType ?changeType }}
        {filter_clause}
    }}
    ORDER BY DESC(?changedAt)
    """)
    rows = []
    for b in r["results"]["bindings"]:
        rows.append({
            "field":       b["field"]["value"],
            "old_value":   b["oldVal"]["value"],
            "new_value":   b["newVal"]["value"],
            "changed_at":  b["changedAt"]["value"],
            "changed_by":  b["changedBy"]["value"],
            "session_id":  b.get("sessionId", {}).get("value", "unknown"),
            "change_type": b.get("changeType", {}).get("value", "updated"),
        })
    return rows