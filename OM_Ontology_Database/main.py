"""
FastAPI application entry point.

Rules enforced here:
  - All routes validate field_name against VALID_FIELDS BEFORE calling any service.
  - No SPARQL lives here. All DB calls go through services/graphdb_client.py.
  - No validation logic lives here. All validation goes through services/validation.py.
  - API key check is applied to all write endpoints (X-API-Key header).
  - Responses are consistent dicts with "status" and "data" keys.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from models.um_fields import VALID_FIELDS, CATEGORY_LABELS, SENSITIVITY_TIERS
from services import graphdb_client as db
from services.validation import validate_field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Open-Memory Robots — UM Memory Service",
    description=(
        "Knowledge-graph-based User Model service for child-robot interaction. "
        "DECRI (Define, Create, Read, Inspect, Update, Delete) endpoints. "
        "Test test version 3"
    ),
    version="0.2.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # restrict to known origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API key dependency ────────────────────────────────────────────────────────

def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Dependency: check X-API-Key header on write endpoints."""
    if config.SKIP_API_KEY_CHECK:
        return
    if x_api_key != config.API_KEY:
        raise HTTPException(401, "Missing or invalid X-API-Key header.")


# ── Pydantic request models ───────────────────────────────────────────────────

class ChildCreate(BaseModel):
    child_id: str
    age: Optional[int] = None
    grade: Optional[str] = None
    session_id: str = "initial"

    model_config = {
        "json_schema_extra": {
            "example": {
                "child_id": "child_001",
                "age": 9,
                "grade": "groep 7",
                "session_id": "checkin_2025_04"
            }
        }
    }


class FieldUpdate(BaseModel):
    """
    fields: { field_name: value } where value can be string, int, or list.
    extra_props: optional companion properties, e.g. {"hobbies_motivation": "..."}
    source: who provided this data.
    session_id: which session this came from.
    """
    fields: dict[str, Any]
    extra_props: dict[str, str] = {}
    source: str = "child_reported"
    session_id: Optional[str] = None
    dry_run: bool = False   # if True: validate only, do not write to GraphDB

    model_config = {
        "json_schema_extra": {
            "example": {
                "fields": {
                    "animal_fav": "hond",
                    "hobbies": "voetbal",
                    "fav_food": "pizza"
                },
                "extra_props": {
                    "hobbies_motivation": "omdat het leuk is"
                },
                "source": "child_reported",
                "session_id": "checkin_001"
            }
        }
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_field_name(field_name: str) -> None:
    """Reject unknown field names before any SPARQL is constructed."""
    if field_name not in VALID_FIELDS:
        raise HTTPException(
            400,
            f"Unknown field '{field_name}'. "
            f"Valid fields are: {sorted(VALID_FIELDS.keys())}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# HEALTH
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Open-Memory Robots UM Service",
        "status": "running",
        "graphdb": config.GRAPHDB_URL,
        "repository": config.GRAPHDB_REPOSITORY,
        "api_docs": "/docs",
    }


@app.get("/health/graphdb", tags=["Health"])
def check_graphdb():
    """Verify GraphDB is reachable and the repository exists."""
    try:
        result = db.sparql_query(
            "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }"
        )
        count = result["results"]["bindings"][0]["count"]["value"]
        return {"status": "connected", "total_triples": int(count)}
    except Exception as e:
        raise HTTPException(503, f"GraphDB unreachable: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# DEFINE — return the fixed schema
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/schema", tags=["Define"])
def get_schema():
    """
    Return the fixed UM field schema.
    Consumers call this to know which fields exist and their metadata.
    The schema is defined in models/um_fields.py and never changes at runtime.
    """
    return {
        "status": "ok",
        "data": {
            "fields": VALID_FIELDS,
            "categories": CATEGORY_LABELS,
            "sensitivity_tiers": SENSITIVITY_TIERS,
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# CREATE
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/um/", tags=["Create"], dependencies=[Depends(require_api_key)])
def create_child(child: ChildCreate):
    """
    Create a new Child node.
    Validates age if provided. Returns 409 if child already exists.
    """
    # Validate age through the UM validation pipeline (age is a v5.1 field
    # with type=integer, min=6, max=13 defined in VALID_FIELDS)
    if child.age is not None:
        res = validate_field("age", child.age)
        if not res.passed:
            raise HTTPException(422, {"field": "age", "errors": res.errors})

    db.create_child(
        child_id=child.child_id,
        age=child.age,
        grade=child.grade,
        session_id=child.session_id,
    )
    return {
        "status": "created",
        "data": {"child_id": child.child_id}
    }


# ──────────────────────────────────────────────────────────────────────────────
# READ
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/um/{child_id}", tags=["Read"])
def get_profile(child_id: str):
    """Full user model for a child — both scalar fields and node-based fields."""
    profile = db.get_child_profile(child_id)
    return {"status": "ok", "data": {"child_id": child_id, "profile": profile}}


@app.get("/api/um/{child_id}/field/{field_name}", tags=["Read"])
def get_field(child_id: str, field_name: str):
    """Get a specific field value with its provenance metadata."""
    _validate_field_name(field_name)
    if not db.child_exists(child_id):
        raise HTTPException(404, f"Child '{child_id}' not found.")

    field_def = VALID_FIELDS[field_name]

    if field_def["storage"] == "scalar":
        val = db._read_scalar_raw(child_id, field_name)
        if val is None:
            raise HTTPException(404, f"Field '{field_name}' not set for child '{child_id}'.")
        return {"status": "ok", "data": {"field": field_name, "value": val}}

    else:
        node_uri = db._get_current_node_uri(child_id, field_name)
        if node_uri is None:
            raise HTTPException(404, f"Field '{field_name}' not set for child '{child_id}'.")
        val = db._get_node_prop_value(node_uri, field_def["node_property"])
        return {"status": "ok", "data": {"field": field_name, "value": val, "node_uri": node_uri}}


@app.get("/api/um/{child_id}/category/{category}", tags=["Read"])
def get_by_category(child_id: str, category: str):
    """
    Return only fields belonging to a specific category.
    Called by Julianna's GUI to display one category panel at a time.
    """
    if category not in CATEGORY_LABELS:
        raise HTTPException(400, f"Unknown category '{category}'. Valid: {list(CATEGORY_LABELS.keys())}")

    profile = db.get_child_profile(child_id)

    fields_in_category = {
        fname for fname, fdef in VALID_FIELDS.items()
        if fdef.get("category") == category
    }

    scalars = {
        k: v for k, v in profile["scalars"].items()
        if k in fields_in_category
    }
    nodes = {
        k: v for k, v in profile["nodes"].items()
        if k in fields_in_category
    }

    return {
        "status": "ok",
        "data": {
            "child_id": child_id,
            "category": category,
            "label": CATEGORY_LABELS[category],
            "scalars": scalars,
            "nodes": nodes,
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# UPDATE
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/um/{child_id}/fields", tags=["Update"], dependencies=[Depends(require_api_key)])
def update_fields(child_id: str, update: FieldUpdate):
    """
    Add or update one or more UM fields.

    Validation runs per-field before any write.
    If a field fails validation, it is skipped and reported in the response.
    Other fields in the same request continue to be processed.

    If dry_run=True: validates all fields but writes nothing. Returns what
    would have happened. Useful for pre-flight checks from the GUI.

    Companion properties (e.g. hobbies_motivation) should be passed in
    extra_props, not in fields.
    """
    if not db.child_exists(child_id):
        raise HTTPException(404, f"Child '{child_id}' not found.")

    session = update.session_id or "unknown"
    results = {"written": [], "skipped": [], "warnings": []}

    for field_name, value in update.fields.items():
        # Step 1: reject unknown fields — allowlist check
        if field_name not in VALID_FIELDS:
            results["skipped"].append({
                "field": field_name,
                "reason": f"Unknown field '{field_name}'. Not in schema."
            })
            continue

        # Step 2: validate
        res = validate_field(field_name, value, update.extra_props)

        if not res.passed:
            results["skipped"].append({
                "field": field_name,
                "value": str(value),
                "flag_type": res.flag_type,
                "errors": res.errors,
            })
            continue

        if res.warnings:
            results["warnings"].append({
                "field": field_name,
                "warnings": res.warnings,
            })

        # Step 3: write (unless dry run)
        if not update.dry_run:
            # Collect companion properties for this field (e.g. motivation)
            field_def     = VALID_FIELDS[field_name]
            relevant_extra = {}
            for extra_key in field_def.get("extra_node_props", []):
                # Convention: companion key = "{field_name}_{extra_prop}"
                # e.g. "hobbies_motivation" for hobbies field
                compound_key = f"{field_name}_{extra_key}"
                if compound_key in update.extra_props:
                    relevant_extra[extra_key] = update.extra_props[compound_key]
                elif extra_key in update.extra_props:
                    relevant_extra[extra_key] = update.extra_props[extra_key]

            # Multi-value node fields: split comma-separated values into
            # individual nodes. e.g. "padel, voetbal, surfen" becomes
            # three separate Sport nodes, not one blob.
            if (field_def.get("multi_value", False)
                    and field_def["storage"] == "node"
                    and isinstance(value, str) and "," in value):
                individual_values = [v.strip() for v in value.split(",") if v.strip()]
                for single_val in individual_values:
                    # Validate each individual value
                    single_res = validate_field(field_name, single_val)
                    if single_res.passed:
                        db.write_field(
                            child_id=child_id,
                            field_name=field_name,
                            value=single_val,
                            source=update.source,
                            session_id=session,
                            extra_props=relevant_extra,
                        )
                    else:
                        results["warnings"].append({
                            "field": field_name,
                            "value": single_val,
                            "warnings": [f"Individual value skipped: {single_res.errors}"],
                        })
            else:
                db.write_field(
                    child_id=child_id,
                    field_name=field_name,
                    value=value,
                    source=update.source,
                    session_id=session,
                    extra_props=relevant_extra,
                )

        results["written"].append({"field": field_name, "value": str(value)})

    return {
        "status": "ok" if not results["skipped"] else "partial",
        "dry_run": update.dry_run,
        "data": results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# INSPECT — history
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/um/{child_id}/history", tags=["Inspect"])
def get_history(child_id: str):
    """Full change history for all fields of a child."""
    if not db.child_exists(child_id):
        raise HTTPException(404, f"Child '{child_id}' not found.")
    history = db.get_history(child_id)
    return {"status": "ok", "data": {"child_id": child_id, "history": history}}


@app.get("/api/um/{child_id}/history/{field_name}", tags=["Inspect"])
def get_field_history(child_id: str, field_name: str):
    """Change history for a specific field."""
    _validate_field_name(field_name)
    if not db.child_exists(child_id):
        raise HTTPException(404, f"Child '{child_id}' not found.")
    history = db.get_history(child_id, field_name)
    return {
        "status": "ok",
        "data": {"child_id": child_id, "field": field_name, "history": history}
    }


@app.get("/api/um/{child_id}/inspect", tags=["Inspect"])
def inspect_categorized(child_id: str):
    """
    Returns the full profile organized by GUI category with change history
    counts per field. Called by Julianna's GUI for the main memory overview.
    """
    profile = db.get_child_profile(child_id)
    history = db.get_history(child_id)

    # Count changes per field for the GUI "last updated" indicator
    change_counts: dict[str, int] = {}
    for h in history:
        change_counts[h["field"]] = change_counts.get(h["field"], 0) + 1

    categorized: dict[str, dict] = {}
    for cat, label in CATEGORY_LABELS.items():
        cat_scalars = {
            f: v for f, v in profile["scalars"].items()
            if VALID_FIELDS.get(f, {}).get("category") == cat
        }
        cat_nodes = {
            f: v for f, v in profile["nodes"].items()
            if VALID_FIELDS.get(f, {}).get("category") == cat
        }
        if cat_scalars or cat_nodes:
            categorized[cat] = {
                "label": label,
                "scalars": cat_scalars,
                "nodes": cat_nodes,
            }

    return {
        "status": "ok",
        "data": {
            "child_id": child_id,
            "categories": categorized,
            "change_counts": change_counts,
            "total_changes": len(history),
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# DELETE
# ──────────────────────────────────────────────────────────────────────────────

@app.delete("/api/um/{child_id}/field/{field_name}", tags=["Delete"],
            dependencies=[Depends(require_api_key)])
def delete_field(child_id: str, field_name: str):
    """
    Delete a specific field (child says 'forget this').
    Preserves history — marks deletion in HistoryEntry.
    """
    _validate_field_name(field_name)
    if not db.child_exists(child_id):
        raise HTTPException(404, f"Child '{child_id}' not found.")

    field_def = VALID_FIELDS[field_name]
    if field_def["storage"] == "scalar":
        db.delete_scalar_field(child_id, field_name)
    else:
        # For node fields without a specific value, delete the current active node
        node_uri = db._get_current_node_uri(child_id, field_name)
        if node_uri:
            val = db._get_node_prop_value(node_uri, field_def["node_property"]) or ""
            db.delete_node_field_value(child_id, field_name, val)

    return {
        "status": "deleted",
        "data": {"child_id": child_id, "field": field_name}
    }


@app.delete("/api/um/{child_id}/field/{field_name}/value/{value}",
            tags=["Delete"], dependencies=[Depends(require_api_key)])
def delete_node_field_value(child_id: str, field_name: str, value: str):
    """
    Delete a specific value from a multi-value node field.
    E.g. remove one hobby from a list of hobbies.
    """
    _validate_field_name(field_name)
    if not db.child_exists(child_id):
        raise HTTPException(404, f"Child '{child_id}' not found.")
    if VALID_FIELDS[field_name]["storage"] != "node":
        raise HTTPException(400, f"Field '{field_name}' is a scalar — use DELETE /field/{field_name}")

    db.delete_node_field_value(child_id, field_name, value)
    return {
        "status": "deleted",
        "data": {"child_id": child_id, "field": field_name, "value": value}
    }


@app.delete("/api/um/{child_id}", tags=["Delete"],
            dependencies=[Depends(require_api_key)])
def delete_child(child_id: str):
    """
    Hard-delete ALL data for a child. GDPR right-to-erasure.
    This is irreversible. History entries are also removed.
    """
    if not db.child_exists(child_id):
        raise HTTPException(404, f"Child '{child_id}' not found.")
    db.delete_child(child_id)
    return {
        "status": "deleted",
        "data": {"child_id": child_id, "message": "All data permanently erased."}
    }


# ──────────────────────────────────────────────────────────────────────────────
# EXPORT — full session data as JSON (for research analysis)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/um/", tags=["Read"])
def list_children():
    """Return all child IDs currently in the knowledge graph."""
    result = db.sparql_query(f"""
    PREFIX um: <{config.UM_PREFIX}>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?childId WHERE {{
        ?child rdf:type um:Child ;
               um:childId ?childId .
    }} ORDER BY ?childId
    """)
    ids = [row["childId"]["value"] for row in result["results"]["bindings"]]
    return {"status": "ok", "data": {"children": ids, "count": len(ids)}}


@app.get("/api/um/{child_id}/export", tags=["Export"])
def export_session(child_id: str):
    """
    Export the complete user model and change history for a child as JSON.

    Called at session end to produce the research data file described in the
    architecture diagram: every CRUD operation with timestamp, field,
    old_val, new_val, and source.

    The response can be saved directly to a .json file.
    Also callable by the SIC bridge after sync_to_graphdb().
    """
    if not db.child_exists(child_id):
        raise HTTPException(404, f"Child '{child_id}' not found.")

    profile = db.get_child_profile(child_id)
    history = db.get_history(child_id)

    # Build a flat summary of current field values for quick analysis
    # (avoids having to unpack the nested scalars/nodes structure)
    flat_current: dict[str, str] = {}
    for field, meta in profile["scalars"].items():
        if field not in ("childId", "createdAt", "createdBySession"):
            flat_current[field] = meta["value"]
    for field, entries in profile["nodes"].items():
        values = [e["value"] for e in entries if e.get("value")]
        if values:
            flat_current[field] = ", ".join(values)

    # Count how many fields were changed at least once during the session
    changed_fields = {h["field"] for h in history}

    return {
        "export_version": "1.0",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "child_id": child_id,
        "summary": {
            "total_fields_populated": len(flat_current),
            "total_fields_changed": len(changed_fields),
            "total_change_events": len(history),
        },
        "current_profile": flat_current,
        "full_profile": profile,          # includes scalars + nodes + provenance
        "change_log": history,            # all HistoryEntry records, newest first
    }




# ──────────────────────────────────────────────────────────────────────────────
# VALIDATE (dry-run only — no write)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/validate", tags=["Validate"])
def validate_only(update: FieldUpdate):
    """
    Validate fields without writing to GraphDB.
    Useful for the GUI to pre-check user input before submitting.
    """
    update.dry_run = True
    results = {"valid": [], "invalid": []}

    for field_name, value in update.fields.items():
        if field_name not in VALID_FIELDS:
            results["invalid"].append({
                "field": field_name,
                "errors": [f"Unknown field '{field_name}'"]
            })
            continue
        res = validate_field(field_name, value)
        if res.passed:
            results["valid"].append({"field": field_name, "value": str(value)})
        else:
            results["invalid"].append({
                "field": field_name,
                "value": str(value),
                "flag_type": res.flag_type,
                "errors": res.errors,
            })

    return {
        "status": "ok",
        "data": results,
        "all_valid": len(results["invalid"]) == 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  Open-Memory Robots — UM Memory Service v0.2.0")
    print("=" * 60)
    print(f"  GraphDB : {config.GRAPHDB_URL}")
    print(f"  Repo    : {config.GRAPHDB_REPOSITORY}")
    print(f"  Ollama  : {config.OLLAMA_URL} (model: {config.OLLAMA_MODEL})")
    print(f"  Docs    : http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)