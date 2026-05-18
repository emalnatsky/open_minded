"""
CRI Scenario service — GraphDB operations for interaction scenarios.

Separated from the main UM (graphdb_client.py) because scenarios are
experiment apparatus, not child data. The UM stores what the child said.
The scenario stores what Leo will say (and get wrong).

Node types:
    CRIScenario         — one per child, links to mistakes + utterances
    CRIMistake          — M1-M4, each with field, wrongValue, type
    CRIUtterance        — per step × branch, tagged by content layer
    CRIInteractionEvent — logged during/after CRI by the frontend

All URIs follow the pattern:
    um:scenario/{child_id}
    um:mistake/{child_id}/{mistake_id}
    um:utterance/{child_id}/{step_id}/{branch}
    um:event/{child_id}/{uuid}
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from config import UM_PREFIX, SPARQL_QUERY_URL, SPARQL_UPDATE_URL
from services.graphdb_client import (
    sparql_query, sparql_update, sparql_ask,
    _escape_sparql_string, _sanitize_for_uri, _child_uri,
)


_PREFIXES = f"""
PREFIX um:   <{UM_PREFIX}>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _scenario_uri(child_id: str) -> str:
    return f"{UM_PREFIX}scenario/{_sanitize_for_uri(child_id)}"

def _mistake_uri(child_id: str, mistake_id: str) -> str:
    return f"{UM_PREFIX}mistake/{_sanitize_for_uri(child_id)}/{_sanitize_for_uri(mistake_id)}"

def _utterance_uri(child_id: str, step_id: str, branch: str) -> str:
    return f"{UM_PREFIX}utterance/{_sanitize_for_uri(child_id)}/{_sanitize_for_uri(step_id)}/{_sanitize_for_uri(branch)}"

def _event_uri(child_id: str) -> str:
    uid = str(uuid.uuid4()).replace("-", "")[:12]
    return f"{UM_PREFIX}event/{_sanitize_for_uri(child_id)}/{uid}"


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO EXISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_exists(child_id: str) -> bool:
    uri = _scenario_uri(child_id)
    return sparql_ask(f"""
    {_PREFIXES}
    ASK {{ <{uri}> rdf:type um:CRIScenario }}
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# CREATE SCENARIO WITH MISTAKES
# ═══════════════════════════════════════════════════════════════════════════════

def create_scenario(
    child_id: str,
    mistakes: list[dict],
    version: str = "1.0",
) -> None:
    """
    Create a CRIScenario node for a child with mistake definitions.

    mistakes: list of dicts, each with:
        id:           "M1", "M2", etc.
        target_field: UM field name (e.g. "hobby_fav")
        wrong_value:  the wrong value Leo will state
        mistake_type: "related-but-wrong" | "completely-wrong"
        spt_level:    "orientation" | "exploratory" | "affective"
        step:         step number in the CRI flow

    Overwrites any existing scenario for this child.
    """
    # Delete existing scenario if present
    if scenario_exists(child_id):
        delete_scenario(child_id)

    child_uri = _child_uri(child_id)
    scen_uri  = _scenario_uri(child_id)
    now       = _now_iso()

    # Create scenario node
    triples = f"""
        <{scen_uri}> rdf:type um:CRIScenario ;
                     um:forChild "{_escape_sparql_string(child_id)}" ;
                     um:scenarioVersion "{_escape_sparql_string(version)}" ;
                     um:generatedAt "{now}"^^xsd:dateTime .
        <{child_uri}> um:hasScenario <{scen_uri}> .
    """

    # Create mistake nodes
    for m in mistakes:
        m_uri = _mistake_uri(child_id, m["id"])
        triples += f"""
        <{m_uri}> rdf:type um:CRIMistake ;
                  um:mistakeId "{_escape_sparql_string(m['id'])}" ;
                  um:targetField "{_escape_sparql_string(m['target_field'])}" ;
                  um:wrongValue "{_escape_sparql_string(m['wrong_value'])}" ;
                  um:mistakeType "{_escape_sparql_string(m['mistake_type'])}" ;
                  um:sptLevel "{_escape_sparql_string(m['spt_level'])}" ;
                  um:step {int(m['step'])} .
        <{scen_uri}> um:hasMistake <{m_uri}> .
        """

    sparql_update(f"{_PREFIXES} INSERT DATA {{ {triples} }}")


# ═══════════════════════════════════════════════════════════════════════════════
# UTTERANCES
# ═══════════════════════════════════════════════════════════════════════════════

def set_utterance(
    child_id: str,
    step_id: str,
    layer: str,
    branch: str,
    text: str,
) -> None:
    """
    Write or update a CRI utterance for a child.

    step_id: e.g. "hobby_bridge", "m1_wrong_statement", "m1_followup"
    layer:   "L2-slot" | "L2-pregen"
    branch:  "default" | "corrected" | "not_corrected"
    text:    the utterance text (stub or final)

    Idempotent: overwrites existing utterance for same step+branch.
    """
    scen_uri = _scenario_uri(child_id)
    utt_uri  = _utterance_uri(child_id, step_id, branch)

    # Delete existing utterance for this step+branch
    sparql_update(f"""
    {_PREFIXES}
    DELETE WHERE {{ <{utt_uri}> ?p ?o }};
    DELETE {{ <{scen_uri}> um:hasUtterance <{utt_uri}> }}
    WHERE  {{ <{scen_uri}> um:hasUtterance <{utt_uri}> }}
    """)

    # Insert new utterance
    sparql_update(f"""
    {_PREFIXES}
    INSERT DATA {{
        <{utt_uri}> rdf:type um:CRIUtterance ;
                    um:stepId "{_escape_sparql_string(step_id)}" ;
                    um:layer "{_escape_sparql_string(layer)}" ;
                    um:branch "{_escape_sparql_string(branch)}" ;
                    um:text "{_escape_sparql_string(text)}" .
        <{scen_uri}> um:hasUtterance <{utt_uri}> .
    }}
    """)


def set_utterances_batch(child_id: str, utterances: list[dict]) -> None:
    """
    Write multiple utterances in one go.
    Each dict: {step_id, layer, branch, text}
    """
    for utt in utterances:
        set_utterance(
            child_id,
            utt["step_id"],
            utt["layer"],
            utt["branch"],
            utt["text"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# READ SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════

def get_scenario(child_id: str) -> dict:
    """
    Return the complete CRI scenario for a child.

    Returns:
        {
            "version": "1.0",
            "generated_at": "...",
            "mistakes": [
                {"id": "M1", "target_field": "hobby_fav", "wrong_value": "bakken",
                 "mistake_type": "related-but-wrong", "spt_level": "orientation", "step": 4},
                ...
            ],
            "utterances": {
                "hobby_bridge": {"default": "Ik weet dat jij van ..."},
                "m1_followup": {
                    "corrected": "Wat vind je het leukst aan tekenen?",
                    "not_corrected": "Wat vind je het leukst aan bakken?"
                },
                ...
            }
        }
    """
    scen_uri = _scenario_uri(child_id)

    if not scenario_exists(child_id):
        raise HTTPException(404, f"No CRI scenario found for child '{child_id}'.")

    # Read scenario metadata
    meta_q = sparql_query(f"""
    {_PREFIXES}
    SELECT ?version ?generatedAt WHERE {{
        <{scen_uri}> um:scenarioVersion ?version ;
                     um:generatedAt ?generatedAt .
    }}
    """)
    meta_bindings = meta_q["results"]["bindings"]
    version = meta_bindings[0]["version"]["value"] if meta_bindings else "unknown"
    generated_at = meta_bindings[0]["generatedAt"]["value"] if meta_bindings else "unknown"

    # Read mistakes
    mistake_q = sparql_query(f"""
    {_PREFIXES}
    SELECT ?id ?field ?wrongVal ?mtype ?spt ?step WHERE {{
        <{scen_uri}> um:hasMistake ?m .
        ?m um:mistakeId ?id ;
           um:targetField ?field ;
           um:wrongValue ?wrongVal ;
           um:mistakeType ?mtype ;
           um:sptLevel ?spt ;
           um:step ?step .
    }}
    ORDER BY ?step
    """)
    mistakes = []
    for b in mistake_q["results"]["bindings"]:
        mistakes.append({
            "id": b["id"]["value"],
            "target_field": b["field"]["value"],
            "wrong_value": b["wrongVal"]["value"],
            "mistake_type": b["mtype"]["value"],
            "spt_level": b["spt"]["value"],
            "step": int(b["step"]["value"]),
        })

    # Read utterances
    utt_q = sparql_query(f"""
    {_PREFIXES}
    SELECT ?stepId ?layer ?branch ?text WHERE {{
        <{scen_uri}> um:hasUtterance ?u .
        ?u um:stepId ?stepId ;
           um:layer ?layer ;
           um:branch ?branch ;
           um:text ?text .
    }}
    ORDER BY ?stepId ?branch
    """)
    utterances: dict[str, dict[str, str]] = {}
    for b in utt_q["results"]["bindings"]:
        step = b["stepId"]["value"]
        branch = b["branch"]["value"]
        text = b["text"]["value"]
        utterances.setdefault(step, {})[branch] = text

    return {
        "version": version,
        "generated_at": generated_at,
        "mistakes": mistakes,
        "utterances": utterances,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

def log_interaction_event(
    child_id: str,
    event_type: str,
    mistake_id: str | None,
    field: str | None,
    wrong_value: str | None,
    child_response: str | None,
    corrected: bool,
    phase: str | None,
    step: int | None,
    session_id: str = "unknown",
) -> None:
    """
    Log a CRI interaction event (correction, no_correction, nudge, etc.).

    Called by the frontend during/after the CRI interaction.
    """
    scen_uri = _scenario_uri(child_id)
    evt_uri  = _event_uri(child_id)
    now      = _now_iso()

    triples = f"""
        <{evt_uri}> rdf:type um:CRIInteractionEvent ;
                    um:eventType "{_escape_sparql_string(event_type)}" ;
                    um:corrected "{str(corrected).lower()}"^^xsd:string ;
                    um:eventTimestamp "{now}"^^xsd:dateTime ;
                    um:sessionId "{_escape_sparql_string(session_id)}" .
    """
    if mistake_id:
        triples += f'    <{evt_uri}> um:mistakeId "{_escape_sparql_string(mistake_id)}" .\n'
    if field:
        triples += f'    <{evt_uri}> um:field "{_escape_sparql_string(field)}" .\n'
    if wrong_value:
        triples += f'    <{evt_uri}> um:wrongValue "{_escape_sparql_string(wrong_value)}" .\n'
    if child_response:
        triples += f'    <{evt_uri}> um:childResponse "{_escape_sparql_string(child_response)}" .\n'
    if phase:
        triples += f'    <{evt_uri}> um:phase "{_escape_sparql_string(phase)}" .\n'
    if step is not None:
        triples += f'    <{evt_uri}> um:step {int(step)} .\n'

    triples += f'    <{scen_uri}> um:hasEvent <{evt_uri}> .\n'

    sparql_update(f"{_PREFIXES} INSERT DATA {{ {triples} }}")


def get_interaction_events(child_id: str) -> list[dict]:
    """Return all CRI interaction events for a child, newest first."""
    scen_uri = _scenario_uri(child_id)
    r = sparql_query(f"""
    {_PREFIXES}
    SELECT ?eventType ?mistakeId ?field ?wrongValue ?childResponse
           ?corrected ?phase ?step ?timestamp ?sessionId
    WHERE {{
        <{scen_uri}> um:hasEvent ?evt .
        ?evt um:eventType ?eventType ;
             um:corrected ?corrected ;
             um:eventTimestamp ?timestamp .
        OPTIONAL {{ ?evt um:mistakeId ?mistakeId }}
        OPTIONAL {{ ?evt um:field ?field }}
        OPTIONAL {{ ?evt um:wrongValue ?wrongValue }}
        OPTIONAL {{ ?evt um:childResponse ?childResponse }}
        OPTIONAL {{ ?evt um:phase ?phase }}
        OPTIONAL {{ ?evt um:step ?step }}
        OPTIONAL {{ ?evt um:sessionId ?sessionId }}
    }}
    ORDER BY DESC(?timestamp)
    """)
    events = []
    for b in r["results"]["bindings"]:
        events.append({
            "event_type":     b["eventType"]["value"],
            "mistake_id":     b.get("mistakeId", {}).get("value"),
            "field":          b.get("field", {}).get("value"),
            "wrong_value":    b.get("wrongValue", {}).get("value"),
            "child_response": b.get("childResponse", {}).get("value"),
            "corrected":      b["corrected"]["value"] == "true",
            "phase":          b.get("phase", {}).get("value"),
            "step":           int(b["step"]["value"]) if "step" in b else None,
            "timestamp":      b["timestamp"]["value"],
            "session_id":     b.get("sessionId", {}).get("value", "unknown"),
        })
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# DELETE SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════

def delete_scenario(child_id: str) -> None:
    """Delete the entire CRI scenario for a child (mistakes + utterances + events)."""
    scen_uri  = _scenario_uri(child_id)
    child_uri = _child_uri(child_id)
    safe_id   = _escape_sparql_string(child_id)

    # Delete all mistake nodes
    sparql_update(f"""
    {_PREFIXES}
    DELETE WHERE {{
        <{scen_uri}> um:hasMistake ?m .
        ?m ?p ?o .
    }}
    """)

    # Delete all utterance nodes
    sparql_update(f"""
    {_PREFIXES}
    DELETE WHERE {{
        <{scen_uri}> um:hasUtterance ?u .
        ?u ?p ?o .
    }}
    """)

    # Delete all event nodes
    sparql_update(f"""
    {_PREFIXES}
    DELETE WHERE {{
        <{scen_uri}> um:hasEvent ?e .
        ?e ?p ?o .
    }}
    """)

    # Delete scenario node itself
    sparql_update(f"""
    {_PREFIXES}
    DELETE WHERE {{ <{scen_uri}> ?p ?o }};
    DELETE {{ <{child_uri}> um:hasScenario <{scen_uri}> }}
    WHERE  {{ <{child_uri}> um:hasScenario <{scen_uri}> }}
    """)