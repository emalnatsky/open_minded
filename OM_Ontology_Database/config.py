"""
All configuration constants for the UM service.
Change values here — never hardcode them in other files.
"""

# ── GraphDB ──────────────────────────────────────────────────────────────────
GRAPHDB_URL        = "http://localhost:7200"
GRAPHDB_REPOSITORY = "open-memory-robots"
SPARQL_QUERY_URL   = f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPOSITORY}"
SPARQL_UPDATE_URL  = f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPOSITORY}/statements"

# ── RDF namespaces ───────────────────────────────────────────────────────────
UM_PREFIX      = "http://example.org/open-memory-robots/"
RDF_PREFIX     = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XSD_PREFIX     = "http://www.w3.org/2001/XMLSchema#"
SHACL_PREFIX   = "http://www.w3.org/ns/shacl#"

# ── Ollama (local LLM) ───────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "mistral"          # or "llama3.1:8b" — change to what you have pulled

# Set True during development if Ollama is not running yet.
# Writes will still go through SHACL, but semantic LLM check is skipped.
SKIP_LLM_VALIDATION = True

# ── Ontology files (produced in Protégé, or hand-written) ────────────────────
SHACL_SHAPES_PATH  = "ontology/um_shapes_test.ttl"
OWL_ONTOLOGY_PATH  = "ontology/um_schema.ttl"

# ── API security ─────────────────────────────────────────────────────────────
# Following Gheorghe (2025): protect endpoints with an API key header.
# Consumers (GUI, robot bridge) must include: X-API-Key: <this value>
API_KEY = "change-me-before-deployment"

# Set True to disable API key checking during local development.
SKIP_API_KEY_CHECK = True