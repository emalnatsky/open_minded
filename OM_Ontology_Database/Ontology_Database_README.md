# Quick instruction (Draft version 1) (In Progress)

For this to work:
- 



## What is still not yet implemented:

In config:
- None of the validations are yet working. Keep `SKIP_LLM_VALIDATION = True` since it is not yet implemented. And SHACL just gives an error but will not stop anything from working. No need to have anything with Ollama yet.
- Keep `SKIP_API_KEY_CHECK = True` during testing so that it doesn't ask for verification key to work with the endpoints. 
- There is no formal ontology yet, but an empty GraphDB repository works fine as long as it is connected and has the same name as the one stated for `GRAPHDB_REPOSITORY`
