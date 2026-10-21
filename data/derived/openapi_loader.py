"""OpenAPI loader: packageflow-api-v08-wip.yaml (65 operations, 101 schemas, 13 tags).

Chunks produced:
- one per operation: method, path, operationId, tag, summary, description,
  parameters, request body fields, response codes. $refs resolved one level.
- one per schema: name, description, fields, types, enums, required.
- one overview per tag.
Metadata: api_method, api_path, api_tag, operation_id, roles
(roles parsed from the "**Who can use this:** ..." line in descriptions).
"""

from pathlib import Path

from ...src.packageflow_rag.models import Chunk


def resolve_ref(spec: dict, ref: str) -> dict:
    """Resolve an internal '#/components/...' ref."""
    node = spec
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node


def load(path: Path) -> list[Chunk]:
    raise NotImplementedError("Phase 1")
