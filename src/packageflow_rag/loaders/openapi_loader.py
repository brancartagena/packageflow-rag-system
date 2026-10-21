"""OpenAPI loader: packageflow-api-v08-wip.yaml (65 operations, 101 schemas, 13 tags).

Chunks produced:
- API overview: info.description (markdown) split by its headings.
- API conventions: servers, auth, shared parameters/headers/error responses.
- one per tag: what the tag is for + list of its operations.
- one per operation: method, path, operationId, tag, summary, who can use it,
  description, parameters, request body fields, responses. $refs resolved one level.
- one per schema: description, fields, types, enums, required (split in parts if huge).

Metadata: api_method, api_path, api_tag, operation_id, roles, schema_name.

Uses ruamel.yaml (pure Python) instead of PyYAML, because PyYAML has no
prebuilt wheel for Python 3.15 yet and fails to install there.
"""

import re
from pathlib import Path

from ruamel.yaml import YAML

from ..chunking import approx_tokens, make_chunk, pack_units, slug
from ..models import Chunk
from .markdown_loader import markdown_to_chunks

HTTP_METHODS = ("get", "post", "put", "patch", "delete")
KNOWN_ROLES = [
    "Package Receiver",
    "Delivery Team Member",
    "Facilities Lead",
    "Restricted-Item Inspector",
    "Requester",
]
WHO_RE = re.compile(r"\*\*Who can use this:\*\*\s*([^\n]+)")
FIELD_DESC_MAX = 220  # chars of each field description kept in schema/body listings


def read_yaml(path: Path):
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))


def resolve_ref(spec: dict, ref: str) -> dict:
    """Resolve an internal '#/components/...' ref."""
    node = spec
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node


def ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def parse_roles(description: str) -> tuple[str | None, list[str]]:
    """Returns (raw 'who can use this' text, [canonical role names])."""
    m = WHO_RE.search(description or "")
    if not m:
        return None, []
    raw = m.group(1).strip()
    if raw.lower().startswith("any signed-in user"):
        return raw, list(KNOWN_ROLES)
    roles = [r for r in KNOWN_ROLES if r.lower() in raw.lower()]
    return raw, roles


def _short(text: str | None, limit: int = FIELD_DESC_MAX) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def type_str(spec: dict, s: dict) -> str:
    """Human readable type: 'string (date-time)', 'array of Assignment', 'PackageState'."""
    if not isinstance(s, dict):
        return "any"
    if "$ref" in s:
        return ref_name(s["$ref"])
    for combo in ("allOf", "oneOf", "anyOf"):
        if combo in s:
            parts = [type_str(spec, x) for x in s[combo]]
            parts = [p for p in parts if p and p != "object"] or ["object"]
            return parts[0] if combo == "allOf" and len(parts) == 1 else f"{combo}({', '.join(parts)})"
    t = s.get("type", "object")
    if t == "array":
        return f"array of {type_str(spec, s.get('items', {}))}"
    if s.get("format"):
        t = f"{t} ({s['format']})"
    if s.get("nullable"):
        t += ", nullable"
    return t


def _enum_of(spec: dict, s: dict) -> list:
    if not isinstance(s, dict):
        return []
    if "enum" in s:
        return s["enum"]
    if s.get("type") == "array":
        return _enum_of(spec, s.get("items", {}))
    return []


def collect_properties(spec: dict, schema: dict, depth: int = 0) -> tuple[list[tuple[str, dict, bool]], list[str]]:
    """Flatten properties of a schema, following allOf one level.

    Returns ([(name, prop_schema, required)], [names of included base schemas]).
    """
    props: list[tuple[str, dict, bool]] = []
    includes: list[str] = []
    if not isinstance(schema, dict):
        return props, includes
    if "$ref" in schema and depth == 0:
        includes.append(ref_name(schema["$ref"]))
        schema = resolve_ref(spec, schema["$ref"])
    for part in schema.get("allOf", []):
        if "$ref" in part:
            includes.append(ref_name(part["$ref"]))
            if depth < 2:
                sub, sub_inc = collect_properties(spec, resolve_ref(spec, part["$ref"]), depth + 1)
                props.extend(sub)
                includes.extend(sub_inc)
        else:
            sub, sub_inc = collect_properties(spec, part, depth + 1)
            props.extend(sub)
            includes.extend(sub_inc)
    required = set(schema.get("required", []))
    for name, p in (schema.get("properties") or {}).items():
        props.append((name, p, name in required))
    # de-dupe keeping the last definition (more specific schema wins)
    seen: dict[str, tuple[str, dict, bool]] = {}
    for name, p, req in props:
        seen[name] = (name, p, req or (seen.get(name, (None, None, False))[2]))
    return list(seen.values()), list(dict.fromkeys(includes))


def property_lines(spec: dict, schema: dict) -> list[str]:
    props, _ = collect_properties(spec, schema)
    lines = []
    for name, p, req in props:
        bits = [type_str(spec, p)]
        if req:
            bits.append("required")
        line = f"- {name} ({', '.join(bits)})"
        desc = p.get("description") if isinstance(p, dict) else None
        if not desc and isinstance(p, dict) and "$ref" in p:
            desc = resolve_ref(spec, p["$ref"]).get("description")
        if desc:
            line += f": {_short(desc)}"
        enum = _enum_of(spec, p)
        if enum:
            line += f" Allowed values: {', '.join(map(str, enum))}."
        lines.append(line)
    return lines


def _param_line(spec: dict, prm: dict) -> str:
    if "$ref" in prm:
        prm = resolve_ref(spec, prm["$ref"])
    bits = [prm.get("in", "?"), type_str(spec, prm.get("schema", {}))]
    if prm.get("required"):
        bits.append("required")
    line = f"- {prm.get('name')} ({', '.join(bits)})"
    if prm.get("description"):
        line += f": {_short(prm['description'])}"
    enum = _enum_of(spec, prm.get("schema", {}))
    if enum:
        line += f" Allowed values: {', '.join(map(str, enum))}."
    return line


def _json_schema(content: dict | None) -> dict | None:
    if not content:
        return None
    for ctype, body in content.items():
        if "json" in ctype or ctype.startswith("multipart") or ctype.startswith("image"):
            return body.get("schema") if isinstance(body, dict) else None
    first = next(iter(content.values()), None)
    return first.get("schema") if isinstance(first, dict) else None


def operation_text(spec: dict, method: str, path: str, op: dict, path_params: list) -> str:
    lines = [f"API operation: {method.upper()} {path}"]
    if op.get("operationId"):
        lines.append(f"operationId: {op['operationId']}")
    if op.get("tags"):
        lines.append(f"Tag: {', '.join(op['tags'])}")
    if op.get("summary"):
        lines.append(f"Summary: {op['summary']}")
    who, _ = parse_roles(op.get("description", ""))
    if who:
        lines.append(f"Who can use this: {who}")
    desc = WHO_RE.sub("", op.get("description", "") or "").strip()
    if desc:
        lines.append(f"Description: {desc}")

    params = [*path_params, *op.get("parameters", [])]
    if params:
        lines.append("Parameters:")
        lines.extend(_param_line(spec, p) for p in params)

    rb = op.get("requestBody")
    if rb:
        if "$ref" in rb:
            rb = resolve_ref(spec, rb["$ref"])
        schema = _json_schema(rb.get("content"))
        name = type_str(spec, schema) if schema else "body"
        lines.append(f"Request body ({name}{', required' if rb.get('required') else ''}):")
        if rb.get("description"):
            lines.append(_short(rb["description"], 400))
        if schema:
            lines.extend(property_lines(spec, schema) or [f"- see schema {name}"])

    responses = op.get("responses", {})
    if responses:
        lines.append("Responses:")
        for code, resp in responses.items():
            ref = None
            if "$ref" in resp:
                ref = ref_name(resp["$ref"])
                resp = resolve_ref(spec, resp["$ref"])
            schema = _json_schema(resp.get("content"))
            line = f"- {code}: {_short(resp.get('description', ''), 300)}"
            if schema:
                line += f" Returns {type_str(spec, schema)}."
            if ref:
                line += f" ({ref})"
            lines.append(line)
    return "\n".join(lines)


def schema_text(spec: dict, name: str, schema: dict) -> tuple[str, list[str]]:
    """Returns (head, property lines)."""
    head = [f"API schema: {name}"]
    if schema.get("description"):
        head.append(f"Description: {' '.join(schema['description'].split())}")
    if "enum" in schema:
        head.append(f"Type: {schema.get('type', 'string')} enum")
        head.append(f"Allowed values: {', '.join(map(str, schema['enum']))}")
        return "\n".join(head), []
    for combo in ("oneOf", "anyOf"):
        if combo in schema:
            opts = [type_str(spec, x) for x in schema[combo]]
            head.append(f"One of: {', '.join(opts)}")
    _, includes = collect_properties(spec, schema)
    if includes:
        head.append(f"Includes all fields of: {', '.join(includes)}")
    if schema.get("type") == "array":
        head.append(f"Type: {type_str(spec, schema)}")
    lines = property_lines(spec, schema)
    if lines:
        head.append("Fields:")
    return "\n".join(head), lines


def load(path: Path, info, max_tokens: int = 400, overlap_tokens: int = 50) -> list[Chunk]:
    spec = read_yaml(path)
    api_info = spec.get("info", {})
    version = str(api_info.get("version")) if api_info.get("version") else None
    if version and not version.lower().startswith("v"):
        version = f"v{version}"
    label = f"{info.label} {version}" if version else info.label
    common = dict(source_file=path.name, doc_type=info.doc_type, doc_label=label, doc_version=version)
    chunks: list[Chunk] = []

    # 1. Overview (info.description is markdown)
    title = api_info.get("title", "API")
    intro = f"# {title}\n\n{api_info.get('description', '')}"
    chunks += markdown_to_chunks(intro, id_prefix="api:overview", common=common, max_tokens=max_tokens,
                                 overlap_tokens=overlap_tokens, section_prefix=["Overview"],
                                 extra_metadata={"api_kind": "overview"})

    # 2. Conventions: servers, auth, shared params / headers / error responses
    comp = spec.get("components", {})
    conv: list[str] = []
    if spec.get("servers"):
        conv.append("Servers (environments):\n" + "\n".join(
            f"- {s.get('description', '')}: {s.get('url')}" for s in spec["servers"]))
    for name, sch in comp.get("securitySchemes", {}).items():
        bits = [f"Authentication scheme '{name}': type {sch.get('type')}"]
        if sch.get("description"):
            bits.append(_short(sch["description"], 600))
        flows = sch.get("flows") or {}
        for fname, flow in flows.items():
            scopes = flow.get("scopes") or {}
            if scopes:
                bits.append(f"{fname} scopes: " + "; ".join(f"{k}: {v}" for k, v in scopes.items()))
        conv.append("\n".join(bits))
    if spec.get("security"):
        conv.append("Default security requirement: " + ", ".join(
            f"{k} ({', '.join(v)})" for req in spec["security"] for k, v in req.items()))
    if comp.get("parameters"):
        conv.append("Shared parameters and headers:\n" + "\n".join(
            _param_line(spec, p) for p in comp["parameters"].values()))
    if comp.get("headers"):
        conv.append("Shared response headers:\n" + "\n".join(
            f"- {name}: {_short(h.get('description', ''), 300)}" for name, h in comp["headers"].items()))
    if comp.get("responses"):
        conv.append("Shared error responses:\n" + "\n".join(
            f"- {name}: {_short(r.get('description', ''), 300)}" for name, r in comp["responses"].items()))
    for n, body in enumerate(pack_units(conv, max_tokens, 0), 1):
        chunks.append(make_chunk(chunk_id=f"api:conventions:{n}", body=body,
                                 section_path=["Conventions"], metadata={"api_kind": "conventions"}, **common))

    # 3. Operations (+ collect per tag)
    tag_ops: dict[str, list[str]] = {}
    for p, item in spec.get("paths", {}).items():
        path_params = item.get("parameters", [])
        for method in HTTP_METHODS:
            op = item.get(method)
            if not op:
                continue
            tag = (op.get("tags") or ["Untagged"])[0]
            who, roles = parse_roles(op.get("description", ""))
            tag_ops.setdefault(tag, []).append(
                f"- {method.upper()} {p}: {op.get('summary', '')}" + (f" (Who: {who})" if who else ""))
            text = operation_text(spec, method, p, op, path_params)
            meta = {"api_kind": "operation", "api_method": method.upper(), "api_path": p, "api_tag": tag,
                    "operation_id": op.get("operationId"), "roles": roles}
            parts = [text] if approx_tokens(text) <= max_tokens * 3 else pack_units(text.split("\n"), max_tokens * 2)
            for k, part in enumerate(parts, 1):
                cid = f"api:op:{method.upper()} {p}" + (f":part{k}" if k > 1 else "")
                body = part if k == 1 else f"API operation: {method.upper()} {p} (continued)\n{part}"
                chunks.append(make_chunk(chunk_id=cid, body=body, section_path=[tag, f"{method.upper()} {p}"],
                                         metadata=meta, **common))

    # 4. Tags
    tag_desc = {t["name"]: t.get("description", "") for t in spec.get("tags", [])}
    for tag in list(tag_desc) + [t for t in tag_ops if t not in tag_desc]:
        body = "\n".join([f"API tag (area): {tag}", f"Description: {tag_desc.get(tag, '')}",
                          f"Operations ({len(tag_ops.get(tag, []))}):", *tag_ops.get(tag, [])])
        for k, part in enumerate(pack_units(body.split("\n"), max_tokens * 2), 1):
            chunks.append(make_chunk(chunk_id=f"api:tag:{slug(tag)}" + (f":part{k}" if k > 1 else ""),
                                     body=part, section_path=["Tags", tag],
                                     metadata={"api_kind": "tag", "api_tag": tag}, **common))

    # 5. Schemas
    for name, schema in comp.get("schemas", {}).items():
        head, lines = schema_text(spec, name, schema)
        parts = pack_units(lines, max_tokens * 2 - approx_tokens(head)) if lines else [""]
        for k, part in enumerate(parts, 1):
            body = head if not part else f"{head}\n{part}"
            if k > 1:
                body = f"API schema: {name} (fields continued)\n{part}"
            chunks.append(make_chunk(chunk_id=f"api:schema:{name}" + (f":part{k}" if k > 1 else ""),
                                     body=body, section_path=["Schemas", name],
                                     metadata={"api_kind": "schema", "schema_name": name}, **common))
    return chunks
