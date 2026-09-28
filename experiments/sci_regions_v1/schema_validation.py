"""Small offline validator for the Draft-07 keywords used by these two schemas.

The JSON Schema files are the contracts. This deliberately rejects unknown schema
keywords instead of silently accepting a contract it cannot enforce.
"""
from __future__ import annotations

import json
import re
from typing import Any

SUPPORTED = {"$schema", "$id", "title", "type", "additionalProperties", "required", "properties", "definitions", "$ref", "const", "enum", "minItems", "maxItems", "uniqueItems", "items", "minLength", "pattern", "minimum", "exclusiveMinimum", "oneOf"}


class SchemaError(ValueError):
    pass


def _type_ok(value: Any, kind: str) -> bool:
    return {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }[kind]()


def validate(value: Any, schema: dict, *, root: dict | None = None, path: str = "$", depth: int = 0) -> None:
    if depth > 100:
        raise SchemaError(f"{path}: schema recursion limit")
    root = root or schema
    unknown = set(schema) - SUPPORTED
    if unknown:
        raise SchemaError(f"{path}: unsupported schema keywords {sorted(unknown)}")
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise SchemaError(f"{path}: only local references supported")
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part]
        validate(value, target, root=root, path=path, depth=depth + 1)
        return
    if "oneOf" in schema:
        passed = 0
        for option in schema["oneOf"]:
            try:
                validate(value, option, root=root, path=path, depth=depth + 1)
                passed += 1
            except SchemaError:
                pass
        if passed != 1:
            raise SchemaError(f"{path}: expected exactly one variant, got {passed}")
    kinds = schema.get("type")
    if kinds is not None:
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if not any(_type_ok(value, kind) for kind in kinds):
            raise SchemaError(f"{path}: expected {kinds}")
    if "const" in schema and value != schema["const"]:
        raise SchemaError(f"{path}: wrong const")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: unknown enum")
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise SchemaError(f"{path}: missing {sorted(missing)}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            spec = properties.get(key, schema.get("additionalProperties", True))
            if spec is False:
                raise SchemaError(f"{path}.{key}: unknown field")
            if isinstance(spec, dict):
                validate(item, spec, root=root, path=f"{path}.{key}", depth=depth + 1)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            raise SchemaError(f"{path}: wrong array length")
        if schema.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            raise SchemaError(f"{path}: duplicate array element")
        if "items" in schema:
            for i, item in enumerate(value):
                validate(item, schema["items"], root=root, path=f"{path}[{i}]", depth=depth + 1)
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise SchemaError(f"{path}: too short")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise SchemaError(f"{path}: pattern mismatch")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaError(f"{path}: below minimum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise SchemaError(f"{path}: below exclusive minimum")
