"""A deliberately small JSON Schema subset validator.

The harness must run with no dependencies beyond `anthropic`, so this covers
the subset that structural assertions actually need: types, required keys,
properties, additionalProperties, items, enum/const, numeric and string bounds,
patterns, and the anyOf/allOf/oneOf combinators. Anything outside that raises
`SchemaError` at task-load time rather than silently passing at run time -- a
validator that quietly ignores a keyword is worse than no validator.
"""

from __future__ import annotations

import re
from typing import Any

SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "pattern",
        "anyOf",
        "allOf",
        "oneOf",
        "not",
        "description",
        "title",
        "$schema",
        "default",
        "examples",
    }
)

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


class SchemaError(Exception):
    """The schema itself is malformed or uses an unsupported keyword."""


def check_schema(schema: Any, path: str = "$") -> None:
    """Raise SchemaError if `schema` uses anything this validator ignores."""
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: schema must be an object")
    unknown = set(schema) - SUPPORTED_KEYWORDS
    if unknown:
        raise SchemaError(f"{path}: unsupported schema keyword(s): {sorted(unknown)}")
    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        for name in names:
            if name not in _TYPES:
                raise SchemaError(f"{path}: unknown type {name!r}")
    for key in ("properties",):
        sub = schema.get(key)
        if sub is not None:
            if not isinstance(sub, dict):
                raise SchemaError(f"{path}.{key}: must be an object")
            for name, value in sub.items():
                check_schema(value, f"{path}.{key}.{name}")
    if isinstance(schema.get("items"), dict):
        check_schema(schema["items"], f"{path}.items")
    if isinstance(schema.get("additionalProperties"), dict):
        check_schema(schema["additionalProperties"], f"{path}.additionalProperties")
    if isinstance(schema.get("not"), dict):
        check_schema(schema["not"], f"{path}.not")
    for key in ("anyOf", "allOf", "oneOf"):
        sub = schema.get(key)
        if sub is not None:
            if not isinstance(sub, list) or not sub:
                raise SchemaError(f"{path}.{key}: must be a non-empty array")
            for i, value in enumerate(sub):
                check_schema(value, f"{path}.{key}[{i}]")


def validate(instance: Any, schema: Any, path: str = "$") -> list[str]:
    """Return a list of human-readable validation errors (empty means valid)."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object"]

    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not any(_is_type(instance, name) for name in names):
            errors.append(f"{path}: expected type {'|'.join(names)}, got {_type_name(instance)}")
            return errors

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in enum {schema['enum']!r}")

    if isinstance(instance, str):
        errors += _validate_string(instance, schema, path)
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        errors += _validate_number(instance, schema, path)
    if isinstance(instance, list):
        errors += _validate_array(instance, schema, path)
    if isinstance(instance, dict):
        errors += _validate_object(instance, schema, path)

    for sub in schema.get("allOf", []):
        errors += validate(instance, sub, path)
    if "anyOf" in schema:
        if not any(not validate(instance, sub, path) for sub in schema["anyOf"]):
            errors.append(f"{path}: does not match any schema in anyOf")
    if "oneOf" in schema:
        matches = sum(1 for sub in schema["oneOf"] if not validate(instance, sub, path))
        if matches != 1:
            errors.append(f"{path}: matched {matches} schemas in oneOf, expected exactly 1")
    if "not" in schema and not validate(instance, schema["not"], path):
        errors.append(f"{path}: matched a schema it must not match")
    return errors


def _validate_string(instance: str, schema: dict, path: str) -> list[str]:
    errors = []
    if "minLength" in schema and len(instance) < schema["minLength"]:
        errors.append(f"{path}: shorter than minLength {schema['minLength']}")
    if "maxLength" in schema and len(instance) > schema["maxLength"]:
        errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
    pattern = schema.get("pattern")
    if pattern is not None and re.search(pattern, instance) is None:
        errors.append(f"{path}: does not match pattern {pattern!r}")
    return errors


def _validate_number(instance: float, schema: dict, path: str) -> list[str]:
    errors = []
    if "minimum" in schema and instance < schema["minimum"]:
        errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
    if "maximum" in schema and instance > schema["maximum"]:
        errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
    if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
        errors.append(f"{path}: {instance} <= exclusiveMinimum {schema['exclusiveMinimum']}")
    if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
        errors.append(f"{path}: {instance} >= exclusiveMaximum {schema['exclusiveMaximum']}")
    return errors


def _validate_array(instance: list, schema: dict, path: str) -> list[str]:
    errors = []
    if "minItems" in schema and len(instance) < schema["minItems"]:
        errors.append(f"{path}: fewer than minItems {schema['minItems']}")
    if "maxItems" in schema and len(instance) > schema["maxItems"]:
        errors.append(f"{path}: more than maxItems {schema['maxItems']}")
    if schema.get("uniqueItems") and len(instance) != len({repr(i) for i in instance}):
        errors.append(f"{path}: items are not unique")
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for i, item in enumerate(instance):
            errors += validate(item, item_schema, f"{path}[{i}]")
    return errors


def _validate_object(instance: dict, schema: dict, path: str) -> list[str]:
    errors = []
    properties = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in instance:
            errors.append(f"{path}: missing required property {key!r}")
    for key, value in instance.items():
        if key in properties:
            errors += validate(value, properties[key], f"{path}.{key}")
        else:
            extra = schema.get("additionalProperties", True)
            if extra is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(extra, dict):
                errors += validate(value, extra, f"{path}.{key}")
    return errors


def _is_type(instance: Any, name: str) -> bool:
    if name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if name == "boolean":
        return isinstance(instance, bool)
    expected = _TYPES.get(name)
    if expected is None:
        return False
    return isinstance(instance, expected)


def _type_name(instance: Any) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _is_type(instance, name):
            return name
    return type(instance).__name__
