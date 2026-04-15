from __future__ import annotations

import re
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][\w\.]+)\}")


def _get_by_path(context: dict[str, Any], dotted_path: str) -> Any:
    current: Any = context
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(f"Context path not found: {dotted_path}")
    return current


def render_template(template: str, context: dict[str, Any]) -> str:
    def _replace(match: re.Match[str]) -> str:
        value = _get_by_path(context, match.group(1))
        return "" if value is None else str(value)

    return _PLACEHOLDER_RE.sub(_replace, template)


def resolve_mapping(mapping: dict[str, str], context: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for arg_name, pattern in mapping.items():
        pure = _PLACEHOLDER_RE.fullmatch(pattern)
        if pure:
            resolved[arg_name] = _get_by_path(context, pure.group(1))
        else:
            resolved[arg_name] = render_template(pattern, context)
    return resolved
