from __future__ import annotations

import json
import re
from typing import Any

from json_repair import repair_json


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_reasoning(text: str) -> tuple[str, str | None]:
    chunks = _THINK_RE.findall(text)
    cleaned = _THINK_RE.sub("", text).strip()
    reasoning = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip()) or None
    return cleaned, reasoning


def parse_json_payload(text: str, auto_repair_json: bool = True) -> Any:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        raise ValueError("No JSON object or array found in model output")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        if not auto_repair_json:
            raise

    repaired = repair_json(candidate, return_objects=True)
    if repaired is None:
        raise ValueError("JSON repair failed")
    return repaired


def _extract_json_candidate(text: str) -> str | None:
    fenced = _FENCED_JSON_RE.findall(text)
    if fenced:
        # Prefer the longest fenced block as the most likely full payload.
        return max((block.strip() for block in fenced), key=len, default=None)

    for opening, closing in (("{", "}"), ("[", "]")):
        candidate = _extract_balanced(text, opening, closing)
        if candidate:
            return candidate
    return None


def _extract_balanced(text: str, opening: str, closing: str) -> str | None:
    start = text.find(opening)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None
