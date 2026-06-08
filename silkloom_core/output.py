from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from .json_utils import extract_reasoning, parse_json_payload

T = TypeVar("T")


class OutputParser(Generic[T]):
    def __init__(self, schema: type[BaseModel] | type[dict] | None = None, *, repair_json: bool = True):
        self.schema = schema
        self.repair_json = repair_json

    def parse(self, text: str) -> tuple[T, str | None]:
        cleaned, reasoning = extract_reasoning(text)

        if self.schema is None:
            return text, reasoning  # type: ignore[return-value]

        parsed = parse_json_payload(cleaned, auto_repair_json=self.repair_json)
        if self.schema is dict:
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object.")
            return parsed, reasoning  # type: ignore[return-value]

        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object for the Pydantic schema.")
            return self.schema.model_validate(parsed), reasoning  # type: ignore[return-value]

        raise TypeError("output must be None, dict, or a pydantic BaseModel subclass.")

    def fingerprint(self) -> Any:
        if self.schema is None:
            return None
        if self.schema is dict:
            return "dict"
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            fields = getattr(self.schema, "model_fields", {})
            return {
                "pydantic": f"{self.schema.__module__}.{self.schema.__qualname__}",
                "fields": sorted((name, str(field.annotation)) for name, field in fields.items()),
            }
        return repr(self.schema)
