from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template


class MessageBuilder:
    def __init__(self, prompt_template: str, system_prompt: str | None = None):
        self.template = Template(prompt_template, undefined=StrictUndefined)
        self.system_prompt = system_prompt

    def build_messages(self, input_data: dict[str, Any]) -> list[dict[str, Any]]:
        text_vars = {k: v for k, v in input_data.items() if k != "images"}
        rendered_prompt = self.template.render(**text_vars)
        images = input_data.get("images") or []

        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": rendered_prompt}]
            for image in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._normalize_image_ref(image)},
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": rendered_prompt})

        return messages

    def _normalize_image_ref(self, image: str) -> str:
        if not isinstance(image, str):
            raise TypeError("images must be a list[str]")

        lower = image.lower()
        if lower.startswith("http://") or lower.startswith("https://"):
            return image
        if lower.startswith("data:image/"):
            return image

        path = Path(image)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image}")

        mime, _ = mimetypes.guess_type(path.as_posix())
        mime = mime or "application/octet-stream"

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
