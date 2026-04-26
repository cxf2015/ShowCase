from __future__ import annotations

from typing import Any


def format_prompt(record: dict[str, Any], template: str = "alpaca") -> str:
    prompt = (record.get("prompt") or "").strip()
    if prompt:
        return prompt

    instruction = (record.get("instruction") or "").strip()
    input_text = (record.get("input") or "").strip()

    if template != "alpaca":
        raise ValueError(f"Unsupported prompt template: {template}")

    if not instruction and not input_text:
        raise ValueError("Record must contain either 'prompt' or 'instruction'/'input'.")

    sections = [
        "Below is an instruction that describes a task. Write a helpful response.",
        "",
        "### Instruction:",
        instruction or "Please answer the user's request.",
    ]

    if input_text:
        sections.extend(["", "### Input:", input_text])

    sections.extend(["", "### Response:"])
    return "\n".join(sections)
