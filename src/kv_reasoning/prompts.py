from __future__ import annotations

from typing import Any


DEDUCTIVE_POSITIVE = ""
DEDUCTIVE_NEGATIVE = ""
INDUCTIVE_POSITIVE = ""
INDUCTIVE_NEGATIVE = ""
ABDUCTIVE_POSITIVE = ""
ABDUCTIVE_NEGATIVE = ""


def chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def format_prompt(record: dict[str, Any], task: str, variant: str) -> str:
    if variant not in {"positive", "negative"}:
        raise ValueError("variant must be 'positive' or 'negative'")

    if task == "deductive":
        template = DEDUCTIVE_POSITIVE if variant == "positive" else DEDUCTIVE_NEGATIVE
        if not template:
            return ""
        return template.format(
            paragraph=str(record["paragraph"]),
            statement=str(record.get("statement", record.get("question", ""))),
        )

    if task == "inductive":
        template = INDUCTIVE_POSITIVE if variant == "positive" else INDUCTIVE_NEGATIVE
        if not template:
            return ""
        facts = record.get("fact")
        if facts is None:
            fact_columns = [
                "short fact 1.1",
                "short fact 1.2",
                "short fact 2.1",
                "short fact 2.2",
                "short fact 3.1",
                "short fact 3.2",
            ]
            facts = "\n".join(str(record.get(col, "")).strip() for col in fact_columns if str(record.get(col, "")).strip())
        return template.format(fact=facts, template=str(record.get("rule template", record.get("template", ""))))

    if task == "abductive":
        template = ABDUCTIVE_POSITIVE if variant == "positive" else ABDUCTIVE_NEGATIVE
        if not template:
            return ""
        return template.format(
            obs1=str(record["observation_1"]),
            obs2=str(record["observation_2"]),
            hyp1=str(record["hypothesis_1"]),
            hyp2=str(record["hypothesis_2"]),
        )

    raise ValueError(f"Unknown task: {task}")
