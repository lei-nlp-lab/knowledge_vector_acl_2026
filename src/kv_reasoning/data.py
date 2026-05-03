from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .prompts import format_prompt
from .utils import read_jsonl


def load_records(path: str | Path | None = None, dataset_name: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
    if dataset_name:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name)
        selected_split = split or "validation"
        return [dict(row) for row in dataset[selected_split]]

    if path is None:
        raise ValueError("Either path or dataset_name must be provided.")

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        return pd.read_json(path).to_dict(orient="records")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path).to_dict(orient="records")
    raise ValueError(f"Unsupported data format: {path.suffix}")


def get_label(record: dict[str, Any], task: str) -> str:
    if task == "deductive":
        return str(record["label"]).strip().upper()
    if task == "inductive":
        return str(record.get("rule", record.get("label", ""))).strip()
    if task == "abductive":
        raw = record["label"]
        if str(raw).strip().upper() in {"A", "B"}:
            return str(raw).strip().upper()
        return "A" if int(raw) == 0 else "B"
    raise ValueError(f"Unknown task: {task}")


def build_examples(records: list[dict[str, Any]], task: str, variant: str) -> list[dict[str, Any]]:
    examples = []
    for record in records:
        examples.append(
            {
                "prompt": format_prompt(record, task=task, variant=variant),
                "label": get_label(record, task),
                "record": record,
            }
        )
    return examples

