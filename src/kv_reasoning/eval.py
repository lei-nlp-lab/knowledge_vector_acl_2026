from __future__ import annotations

import re
from typing import Any


def parse_answer(text: str, task: str) -> str | None:
    text = text.strip().upper()
    if task == "deductive":
        matches = re.findall(r"\b(TRUE|FALSE|UNCERTAIN)\b", text)
        return matches[-1] if matches else None
    if task == "abductive":
        answer_line = re.findall(r"ANSWER\s*:\s*([AB])\b", text)
        if answer_line:
            return answer_line[-1]
        matches = re.findall(r"\b(A|B)\b", text)
        return matches[-1] if matches else None
    raise ValueError(f"parse_answer is only defined for classification tasks, got {task}")


def classification_accuracy(predictions: list[str], labels: list[str], task: str) -> dict[str, Any]:
    correct = 0
    total = 0
    for prediction, label in zip(predictions, labels):
        parsed = parse_answer(prediction, task)
        if parsed is None:
            continue
        total += 1
        correct += int(parsed == label.strip().upper())
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def inductive_meteor(predictions: list[str], references: list[str]) -> dict[str, Any]:
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for prediction, reference in zip(predictions, references):
        candidates = last_nonempty_lines(prediction, n=3) or [prediction]
        ref_tokens = reference.strip().split()
        best = max(meteor_score([ref_tokens], candidate.split()) for candidate in candidates)
        scores.append(best)
    return {"meteor": sum(scores) / len(scores) if scores else 0.0, "total": len(scores)}


def last_nonempty_lines(text: str, n: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-n:]

