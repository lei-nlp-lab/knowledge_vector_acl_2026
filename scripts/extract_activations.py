from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from kv_reasoning.activations import generate_with_average_activations
from kv_reasoning.data import build_examples, load_records
from kv_reasoning.eval import classification_accuracy, inductive_meteor
from kv_reasoning.models import load_causal_lm
from kv_reasoning.utils import ensure_parent, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract contrastive activations.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--task", choices=["deductive", "inductive", "abductive"], required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default=None, help="Optional Hugging Face dataset name, e.g. allenai/art.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--inductive-positive-threshold", type=float, default=0.3)
    parser.add_argument("--inductive-negative-threshold", type=float, default=0.1)
    return parser.parse_args()


def is_success(text: str, label: str, task: str, positive_threshold: float, negative_threshold: float, positive_side: bool) -> bool:
    if task in {"deductive", "abductive"}:
        result = classification_accuracy([text], [label], task)
        return result["correct"] == 1

    score = inductive_meteor([text], [label])["meteor"]
    threshold = positive_threshold if positive_side else negative_threshold
    return score >= threshold


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", 42))

    records = load_records(args.data_path, dataset_name=args.dataset_name, split=args.split)
    if args.limit:
        records = records[: args.limit]

    positive = build_examples(records, args.task, "positive")
    negative = build_examples(records, args.task, "negative")

    model, tokenizer = load_causal_lm(**cfg["model"])
    gen_cfg = cfg["generation"]
    batch_size = gen_cfg["batch_size"]
    layer_idx = gen_cfg["layer_idx"]

    positive_acts = []
    negative_acts = []
    kept = []

    for start in tqdm(range(0, len(records), batch_size), desc="Extracting"):
        pos_batch = positive[start : start + batch_size]
        neg_batch = negative[start : start + batch_size]

        pos_texts, pos_avg = generate_with_average_activations(
            model,
            tokenizer,
            [item["prompt"] for item in pos_batch],
            layer_idx=layer_idx,
            max_new_tokens=gen_cfg["max_new_tokens"],
        )
        neg_texts, neg_avg = generate_with_average_activations(
            model,
            tokenizer,
            [item["prompt"] for item in neg_batch],
            layer_idx=layer_idx,
            max_new_tokens=gen_cfg["max_new_tokens"],
        )

        for offset, (pos_text, neg_text, pos_item, neg_item) in enumerate(zip(pos_texts, neg_texts, pos_batch, neg_batch)):
            pos_ok = is_success(
                pos_text,
                pos_item["label"],
                args.task,
                args.inductive_positive_threshold,
                args.inductive_negative_threshold,
                positive_side=True,
            )
            neg_ok = is_success(
                neg_text,
                neg_item["label"],
                args.task,
                args.inductive_positive_threshold,
                args.inductive_negative_threshold,
                positive_side=False,
            )
            if pos_ok and not neg_ok:
                positive_acts.append(pos_avg[offset])
                negative_acts.append(neg_avg[offset])
                kept.append({"index": start + offset, "label": pos_item["label"]})

    if not positive_acts:
        raise RuntimeError("No valid contrastive pairs were found. Check prompts, labels, and thresholds.")

    payload = {
        "task": args.task,
        "layer_idx": layer_idx,
        "positive": torch.stack(positive_acts),
        "negative": torch.stack(negative_acts),
        "metadata": kept,
    }
    ensure_parent(args.output)
    torch.save(payload, args.output)
    print(f"Saved {len(kept)} contrastive pairs to {Path(args.output)}")


if __name__ == "__main__":
    main()
