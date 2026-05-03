from __future__ import annotations

import argparse

import torch
from tqdm.auto import tqdm

from kv_reasoning.data import build_examples, load_records
from kv_reasoning.eval import classification_accuracy, inductive_meteor
from kv_reasoning.models import load_causal_lm
from kv_reasoning.steering import generate_batch
from kv_reasoning.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run steering evaluation.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--task", choices=["deductive", "inductive", "abductive"], required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--coeff", type=float, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_vector(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    return payload["vector"] if isinstance(payload, dict) and "vector" in payload else payload


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", 42))

    records = load_records(args.data_path, dataset_name=args.dataset_name, split=args.split)
    if args.limit:
        records = records[: args.limit]
    examples = build_examples(records, args.task, "negative")

    model, tokenizer = load_causal_lm(**cfg["model"])
    vector = load_vector(args.vector)
    gen_cfg = cfg["generation"]
    batch_size = gen_cfg["batch_size"]

    predictions = []
    labels = []
    for start in tqdm(range(0, len(examples), batch_size), desc="Generating"):
        batch = examples[start : start + batch_size]
        outputs = generate_batch(
            model,
            tokenizer,
            [item["prompt"] for item in batch],
            max_new_tokens=gen_cfg["max_new_tokens"],
            steering_vector=vector,
            layer_idx=gen_cfg["layer_idx"],
            coeff=args.coeff,
            steering_mode=gen_cfg["steering_mode"],
        )
        predictions.extend(outputs)
        labels.extend(item["label"] for item in batch)

    if args.task == "inductive":
        result = inductive_meteor(predictions, labels)
        print(f"METEOR: {result['meteor']:.4f} ({result['total']} examples)")
    else:
        result = classification_accuracy(predictions, labels, args.task)
        print(f"Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']} valid predictions)")


if __name__ == "__main__":
    main()
