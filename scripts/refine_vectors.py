from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kv_reasoning.utils import load_config, set_seed
from kv_reasoning.vectors import refine_vectors


TASKS = ("deductive", "inductive", "abductive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine reasoning vectors with complementary and subspace losses.")
    parser.add_argument("--config", default="configs/config.yaml")
    for task in TASKS:
        parser.add_argument(f"--{task}-activations", required=True)
        parser.add_argument(f"--{task}-vector", default=None)
        parser.add_argument(f"--{task}-subspace", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_vector(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    return payload["vector"] if isinstance(payload, dict) and "vector" in payload else payload


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", 42))

    activations = {}
    initial_vectors = {}
    subspaces = {}
    for task in TASKS:
        payload = torch.load(getattr(args, f"{task}_activations"), map_location="cpu")
        activations[task] = (payload["positive"], payload["negative"])

        vector = load_vector(getattr(args, f"{task}_vector"))
        if vector is not None:
            initial_vectors[task] = vector

        subspace_path = getattr(args, f"{task}_subspace")
        if subspace_path:
            subspaces[task] = torch.load(subspace_path, map_location="cpu")

    refined = refine_vectors(
        activations,
        initial_vectors=initial_vectors or None,
        subspaces=subspaces or None,
        epochs=cfg["training"]["refine_epochs"],
        lr=cfg["training"]["lr"],
        lambda_com=cfg["training"]["lambda_com"],
        lambda_sub=cfg["training"]["lambda_sub"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, vector in refined.items():
        path = output_dir / f"{task}_refined_vector.pt"
        torch.save(vector, path)
        print(f"Saved {task} refined vector to {path}")


if __name__ == "__main__":
    main()
