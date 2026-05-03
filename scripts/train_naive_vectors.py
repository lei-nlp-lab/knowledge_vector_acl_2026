from __future__ import annotations

import argparse

import torch

from kv_reasoning.utils import ensure_parent, load_config, set_seed
from kv_reasoning.vectors import train_linear_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a naive reasoning vector from contrastive activations.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--activation-file", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", 42))
    payload = torch.load(args.activation_file, map_location="cpu")
    result = train_linear_probe(
        payload["positive"],
        payload["negative"],
        epochs=cfg["training"]["probe_epochs"],
        lr=cfg["training"]["lr"],
    )
    ensure_parent(args.output)
    torch.save({"vector": result.vector, "bias": result.bias, "losses": result.losses, "task": payload.get("task")}, args.output)
    print(f"Saved naive vector to {args.output}")


if __name__ == "__main__":
    main()
