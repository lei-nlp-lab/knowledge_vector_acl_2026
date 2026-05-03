from __future__ import annotations

import argparse

import torch

from kv_reasoning.sae import build_decoder_subspace, select_reasoning_features
from kv_reasoning.utils import ensure_parent, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a QR reasoning subspace from saved SAE latents and decoder weights.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--positive-latents", required=True)
    parser.add_argument("--negative-latents", required=True)
    parser.add_argument("--decoder-weight", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    sae_cfg = cfg["sae"]

    positive = torch.load(args.positive_latents, map_location="cpu")
    negative = torch.load(args.negative_latents, map_location="cpu")
    decoder = torch.load(args.decoder_weight, map_location="cpu")

    feature_ids = select_reasoning_features(
        positive,
        negative,
        quantile=sae_cfg["quantile"],
        top_k=sae_cfg["top_k"],
        eps=sae_cfg["eps"],
    )
    basis = build_decoder_subspace(decoder, feature_ids)
    ensure_parent(args.output)
    torch.save(basis, args.output)
    print(f"Saved subspace with shape {tuple(basis.shape)} to {args.output}")


if __name__ == "__main__":
    main()
