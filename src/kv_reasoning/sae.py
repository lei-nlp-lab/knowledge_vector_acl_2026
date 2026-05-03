from __future__ import annotations

import torch


def select_reasoning_features(
    positive_latents: torch.Tensor,
    negative_latents: torch.Tensor,
    quantile: float = 0.9,
    top_k: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Select SAE features that are stronger in successful generations."""

    pos_strength = positive_latents.float().pow(2).mean(dim=0)
    neg_strength = negative_latents.float().pow(2).mean(dim=0)
    ratio = pos_strength / (neg_strength + eps)
    threshold = torch.quantile(ratio, quantile)
    candidates = torch.nonzero(ratio >= threshold, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return candidates
    k = min(top_k, candidates.numel())
    ranked = torch.topk(pos_strength[candidates], k=k).indices
    return candidates[ranked]


def build_decoder_subspace(decoder_weight: torch.Tensor, feature_ids: torch.Tensor) -> torch.Tensor:
    """Build an orthonormal QR basis from selected SAE decoder directions."""

    max_feature = int(feature_ids.max().item()) if feature_ids.numel() else -1
    if max_feature < decoder_weight.shape[0]:
        directions = decoder_weight[feature_ids].float().T
    elif max_feature < decoder_weight.shape[1]:
        directions = decoder_weight[:, feature_ids].float()
    else:
        raise ValueError("feature_ids do not match either axis of decoder_weight")
    q, _ = torch.linalg.qr(directions, mode="reduced")
    return q
