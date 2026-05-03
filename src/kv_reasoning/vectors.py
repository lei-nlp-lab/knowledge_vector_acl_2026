from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ProbeResult:
    vector: torch.Tensor
    bias: torch.Tensor
    losses: list[float]


def make_probe_data(positive: torch.Tensor, negative: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([positive.float(), negative.float()], dim=0)
    y = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))], dim=0)
    return x, y


def train_linear_probe(
    positive: torch.Tensor,
    negative: torch.Tensor,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cuda",
) -> ProbeResult:
    x, y = make_probe_data(positive, negative)
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    x = x.to(device)
    y = y.to(device)

    vector = torch.zeros(x.shape[1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([vector, bias], lr=lr, weight_decay=weight_decay)
    losses = []

    for _ in range(epochs):
        logits = x @ vector + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return ProbeResult(vector.detach().cpu(), bias.detach().cpu(), losses)


def complementary_loss(vectors: dict[str, torch.Tensor]) -> torch.Tensor:
    names = list(vectors)
    loss = torch.zeros((), device=next(iter(vectors.values())).device)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            loss = loss - F.cosine_similarity(vectors[left], vectors[right], dim=0)
    return loss


def subspace_loss(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    basis = basis.to(device=vector.device, dtype=vector.dtype)
    projection = basis @ (basis.T @ vector)
    return torch.sum((vector - projection) ** 2)


def refine_vectors(
    activations: dict[str, tuple[torch.Tensor, torch.Tensor]],
    initial_vectors: dict[str, torch.Tensor] | None = None,
    subspaces: dict[str, torch.Tensor] | None = None,
    epochs: int = 300,
    lr: float = 1e-3,
    lambda_com: float = 1e-1,
    lambda_sub: float = 1e-2,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    params = {}
    biases = {}
    data = {}
    for task, (positive, negative) in activations.items():
        x, y = make_probe_data(positive, negative)
        data[task] = (x.to(device), y.to(device))
        init = initial_vectors[task].float() if initial_vectors and task in initial_vectors else torch.randn(x.shape[1]) * 1e-3
        params[task] = init.to(device).clone().detach().requires_grad_(True)
        biases[task] = torch.zeros((), device=device, requires_grad=True)

    optimizer = torch.optim.Adam(list(params.values()) + list(biases.values()), lr=lr)

    for _ in range(epochs):
        loss = torch.zeros((), device=device)
        for task, (x, y) in data.items():
            logits = x @ params[task] + biases[task]
            loss = loss + F.binary_cross_entropy_with_logits(logits, y)
        loss = loss + lambda_com * complementary_loss(params)
        if subspaces:
            for task, basis in subspaces.items():
                if task in params:
                    loss = loss + lambda_sub * subspace_loss(params[task], basis)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {task: vector.detach().cpu() for task, vector in params.items()}
