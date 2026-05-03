from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not find decoder layers on this model.")


def _replace_hidden(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def _hidden_from_output(output) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def steering_hook(
    model,
    vector: torch.Tensor,
    layer_idx: int,
    coeff: float,
    mode: str = "generate",
) -> Iterator[None]:
    """Add a steering vector to the last-token residual stream during generation."""

    if mode not in {"prompt", "generate", "all"}:
        raise ValueError("mode must be one of: prompt, generate, all")

    state = {"step": 0}

    def hook(_module, _inputs, output):
        hidden = _hidden_from_output(output)
        should_apply = (
            mode == "all"
            or (mode == "prompt" and state["step"] == 0)
            or (mode == "generate" and state["step"] > 0)
        )
        state["step"] += 1
        if not should_apply:
            return output

        steered = hidden.clone()
        direction = vector.to(device=steered.device, dtype=steered.dtype)
        steered[:, -1, :] = steered[:, -1, :] + coeff * direction
        return _replace_hidden(output, steered)

    handle = decoder_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    steering_vector: torch.Tensor | None = None,
    layer_idx: int = 13,
    coeff: float = 1.0,
    steering_mode: str = "generate",
) -> list[str]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    context = (
        steering_hook(model, steering_vector, layer_idx, coeff, steering_mode)
        if steering_vector is not None
        else null_context()
    )
    with context, torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_width = inputs["input_ids"].shape[1]
    decoded = []
    for row in outputs[:, prompt_width:]:
        decoded.append(tokenizer.decode(row, skip_special_tokens=True))
    return decoded


@contextmanager
def null_context() -> Iterator[None]:
    yield
