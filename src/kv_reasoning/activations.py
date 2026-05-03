from __future__ import annotations

import torch

from .steering import _hidden_from_output, decoder_layers


def generate_with_average_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    max_new_tokens: int = 512,
) -> tuple[list[str], torch.Tensor]:
    """Generation plus mean residual activation over generated-token steps."""

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

    collected: list[torch.Tensor] = []
    state = {"step": 0}

    def hook(_module, _inputs, output):
        hidden = _hidden_from_output(output)
        if state["step"] > 0:
            collected.append(hidden[:, -1, :].detach().float().cpu())
        state["step"] += 1
        return output

    handle = decoder_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    prompt_width = inputs["input_ids"].shape[1]
    decoded = [tokenizer.decode(row, skip_special_tokens=True) for row in outputs[:, prompt_width:]]

    if not collected:
        hidden_size = model.config.hidden_size
        return decoded, torch.zeros((len(prompts), hidden_size), dtype=torch.float32)

    stacked = torch.stack(collected, dim=0)
    return decoded, stacked.mean(dim=0)
