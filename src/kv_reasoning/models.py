from __future__ import annotations

import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import torch_dtype


def load_causal_lm(
    model_id: str,
    dtype: str = "float16",
    device_map: str | dict = "auto",
    hf_token: str | None = None,
):
    token = hf_token or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch_dtype(dtype),
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer

