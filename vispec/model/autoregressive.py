"""Greedy autoregressive decoding on the shared SpecModel.

Runs on the same loaded ``SpecModel`` instance that ViSpec uses, so both methods share
weights, KV-cache fork kernels and the preallocated KV buffer (``kv_layout="fixed"``). With
``kv_layout="compact"`` the target cache is sized per sample instead, as in ``specgenerate``.
"""

import time

import torch

from .kv_cache import (
    allocate_past_key_values,
    compact_past_key_values,
    expected_kv_capacity,
    initialize_past_key_values,
)
from .utils import reset_tree_mode


def _allocate_kv(model):
    try:
        return initialize_past_key_values(model.base_model, max_length=model.kv_capacity)
    except Exception:
        return initialize_past_key_values(
            model.base_model.language_model, max_length=model.kv_capacity
        )


@torch.inference_mode()
def autoregressive_generate(
    model,
    model_inputs: dict,
    max_new_tokens: int,
    on_first_token=None,
    generation_stats: dict | None = None,
):
    """Greedy decoding. Returns (prompt+output ids [1, L], new_token_count, steps).

    ``model_inputs`` holds ``input_ids`` plus the multimodal tensors (pixel values, grid
    metadata, ...) that are forwarded unchanged to the base model on the prefill call.
    ``on_first_token(token)`` is called once, right after the prefill's argmax.
    ``generation_stats`` receives ``stop_reason``, ``iterations``, ``generation_limits``
    and ``decode_time`` (seconds from the first token being ready to the end of the loop).
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    kwargs = dict(model_inputs)
    input_ids = kwargs.pop("input_ids")
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    input_len = input_ids.shape[1]
    tokenizer = model.tokenizer

    if model.kv_capacity is not None and input_len + max_new_tokens + 10 > model.kv_capacity:
        raise ValueError(
            f"KV capacity {model.kv_capacity} is too small: prompt_len={input_len} "
            f"+ max_new_tokens={max_new_tokens} + 10 = {input_len + max_new_tokens + 10} > "
            f"kv_capacity={model.kv_capacity}"
        )

    expected = expected_kv_capacity(model)
    compact = getattr(model, "kv_layout", "fixed") == "compact"
    if compact:
        # A fresh buffer for the prefill only (prompt + margin); resized after prefill.
        # allocate_past_key_values frees the previous sample's buffer first.
        allocate_past_key_values(model, input_len + 10)
        past_key_values = model.past_key_values
    elif (
        hasattr(model, "past_key_values")
        and model.past_key_values_data[0].shape[3] == expected
    ):
        past_key_values = model.past_key_values
        # Reset the past key and value states
        model.current_length_data.zero_()
    else:
        past_key_values, past_key_values_data, current_length_data = _allocate_kv(model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    reset_tree_mode(model)

    slim = {"logits_to_keep": 1} if getattr(model, "slim_prefill", False) else {}

    outputs = model.base_model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        **slim,
        **kwargs,
    )
    input_id = outputs.logits[:, -1:].argmax(dim=-1)
    input_ids = torch.cat([input_ids, input_id], dim=-1)
    if on_first_token is not None:
        on_first_token(input_id)

    eos = tokenizer.eos_token_id
    stop_reason = "token_limit"
    steps = 0
    torch.cuda.synchronize()
    start_time = time.time()
    # As in specgenerate, the compaction is the first step of the decode phase: decode_time
    # includes it and TTFT does not. Afterwards the cache holds input_len - prompt_kept fewer
    # positions than there are tokens, so decode positions come from the token count.
    position_ids = None
    if compact:
        # keep=None: nothing is evicted, so every prefill position is kept. The capacity
        # formula matches specgenerate's (one draft tree of margin included).
        prompt_kept = int(model.current_length_data[0])
        tree_slots = model.spec_layer.total_tokens + 1
        compact_past_key_values(
            model, None, prompt_kept + max_new_tokens + tree_slots + 10
        )
        # Rebind, so no local keeps the prefill-sized buffer alive.
        past_key_values = model.past_key_values
        rope_deltas = model.base_model.rope_deltas
    while True:
        if eos is not None and input_id.item() == eos:
            stop_reason = "eos"
            break
        if input_ids.shape[1] - input_len >= max_new_tokens:
            stop_reason = "token_limit"
            break
        if compact:
            # input_id is the last token of input_ids, at logical index shape[1] - 1. Its
            # M-RoPE position is that index plus rope_deltas on all 3 axes (as tree_decoding
            # builds them), never the cache length the base model would otherwise use.
            position_ids = (rope_deltas + (input_ids.shape[1] - 1)).unsqueeze(0).expand(
                3, -1, -1
            )
        outputs = model.base_model(
            input_id,
            use_cache=True,
            past_key_values=past_key_values,
            position_ids=position_ids,
            **slim,
        )
        input_id = outputs.logits[:, -1:].argmax(dim=-1)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        steps += 1
    torch.cuda.synchronize()
    end_time = time.time()

    if generation_stats is not None:
        generation_stats.update(
            stop_reason=stop_reason,
            iterations=steps + 1,
            generation_limits={"max_new_tokens": max_new_tokens},
            decode_time=end_time - start_time,
        )
    new_token = input_ids.shape[1] - input_len
    return input_ids, new_token, steps
