# The HiViS drafter, vendored from https://github.com/lnn-ops/HiViS at commit
# 6db36b241eaf733ec313cbc9b86206c0f20762c9, under the Apache License, Version 2.0. This
# repository's own LICENSE is also Apache-2.0.
#
# Paper: "HiViS: Hiding Visual Tokens from the Drafter for Speculative Decoding in
# Vision-Language Models", arXiv:2509.23928, CVPR 2026 Findings.
#
# Copied code, in this order:
# 1. hivis/model/cnets_hivis.py, lines 1-1000: the original licence notice below, the rotary
#    classes, the Llama layer and Model.
# 2. hivis/model/utils_hivis.py, lines 21-105: build_mask and prune_image_tokens.
#
# Changes to the copied code (Apache-2.0 section 4(b)):
# - In part 1, the import "try: from .configs import EConfig / except: from configs import
#   EConfig" is replaced by "from .configs import EConfig", this repository's config class.
# - Lines 1002-1084 of hivis/model/cnets_hivis.py (acc, Vhead, count_parameters and the
#   __main__ block) are left out.
# - Before part 2, "from torch.nn.utils.rnn import pad_sequence" is added, because
#   prune_image_tokens uses it (hivis/model/utils_hivis.py imports it at line 8).
# - Added after the copied code: the HiViSDraft adapter and load_hivis_draft. They let this
#   repository's unchanged tree loop (vispec/model/utils.py) drive the HiViS drafter.
# Apart from these changes, the copied lines are unchanged.
#
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""
import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
import os
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN

from .configs import EConfig


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).")
        if hasattr(config, "qkv_bias"):
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.qkv_bias,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.qkv_bias,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=False
            )
            self.k_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
            )
            self.v_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
            )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                )
            else:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings
                )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_ids is None:
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if len(past_key_value == 2):
                    kv_seq_len += past_key_value[0].shape[-2]
                elif len(past_key_value == 3):
                    kv_seq_len += past_key_value[2]
                else:
                    raise NotImplementedError
        else:
            kv_seq_len = position_ids.max() + 1
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            attn_mask=attention_mask.to(query_states.dtype),
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        return x + self.dummy - self.dummy


def len_list(x, n):
    return [i for i in x if len(i) <= n]


class Model(nn.Module):

    def __init__(
        self,
        config,
        load_emb=False,
        path=None,
        bias=True,
        total_tokens=30,
        depth=3,
        top_k=8,
        threshold=1.0,
        residual_count=4,
    ):
        super().__init__()

        if residual_count < 1:
            raise ValueError(
                f"residual_count must be positive, got {residual_count}"
            )

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["language_model.model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path), framework="pt", device="cpu") as f:
                    tensor_slice = f.get_slice("language_model.model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path), framework="pt", device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()

            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        # Qwen's pruned visual span leaves a position offset; LLaVA uses zero.
        self.add = 0
        # print("total_tokens",total_tokens)
        # print("depth",depth)
        # print("top_k",top_k)
        # print("threshold",threshold)

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.fc = nn.Linear(3 * config.hidden_size, config.hidden_size, bias=bias)
        self.act = ACT2FN[config.hidden_act]
        self.residual = nn.Parameter(
            torch.empty(residual_count, config.hidden_size)
        )
        nn.init.xavier_uniform_(self.residual)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.register_buffer(
            "tree_mask_init",
            torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None],
            persistent=False,
        )
        self.register_buffer(
            "position_ids",
            torch.zeros(
                self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long
            ),
            persistent=False,
        )

    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, torch.float32, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
            ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
        self,
        hidden_states,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        forward_num = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        batch_size, seq_length, hidden_size = hidden_states.shape
        past_key_values_length = 0

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if inputs_embeds is None:
            with torch.no_grad():
                inputs_embeds = self.embed_tokens(input_ids)
                
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        if position_ids is not None:
            past_key_values_real_length = position_ids.min()
        else:
            past_key_values_real_length = past_key_values_length

        if position_ids is None:
            device = (
                hidden_states.device
                if hidden_states is not None
                else inputs_embeds.device
            )
            position_ids = torch.arange(
                past_key_values_real_length,
                seq_length + past_key_values_real_length,
                dtype=torch.long,
                device=device,
            )+self.add
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        # print(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        index = forward_num if forward_num < self.residual.shape[0] else -1
        residual = self.residual[index].expand(batch_size, seq_length, hidden_size).to(hidden_states.dtype)

        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states, residual), dim=-1))


        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None

    @torch.no_grad()
    def topK_genrate(self, 
                     hidden_states, 
                     input_ids, 
                     head, 
                     logits_processor, 
                     input_embeds=None, 
                     position_ids=None):
        # print(position_ids)

        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k
        # print("depth: ", depth)

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        if input_embeds is not None:
            inputs_embeds = input_embeds[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        if position_ids is not None:
            len_posi = position_ids.max()+1
            self.add = len_posi - input_ids.size(1)
        else:
            len_posi = input_ids.size(1) + self.add
        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids[:, kv_len:],
                past_key_values=self.stable_kv,
                use_cache=True,
            )
        else:
            if inputs_embeds is not None:
                input_ids = None
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                position_ids=position_ids,
            )
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = head(last_hidden)

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            out_hidden, past_key_values = self(
                input_hidden,
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
                forward_num=i+1
            )
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k**2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat(
                (tree_mask[:, :, out_ids], self.tree_mask_init), dim=3
            )

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(
            top_scores_index, draft_parents - 1, right=False
        )
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents


        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del (
            mask_index,
            mask_index_list,
            noleaf_index,
            noleaf_num,
            leaf_num,
            max_depth,
            rid,
        )
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


# ---------------------------------------------------------------------------------------------
# Part 2: copied from HiViS hivis/model/utils_hivis.py, lines 21-105.
# ---------------------------------------------------------------------------------------------
from torch.nn.utils.rnn import pad_sequence


def build_mask(input_ids: torch.Tensor):
    VISION_START = 151652
    IMAGE_TOKEN  = 151655
    # IMAGE_TOKEN  = 151656

    mask = torch.ones_like(input_ids, dtype=torch.bool)

    pos = (input_ids == VISION_START).nonzero(as_tuple=False).squeeze(-1)
    if len(pos) == 0:
        return mask

    start = pos.item()
    count = int((input_ids == IMAGE_TOKEN).sum().item())
    end = min(start + count + 1, input_ids.numel() - 1)

    mask[start:end+2] = False

    return mask

def prune_image_tokens(
    input_ids: torch.LongTensor,
    embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    text_position_ids: torch.Tensor = None,
    image_token_id: int = 32000,
    pad_token_id: int = 0,
):
    batch_size = input_ids.shape[0]
    hidden_size = embeds.size(-1)
    device = hidden_states.device
    input_ids = input_ids.to(device)
    embeds = embeds.to(device)

    ids_list = []
    embeds_list = []
    states_list = []
    position_ids_list = []

    for batch_index in range(batch_size):
        if text_position_ids is None:
            keep_mask = input_ids[batch_index] != image_token_id
        else:
            keep_mask = build_mask(input_ids[batch_index])

        kept_ids = input_ids[batch_index][keep_mask]
        kept_embeds = embeds[batch_index][keep_mask]
        kept_states = hidden_states[batch_index][keep_mask]
        kept_position_ids = (
            text_position_ids[batch_index][keep_mask]
            if text_position_ids is not None
            else None
        )

        if kept_ids.numel() == 0:
            kept_ids = torch.tensor([pad_token_id], device=device, dtype=input_ids.dtype)
            kept_embeds = torch.zeros((1, hidden_size), device=device, dtype=embeds.dtype)
            kept_states = torch.zeros((1, hidden_size), device=device, dtype=hidden_states.dtype)
            if kept_position_ids is not None:
                kept_position_ids = torch.zeros(
                    1, device=device, dtype=text_position_ids.dtype
                )

        ids_list.append(kept_ids)
        embeds_list.append(kept_embeds)
        states_list.append(kept_states)
        if kept_position_ids is not None:
            position_ids_list.append(kept_position_ids)

    if batch_size == 1:
        pruned_input_ids = ids_list[0].unsqueeze(0)
        pruned_embeds = embeds_list[0].unsqueeze(0)
        pruned_hidden_states = states_list[0].unsqueeze(0)
    else:
        pruned_input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_token_id)
        pruned_embeds = pad_sequence(embeds_list, batch_first=True, padding_value=0.0)
        pruned_hidden_states = pad_sequence(states_list, batch_first=True, padding_value=0.0)

    if text_position_ids is None:
        return pruned_input_ids, pruned_embeds, pruned_hidden_states

    if batch_size == 1:
        pruned_position_ids = position_ids_list[0].unsqueeze(0)
    else:
        pruned_position_ids = pad_sequence(position_ids_list, batch_first=True, padding_value=0)
    return pruned_input_ids, pruned_embeds, pruned_hidden_states, pruned_position_ids


# ---------------------------------------------------------------------------------------------
# Adapter for this repository (not part of HiViS).
# ---------------------------------------------------------------------------------------------
import json

# The token ids that HiViS's build_mask hard-codes for Qwen2.5-VL: VISION_START and
# IMAGE_TOKEN, and the video pad that its comment names. The target config must agree.
HIVIS_VISION_START_ID = 151652
HIVIS_IMAGE_TOKEN_ID = 151655
HIVIS_VIDEO_TOKEN_ID = 151656
# <|vision_end|>, which multi_image_mask uses to find where each image ends.
HIVIS_VISION_END_ID = 151653
POSITION_MODES = ("original", "pruned")


def multi_image_mask(input_ids: torch.Tensor):
    """HiViS's ``build_mask`` rule applied to every image of a 1-D prompt.

    HiViS's own ``build_mask`` handles one image. It removes ``<|vision_start|>``, the image
    pads, ``<|vision_end|>`` and the one token after it, with an end clamp at
    ``numel - 1``. This function removes the same span for each image in turn, counting
    that image's own pads and keeping the same clamp. A prompt with at most one
    ``<|vision_start|>`` goes to ``build_mask`` unchanged, so the single-image mask is
    HiViS's exactly.

    Raises ValueError unless every image is ``<|vision_start|>``, image pads only and then
    ``<|vision_end|>``, before the next image starts.
    """
    starts = (input_ids == HIVIS_VISION_START_ID).nonzero().flatten().tolist()
    if len(starts) <= 1:
        return build_mask(input_ids)
    ends = (input_ids == HIVIS_VISION_END_ID).nonzero().flatten().tolist()
    if len(ends) != len(starts):
        raise ValueError(
            f"the prompt has {len(starts)} <|vision_start|> but {len(ends)} <|vision_end|> tokens"
        )
    length = input_ids.numel()
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for start, end, following in zip(starts, ends, starts[1:] + [length]):
        if not start < end < following:
            raise ValueError(
                f"image span {start}..{end} is not closed before the next image or the end "
                f"of the prompt ({following})"
            )
        count = int((input_ids[start + 1 : end] == HIVIS_IMAGE_TOKEN_ID).sum())
        if count != end - start - 1:
            raise ValueError(
                f"image span {start}..{end} holds {end - start - 1 - count} tokens that are "
                "not image pads"
            )
        stop = min(start + count + 1, length - 1)
        mask[start : stop + 2] = False
    return mask


class HiViSDraft(Model):
    """The HiViS drafter, driven by the unchanged tree loop in ``vispec/model/utils.py``.

    ``initialize_tree`` and ``update_inference_inputs`` call ``topK_genrate`` with the full
    token ids, as they do for the ViSpec draft. This class turns those calls into the calls
    that HiViS's own loop makes, and leaves the copied ``Model`` to do the rest:

    - First call (no draft KV yet). ``input_ids`` is the prompt plus the first token
      (``L + 1``), and ``hidden_states`` and ``inputs_embeds`` cover the prompt (``L``).
      ``build_mask`` on the prompt ids (``L``, so its ``numel - 1`` clamp is HiViS's) selects
      the kept tokens. The ids, the target's embeddings and the target's hidden states are
      kept under that mask, and each kept token keeps its index in ``arange(L)`` as its
      position. The draft embeddings are the kept target embeddings plus the draft's own
      embedding of the first token. This follows ``hivis/model/utils_hivis.py`` lines
      349-410. The copied ``topK_genrate`` then sets ``add = L - P``, where ``P`` is the kept
      length.
    - Later calls. ``input_ids`` is the full sequence plus the new token. The draft ids are
      the kept prompt ids followed by every token after the prompt, which is the
      ``cat(draft_input_ids, token)`` of HiViS's ``update_inference_inputs`` (lines 618-656).

    With one image in the prompt, the draft drops ``<|vision_start|>``, the image pads,
    ``<|vision_end|>`` and the one token after it. With the image last in the user turn that
    token is ``<|im_end|>``, so only ``\\n<|im_start|>assistant\\n`` follows the image in the
    draft's prompt. With the image first it is the first question token, as in HiViS's own
    evaluation. A prompt without an image keeps every token, and ``add`` stays 0.

    With several images, as in the Re:Verse+ page windows, ``multi_image_mask`` removes the
    same span for each image. The pages there follow one another, so the token after each
    page's ``<|vision_end|>`` is the next page's ``<|vision_start|>``, and after the last page
    it is the first instruction token, as with one image placed first. HiViS was trained and
    released for one image per prompt, so several images are an extension of it.

    The class attribute ``position_mode`` is ``"original"``. Setting it to ``"pruned"`` gives
    the kept prompt tokens the positions ``0 .. P-1`` instead, so ``add`` is 0. Only a
    position ablation sets it.

    ``draft_sparsity`` stays None: the draft sees no image tokens, so draft-side pruning has
    nothing to act on.
    """

    position_mode = "original"

    def __init__(
        self,
        config,
        *,
        bias,
        total_tokens,
        depth,
        top_k,
        threshold,
        residual_count,
        vision_start_id,
        image_token_id,
        video_token_id,
    ):
        given = (vision_start_id, image_token_id, video_token_id)
        expected = (HIVIS_VISION_START_ID, HIVIS_IMAGE_TOKEN_ID, HIVIS_VIDEO_TOKEN_ID)
        if given != expected:
            raise ValueError(
                "the HiViS draft needs the Qwen2.5-VL token ids (vision start, image pad, "
                f"video pad) = {expected}, got {given}"
            )
        super().__init__(
            config,
            bias=bias,
            total_tokens=total_tokens,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            residual_count=residual_count,
        )
        self.vision_start_id = vision_start_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.draft_sparsity = None
        self.draft_image_counts = None
        self.reset_kv()

    def reset_kv(self):
        """Clear the draft KV, the prompt state and the position offset ``add``."""
        super().reset_kv()
        self.prompt_len = None
        self.pruned_prompt_ids = None
        self.prompt_positions = None
        self.add = 0

    def keep_mask(self, prompt_ids, image_mask=None):
        """HiViS's keep mask for one prompt row, after the guards.

        ``prompt_ids`` is 1-D, the prompt without the first generated token. With at most one
        image the mask is HiViS's ``build_mask``, and with several it is ``multi_image_mask``,
        the same rule for each image. Raises ValueError for any video pad and for an image
        position outside the removed spans. The image positions are those of ``image_mask``
        (when given, one flag per prompt token) and the image pads in ``prompt_ids``.
        """
        if prompt_ids.dim() != 1:
            raise ValueError(f"prompt_ids must be 1-D, got shape {tuple(prompt_ids.shape)}")
        videos = int((prompt_ids == self.video_token_id).sum())
        if videos:
            raise ValueError(f"the HiViS draft takes no video, but the prompt has {videos} video pads")
        keep = multi_image_mask(prompt_ids)
        image_positions = {"image pad ids": prompt_ids == self.image_token_id}
        if image_mask is not None:
            image_mask = image_mask.reshape(-1)
            if image_mask.numel() != prompt_ids.numel():
                raise ValueError(
                    f"image_mask has {image_mask.numel()} entries for a prompt of "
                    f"{prompt_ids.numel()} tokens"
                )
            image_positions["image_mask"] = image_mask.to(device=keep.device, dtype=torch.bool)
        for name, positions in image_positions.items():
            outside = (positions & keep).nonzero().flatten().tolist()
            if outside:
                raise ValueError(
                    f"{name} marks positions {outside[:8]} outside the image span that the "
                    "HiViS draft removes"
                )
        return keep

    @torch.no_grad()
    def topK_genrate(
        self,
        hidden_states,
        input_ids,
        head,
        logits_processor,
        inputs_embeds=None,
        embed_weights=None,
        image_mask=None,
    ):
        if embed_weights is not None:
            raise ValueError("the HiViS draft takes no embed_weights")
        if input_ids.shape[0] != 1:
            raise ValueError(f"the HiViS draft supports batch size 1, got {input_ids.shape[0]}")
        if getattr(self, "stable_kv", None) is None:
            return self._first_topK_genrate(
                hidden_states, input_ids, head, logits_processor, inputs_embeds, image_mask
            )
        if self.prompt_len is None:
            raise RuntimeError("the HiViS draft has a KV cache but no prompt; call reset_kv() first")
        if input_ids.shape[1] < self.prompt_len + 2:
            raise ValueError(
                f"a later draft call needs the prompt ({self.prompt_len} tokens) plus at least "
                f"two tokens, got {input_ids.shape[1]}"
            )
        device = hidden_states.device
        draft_ids = torch.cat(
            (self.pruned_prompt_ids, input_ids[:, self.prompt_len :].to(device)), dim=1
        )
        return super().topK_genrate(hidden_states, draft_ids, head, logits_processor)

    def _first_topK_genrate(
        self, hidden_states, input_ids, head, logits_processor, inputs_embeds, image_mask
    ):
        if self.position_mode not in POSITION_MODES:
            raise ValueError(
                f"position_mode must be one of {POSITION_MODES}, got {self.position_mode!r}"
            )
        if inputs_embeds is None:
            raise ValueError("the HiViS draft needs the target's prompt embeddings on its first call")
        prompt_len = hidden_states.shape[1]
        if input_ids.shape[1] != prompt_len + 1 or inputs_embeds.shape[1] != prompt_len:
            raise ValueError(
                "the first HiViS draft call needs input_ids of length L + 1 and hidden states "
                f"and embeddings of length L; got {input_ids.shape[1]}, {prompt_len} and "
                f"{inputs_embeds.shape[1]}"
            )
        device = hidden_states.device
        input_ids = input_ids.to(device)
        prompt_ids = input_ids[:, :-1]
        token = input_ids[:, -1:]
        keep = self.keep_mask(prompt_ids[0], image_mask)

        kept_ids = prompt_ids[:, keep]
        kept_embeds = inputs_embeds.to(device)[:, keep]
        kept_hidden = hidden_states[:, keep]
        kept_len = kept_ids.shape[1]
        if kept_len == 0:
            raise ValueError("the HiViS draft mask removed every prompt token")
        if self.position_mode == "original":
            positions = torch.arange(prompt_len, dtype=torch.long, device=device)[keep]
        else:
            positions = torch.arange(kept_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0)

        new_embeds = self.embed_tokens(token)
        draft_embeds = torch.cat((kept_embeds, new_embeds), dim=1)
        draft_ids = torch.cat((kept_ids, token), dim=1)

        self.prompt_len = prompt_len
        self.pruned_prompt_ids = kept_ids
        self.prompt_positions = positions
        return super().topK_genrate(
            kept_hidden,
            draft_ids,
            head,
            logits_processor,
            input_embeds=draft_embeds,
            position_ids=positions,
        )


def load_hivis_draft(config_path, state_dict, total_token, depth, top_k, threshold, base_config):
    """Build a ``HiViSDraft`` from the draft's ``config.json`` and load ``state_dict`` strictly.

    ``residual_count`` comes from ``state_dict["residual"]``, as in HiViS's
    ``hivis/model/model_hivis.py`` lines 95-97, and the token ids from the target config
    ``base_config``. ``bias`` is read from the config file as ``SpecModel`` reads it for the
    ViSpec draft.
    """
    if state_dict is None or "residual" not in state_dict:
        raise ValueError(
            "the HiViS draft needs a checkpoint with a 'residual' tensor; "
            f"none was found (config {config_path})"
        )
    vision_end_id = getattr(base_config, "vision_end_token_id", None)
    if vision_end_id != HIVIS_VISION_END_ID:
        raise ValueError(
            f"the HiViS draft needs the Qwen2.5-VL vision end id {HIVIS_VISION_END_ID}, "
            f"got {vision_end_id}"
        )
    config = EConfig.from_pretrained(config_path)
    with open(config_path, "r") as f:
        con = json.loads(f.read())
    bias = con.get("bias", True)
    draft = HiViSDraft(
        config,
        bias=bias,
        total_tokens=total_token,
        depth=depth,
        top_k=top_k,
        threshold=threshold,
        residual_count=state_dict["residual"].shape[0],
        vision_start_id=getattr(base_config, "vision_start_token_id", None),
        image_token_id=getattr(base_config, "image_token_id", None),
        video_token_id=getattr(base_config, "video_token_id", None),
    )
    draft.load_state_dict(state_dict, strict=True)
    return draft
