"""Shared image-token selection for visual sparsity.

Target-side visual KV eviction and draft-side visual pruning both pick which image tokens to
keep with the functions here. Text positions are never dropped, every contiguous image run
keeps at least one token, and retention 1.0 returns every position unchanged.
"""
from dataclasses import dataclass

import torch

POLICIES = ("spatial", "attention")
RETENTIONS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class SparsityConfig:
    """
    Visual sparsity settings.

    Attributes:
        retention (float): Fraction of each image's tokens to keep: 0.25, 0.5, 0.75 or 1.0.
        policy (str): "spatial" (uniform raster stride) or "attention" (top scores).
    """

    retention: float = 1.0
    policy: str = "spatial"

    def __post_init__(self):
        if self.retention not in RETENTIONS:
            raise ValueError(f"retention must be one of {RETENTIONS}, got {self.retention!r}")
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")


def image_spans(image_mask):
    """
    Find the contiguous runs of image tokens.

    Args:
        image_mask (torch.Tensor): 1-D bool (or 0/1 integer) mask, True at image tokens.

    Returns:
        list[tuple[int, int]]: [start, end) for each run, in ascending order.
    """
    if image_mask.dim() != 1:
        raise ValueError(f"image_mask must be 1-D, got shape {tuple(image_mask.shape)}")
    mask = image_mask.bool()
    if mask.numel() == 0:
        return []
    padded = torch.cat([mask.new_zeros(1), mask, mask.new_zeros(1)]).to(torch.int8)
    edges = (padded[1:] - padded[:-1]).nonzero().flatten().tolist()
    return [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]


def keep_count(n, retention):
    """
    Number of tokens to keep from a span of n image tokens.

    Returns n when retention is 1.0, otherwise max(1, round(retention * n)). round() is
    Python's, so halves go to the even integer.
    """
    if retention == 1.0:
        return n
    return max(1, round(retention * n))


def select_in_span(n, k, policy, scores):
    """
    Choose k of the n positions inside one image span.

    Args:
        n (int): Span length.
        k (int): Number to keep, 1 <= k <= n.
        policy (str): "spatial" or "attention".
        scores (torch.Tensor | None): Shape [n] importance scores. Required for "attention".

    Returns:
        torch.Tensor: Ascending long indices into the span, shape [k].

    Raises:
        ValueError: Unless 1 <= k <= n, so a bad count fails here instead of returning a
            short, repeated or empty selection.
    """
    if not 1 <= k <= n:
        raise ValueError(f"select_in_span needs 1 <= k <= n, got k={k}, n={n}")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if policy == "attention":
        if scores is None:
            raise ValueError("policy 'attention' requires scores")
        if scores.shape != (n,):
            raise ValueError(f"scores must have shape ({n},), got {tuple(scores.shape)}")
        return scores.topk(k).indices.sort().values
    idx = torch.linspace(0, n - 1, k).round().long().unique()
    if idx.numel() < k:
        unused = torch.ones(n, dtype=torch.bool)
        unused[idx] = False
        extra = unused.nonzero().flatten()[: k - idx.numel()]
        idx = torch.cat([idx, extra]).sort().values
    return idx


def keep_positions(image_mask, config, scores=None):
    """
    Positions to keep: every non-image position plus the selected tokens of each image span.

    Args:
        image_mask (torch.Tensor): 1-D bool (or 0/1 integer) mask, True at image tokens.
        config (SparsityConfig): Retention and policy.
        scores (torch.Tensor | None): Shape [len(image_mask)] scores, sliced per span.
            Required for the "attention" policy unless retention is 1.0.

    Returns:
        torch.Tensor: Ascending long indices on image_mask's device.
    """
    if image_mask.dim() != 1:
        raise ValueError(f"image_mask must be 1-D, got shape {tuple(image_mask.shape)}")
    mask = image_mask.bool()
    total = mask.numel()
    if config.retention == 1.0:
        return torch.arange(total, device=mask.device)
    if config.policy == "attention" and (scores is None or scores.shape != (total,)):
        raise ValueError(f"policy 'attention' requires scores of shape ({total},)")
    kept = [(~mask).nonzero().flatten()]
    for start, end in image_spans(mask):
        n = end - start
        k = keep_count(n, config.retention)
        span_scores = scores[start:end] if scores is not None else None
        if span_scores is not None:
            span_scores = span_scores.to(mask.device)
        sel = select_in_span(n, k, config.policy, span_scores if config.policy == "attention" else None)
        kept.append(sel.to(mask.device) + start)
    return torch.cat(kept).sort().values


def kept_per_image(image_mask, positions):
    """
    Kept token count of each contiguous image run, read from the kept positions.

    Args:
        image_mask (torch.Tensor): 1-D bool (or 0/1 integer) mask, True at image tokens.
        positions (torch.Tensor): The kept positions (indices into image_mask), as
            keep_positions returns them.

    Returns:
        list[int]: One count per image span, in span order. Computed on the CPU.
    """
    mask = image_mask.bool().cpu()
    kept = torch.zeros(mask.numel(), dtype=torch.bool)
    kept[torch.as_tensor(positions).cpu().long()] = True
    return [int(kept[start:end].sum()) for start, end in image_spans(mask)]
