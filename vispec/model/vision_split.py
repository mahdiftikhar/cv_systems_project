"""Per-image vision encoding for Qwen2.5-VL multi-image prompts."""

import torch


def encode_images_separately(visual):
    """Make a Qwen2.5-VL vision tower encode each image of a call on its own.

    Why: the SDPA vision attention builds a dense [1, L, L] boolean mask over the
    patches of every image in the call, and the kernel materialises about 16 * L^2
    fp32 scores. With six manga pages L is about 27.5k, which asks for 45 GiB and
    runs out of memory before any decoding.

    Why it is exact: in Qwen2.5-VL the full-attention blocks use cu_seqlens that
    block-diagonalise attention per image, the window index and window cu_seqlens
    are computed per image, the rotary embedding is built from each image's own
    (t, h, w) grid, and the patch merger works on groups inside one image. Encoding
    the images one at a time and concatenating the outputs therefore gives the same
    result as the batched call, with an L x L mask per image only.

    A call with a single image runs the original forward with unchanged arguments.
    Installing twice does not wrap twice. Returns the vision tower.
    """
    if getattr(visual, "_encodes_images_separately", False):
        return visual
    original_forward = visual.forward

    def forward(hidden_states, grid_thw, *args, **kwargs):
        if grid_thw.shape[0] <= 1:
            return original_forward(hidden_states, grid_thw, *args, **kwargs)
        sizes = [int(t) * int(h) * int(w) for t, h, w in grid_thw.tolist()]
        if sum(sizes) != hidden_states.shape[0]:
            raise ValueError(
                f"grid_thw covers {sum(sizes)} patches but hidden_states has "
                f"{hidden_states.shape[0]} rows"
            )
        outputs = [
            original_forward(chunk, grid_thw[i : i + 1], *args, **kwargs)
            for i, chunk in enumerate(hidden_states.split(sizes, dim=0))
        ]
        return torch.cat(outputs, dim=0)

    visual.forward = forward
    visual._encodes_images_separately = True
    return visual
