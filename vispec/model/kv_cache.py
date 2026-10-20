import torch

# "fixed" preallocates one buffer per model and reuses it; "compact" sizes the target cache
# per sample and lets its length (physical) differ from the token count (logical).
KV_LAYOUTS = ("fixed", "compact")


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)


def expected_kv_capacity(model):
    """Token slots a fresh preallocated KV buffer would have for this SpecModel.

    ``model.kv_capacity`` when set, otherwise ``max_position_embeddings`` of the base
    model's config, or of its ``language_model`` config for wrappers (LLaVA) whose
    top-level config has no such field. This mirrors the capacity that
    ``initialize_past_key_values`` gives the same model.
    """
    if model.kv_capacity is not None:
        return model.kv_capacity
    try:
        return model.base_model.config.max_position_embeddings
    except AttributeError:
        return model.base_model.language_model.config.max_position_embeddings


def initialize_past_key_values(model, max_length: int | None = None):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.
        max_length (int, optional): Number of token slots to preallocate per layer. ``None`` keeps
            ``config.max_position_embeddings`` (the original behaviour).

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    if max_length is None:
        capacity = config.max_position_embeddings
    else:
        capacity = int(max_length)
        if capacity < 1:
            raise ValueError(f"KV capacity must be positive, got {max_length}")
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers

    devices = []
    for i in range(config.num_hidden_layers):
        try:
            device = model.model.layers[i].self_attn.q_proj.weight.device
        except:
            device = model.layers[i].self_attn.q_proj.weight.device
        devices.append(device)
    past_key_values_data_list = []
    startnum = 0
    startdevice = devices[0]
    for id, i in enumerate(devices):
        assert i.type == "cuda"  # TODO

        if startdevice != i:
            past_key_values_data = torch.zeros(
                startnum * 2,
                batch_size,
                config.num_key_value_heads,
                capacity,
                config.hidden_size // config.num_attention_heads,
                device=startdevice,
                dtype=model.dtype,
            )
            past_key_values_data_list.append(past_key_values_data)
            startdevice = i
            startnum = 0
        startnum += 1
    past_key_values_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        capacity,
        config.hidden_size // config.num_attention_heads,
        device=startdevice,
        dtype=model.dtype,
    )
    past_key_values_data_list.append(past_key_values_data)
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers

    bias = 0
    start_data_m = devices[0].index
    for i in range(config.num_hidden_layers):
        data_m = devices[i].index
        if data_m != start_data_m:
            bias = 0
            start_data_m = data_m
        try:
            past_key_values.append(
                [
                    KVCache(
                        past_key_values_data_list[data_m - devices[0].index][
                            2 * bias + j
                        ],
                        current_length_data[i * 2 + j],
                    )
                    for j in range(2)
                ]
            )
        except:
            past_key_values.append(
                [
                    KVCache(
                        past_key_values_data_list[0][2 * bias + j],
                        current_length_data[i * 2 + j],
                    )
                    for j in range(2)
                ]
            )
        bias += 1
    return past_key_values, past_key_values_data_list, current_length_data


def _kv_text_model(model):
    """The SpecModel's decoder that owns the target KV cache (as specgenerate resolves it)."""
    base = model.base_model
    return base.language_model if hasattr(base, "language_model") else base


def release_past_key_values(model):
    """Drop the SpecModel's references to its target KV buffers so they can be freed."""
    for name in ("past_key_values", "past_key_values_data", "current_length_data"):
        if hasattr(model, name):
            delattr(model, name)


def allocate_past_key_values(model, capacity: int) -> None:
    """(Re)build ``model.past_key_values``, ``past_key_values_data`` and
    ``current_length_data`` with exactly ``capacity`` token slots.

    The previous buffers are released first, so they do not coexist with the new ones. The
    caller must not hold its own references to them either.
    """
    capacity = int(capacity)
    if capacity < 1:
        raise ValueError(f"KV capacity must be positive, got {capacity}")
    release_past_key_values(model)
    (
        model.past_key_values,
        model.past_key_values_data,
        model.current_length_data,
    ) = initialize_past_key_values(_kv_text_model(model), max_length=capacity)


def _owner(view, bases):
    """(index into ``bases``, index along dim 0) of ``view``, a dim-0 slice of one base."""
    for index, base in enumerate(bases):
        if view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr():
            return index, (view.storage_offset() - base.storage_offset()) // base.stride(0)
    raise ValueError("KV cache entry does not view any of the model's KV buffers")


def compact_past_key_values(model, keep: torch.LongTensor | None, capacity: int) -> int:
    """Copy the cached positions ``keep`` into fresh buffers of ``capacity`` slots.

    ``keep`` holds strictly ascending cache indices below the current length; ``None`` keeps
    all ``current_length`` positions. Kept entry ``i`` lands in slot ``i`` of every layer, and
    the cache length becomes the kept count, which is returned. The new buffers keep the old
    per-device split and dtype. The old buffers are dropped once copied, so they are freed as
    soon as the caller rebinds its own references to ``model.past_key_values*``.
    """
    old_caches = model.past_key_values
    old_data = model.past_key_values_data
    old_lengths = model.current_length_data
    length = int(old_lengths[0])
    if keep is None:
        kept = length
    else:
        keep = torch.as_tensor(keep, dtype=torch.long)
        check = keep.cpu()
        if check.dim() != 1:
            raise ValueError(f"keep must be 1-D, got shape {tuple(check.shape)}")
        if check.numel() and (int(check[0]) < 0 or int(check[-1]) >= length):
            raise ValueError(f"keep indices must lie in [0, {length})")
        if check.numel() > 1 and not bool((check[1:] > check[:-1]).all()):
            raise ValueError("keep indices must be strictly ascending")
        kept = int(check.numel())
    if kept < 1:
        raise ValueError("compaction must keep at least one cached position")
    capacity = int(capacity)
    if capacity < kept:
        raise ValueError(f"KV capacity {capacity} cannot hold {kept} kept positions")

    new_data = []
    for old in old_data:
        # [2 * layers on this device, batch, kv heads, slots, head dim]
        new = torch.zeros(
            old.shape[:3] + (capacity,) + old.shape[4:], dtype=old.dtype, device=old.device
        )
        if keep is None:
            new[:, :, :, :kept].copy_(old[:, :, :, :kept])
        else:
            index = keep.to(old.device)
            # One layer at a time keeps the gathered temporary to a single layer's entries.
            for entry in range(old.shape[0]):
                new[entry, :, :, :kept].copy_(old[entry].index_select(2, index))
        new_data.append(new)
    new_lengths = torch.zeros_like(old_lengths)
    new_lengths.fill_(kept)
    new_caches = []
    for layer in old_caches:
        pair = []
        for cache in layer:
            buffer, entry = _owner(cache.data, old_data)
            _, length_entry = _owner(cache.current_length, [old_lengths])
            pair.append(KVCache(new_data[buffer][entry], new_lengths[length_entry]))
        new_caches.append(pair)

    del old_caches, old_data, old_lengths
    release_past_key_values(model)
    model.past_key_values = new_caches
    model.past_key_values_data = new_data
    model.current_length_data = new_lengths
    return kept


def compact_for_decode(model, keep: torch.LongTensor | None, max_new_tokens: int, slack: int) -> int:
    """Compact the target cache after prefill and size it for decoding; return the kept length.

    The caller supplies only ``keep`` (``None`` keeps every cached position). The kept length
    comes from ``keep`` itself (its length, or the current cache length for ``None``), and the
    new capacity is ``kept + max_new_tokens + slack``. Decoders derive their offset from the
    returned value, ``kv_offset = prompt_len - kept``, never from a length read beforehand.
    """
    kept = int(model.current_length_data[0]) if keep is None else int(torch.as_tensor(keep).numel())
    return compact_past_key_values(model, keep, kept + int(max_new_tokens) + int(slack))
