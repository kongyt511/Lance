# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
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
# coding: utf-8

import contextlib
import os
from typing import Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention


ATTENTION_BACKEND_ENV = "LANCE_ATTENTION_BACKEND"
BACKEND_AUTO = "auto"
BACKEND_FLASH_ATTN = "flash_attn"
BACKEND_CUDNN_SDPA = "cudnn_sdpa"
BACKEND_SDPA = "sdpa"
BACKEND_FLASH_SDPA = "flash_sdpa"
BACKEND_EFFICIENT_SDPA = "efficient_sdpa"
BACKEND_MATH_SDPA = "math_sdpa"

_FLASH_ATTN_IMPORT_ATTEMPTED = False
_FLASH_ATTN_VARLEN_FUNC = None


def normalize_attention_backend(backend: Optional[str]) -> str:
    backend = (backend or os.getenv(ATTENTION_BACKEND_ENV) or BACKEND_AUTO).strip().lower()
    aliases = {
        "flash": BACKEND_FLASH_ATTN,
        "flash-attn": BACKEND_FLASH_ATTN,
        "flash_attention_2": BACKEND_FLASH_ATTN,
        "fa2": BACKEND_FLASH_ATTN,
        "cudnn": BACKEND_CUDNN_SDPA,
        "cudnn_attention": BACKEND_CUDNN_SDPA,
        "cudnn-attention": BACKEND_CUDNN_SDPA,
        "torch": BACKEND_SDPA,
        "torch_sdpa": BACKEND_SDPA,
        "pytorch_sdpa": BACKEND_SDPA,
        "flash_attention": BACKEND_FLASH_SDPA,
        "flash-sdpa": BACKEND_FLASH_SDPA,
        "efficient": BACKEND_EFFICIENT_SDPA,
        "efficient_attention": BACKEND_EFFICIENT_SDPA,
        "efficient-sdpa": BACKEND_EFFICIENT_SDPA,
        "math": BACKEND_MATH_SDPA,
        "math-sdpa": BACKEND_MATH_SDPA,
    }
    backend = aliases.get(backend, backend)
    valid = {
        BACKEND_AUTO,
        BACKEND_FLASH_ATTN,
        BACKEND_CUDNN_SDPA,
        BACKEND_SDPA,
        BACKEND_FLASH_SDPA,
        BACKEND_EFFICIENT_SDPA,
        BACKEND_MATH_SDPA,
    }
    if backend not in valid:
        raise ValueError(
            f"Unsupported attention backend: {backend}. "
            f"Choose one of: {', '.join(sorted(valid))}."
        )
    return backend


def get_attention_backend(config=None, default: str = BACKEND_AUTO) -> str:
    backend = getattr(config, "attention_backend", None) if config is not None else None
    return normalize_attention_backend(backend or os.getenv(ATTENTION_BACKEND_ENV) or default)


def flash_attn_varlen_available() -> bool:
    global _FLASH_ATTN_IMPORT_ATTEMPTED, _FLASH_ATTN_VARLEN_FUNC
    if not _FLASH_ATTN_IMPORT_ATTEMPTED:
        _FLASH_ATTN_IMPORT_ATTEMPTED = True
        try:
            from flash_attn import flash_attn_varlen_func
        except Exception:
            _FLASH_ATTN_VARLEN_FUNC = None
        else:
            _FLASH_ATTN_VARLEN_FUNC = flash_attn_varlen_func
    return _FLASH_ATTN_VARLEN_FUNC is not None


def get_flash_attn_varlen_func():
    if not flash_attn_varlen_available():
        raise RuntimeError(
            "flash-attn is not available. Install a compatible flash-attn build, "
            "or set --attention_backend auto / cudnn_sdpa / sdpa."
        )
    return _FLASH_ATTN_VARLEN_FUNC


def resolve_auto_backend() -> str:
    return BACKEND_FLASH_ATTN if flash_attn_varlen_available() else BACKEND_CUDNN_SDPA


def _run_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    backend: str = BACKEND_SDPA,
    is_causal: bool = False,
) -> torch.Tensor:
    with sdpa_backend_context(backend):
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )


def _run_auto_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    if flash_attn_varlen_available():
        return _run_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            backend=BACKEND_EFFICIENT_SDPA,
            is_causal=is_causal,
        )
    try:
        return _run_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            backend=BACKEND_CUDNN_SDPA,
            is_causal=is_causal,
        )
    except RuntimeError:
        return _run_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            backend=BACKEND_SDPA,
            is_causal=is_causal,
        )


@contextlib.contextmanager
def sdpa_backend_context(backend: str):
    backend = normalize_attention_backend(backend)
    if backend == BACKEND_AUTO:
        backend = resolve_auto_backend()
    if backend == BACKEND_FLASH_ATTN:
        # This context is used for PyTorch SDPA fallback paths. Keep the old
        # behavior when the main backend is flash-attn.
        backend = BACKEND_EFFICIENT_SDPA

    if backend == BACKEND_CUDNN_SDPA:
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            yield
    elif backend == BACKEND_FLASH_SDPA:
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            yield
    elif backend == BACKEND_EFFICIENT_SDPA:
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            yield
    elif backend == BACKEND_MATH_SDPA:
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            yield
    elif backend == BACKEND_SDPA:
        yield
    else:
        raise ValueError(f"Unsupported SDPA backend: {backend}")


def repeat_kv_for_gqa(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_heads = key_states.shape[1]
    if query_heads == key_heads:
        return key_states, value_states
    if query_heads % key_heads != 0:
        raise ValueError(f"query_heads={query_heads} is not divisible by key_heads={key_heads}")
    num_groups = query_heads // key_heads
    return (
        key_states.repeat_interleave(num_groups, dim=1),
        value_states.repeat_interleave(num_groups, dim=1),
    )


def _bottom_right_causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    q_positions = torch.arange(q_len, device=device) + (k_len - q_len)
    k_positions = torch.arange(k_len, device=device)
    return k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    backend: str = BACKEND_AUTO,
) -> torch.Tensor:
    backend = normalize_attention_backend(backend)
    auto_requested = backend == BACKEND_AUTO
    if backend == BACKEND_AUTO:
        backend = resolve_auto_backend()

    if backend == BACKEND_FLASH_ATTN:
        try:
            return get_flash_attn_varlen_func()(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=causal,
            )
        except RuntimeError:
            if not auto_requested:
                raise
            backend = BACKEND_CUDNN_SDPA

    outputs = []
    num_sequences = cu_seqlens_q.numel() - 1
    query_heads = q.shape[1]
    for index in range(num_sequences):
        q_start = int(cu_seqlens_q[index].item())
        q_end = int(cu_seqlens_q[index + 1].item())
        k_start = int(cu_seqlens_k[index].item())
        k_end = int(cu_seqlens_k[index + 1].item())
        q_i = q[q_start:q_end]
        k_i = k[k_start:k_end]
        v_i = v[k_start:k_end]
        k_i, v_i = repeat_kv_for_gqa(k_i, v_i, query_heads)

        attn_mask = None
        if causal:
            attn_mask = _bottom_right_causal_mask(q_i.shape[0], k_i.shape[0], q_i.device)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        q_i = q_i.transpose(0, 1).unsqueeze(0)
        k_i = k_i.transpose(0, 1).unsqueeze(0)
        v_i = v_i.transpose(0, 1).unsqueeze(0)
        if backend == BACKEND_CUDNN_SDPA:
            try:
                output_i = _run_sdpa(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=attn_mask,
                    backend=BACKEND_CUDNN_SDPA,
                    is_causal=False,
                )
            except RuntimeError:
                output_i = _run_sdpa(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=attn_mask,
                    backend=BACKEND_SDPA,
                    is_causal=False,
                )
        else:
            output_i = _run_sdpa(
                q_i,
                k_i,
                v_i,
                attn_mask=attn_mask,
                backend=backend,
                is_causal=False,
            )
        outputs.append(output_i.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    backend: str = BACKEND_AUTO,
    is_causal: bool = False,
) -> torch.Tensor:
    backend = normalize_attention_backend(backend)
    if backend == BACKEND_AUTO:
        return _run_auto_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
    return _run_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        backend=backend,
        is_causal=is_causal,
    )
