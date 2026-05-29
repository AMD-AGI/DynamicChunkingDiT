# Copyright (c) Advanced Micro Devices, Inc.
# Portions copyright (c) 2025 brwa-cartesia, licensed under the MIT License.
# Originally based on https://github.com/goombalab/hnet
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn

from timm.models.vision_transformer import Mlp


class FlashAttention(nn.Module):
    """Multi-head self-attention using flash-attn.

    Supports three input regimes:
      - (B, N, C) with ``mask=None``  -> packed (all tokens attend within each row).
      - (B, N, C) with a (B, N) boolean ``mask``  -> unpad/pad roundtrip.
      - (1, T, C) with ``cu_seqlens`` and ``max_seqlen``  -> batch-packed varlen.
        Each original sample is a contiguous run in T, and attention is confined
        to each run so samples never cross-attend.

    State dict keys: qkv.weight, qkv.bias, proj.weight, proj.bias.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        **kwargs,
    ):
        super().__init__()
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_func
        from flash_attn.bert_padding import unpad_input, pad_input
        self._flash_packed_fn = flash_attn_qkvpacked_func
        self._flash_varlen_fn = flash_attn_varlen_func
        self._unpad = unpad_input
        self._pad = pad_input

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        **kwargs,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        orig_dtype = qkv.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            qkv = qkv.to(torch.bfloat16)

        drop_p = self.attn_drop.p if self.training else 0.

        if cu_seqlens is not None:
            # Batch-packed varlen path. x is (1, T, C); each sample is a contiguous
            # run delimited by cu_seqlens, so we can feed flash-attn directly
            # without any pad/unpad work.
            T = B * N
            qkv_flat = qkv.reshape(T, 3, self.num_heads, self.head_dim)
            q, k, v = qkv_flat[:, 0], qkv_flat[:, 1], qkv_flat[:, 2]
            out = self._flash_varlen_fn(
                q, k, v,
                cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen,
                dropout_p=drop_p,
            )
            x = out.reshape(B, N, C)
        elif mask is None or mask.all():
            x = self._flash_packed_fn(qkv, dropout_p=drop_p)
            x = x.reshape(B, N, C)
        else:
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = self._unpad(q, mask)
            k_unpad, _, cu_seqlens_k, max_seqlen_k, _ = self._unpad(k, mask)
            v_unpad, _, _, _, _ = self._unpad(v, mask)
            out_unpad = self._flash_varlen_fn(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=drop_p,
            )
            x = self._pad(out_unpad, indices_q, B, N)
            x = x.reshape(B, N, C)

        # Record FLOPs for the attention kernel itself (Q@K^T + softmax + Attn@V).
        # flash-attn bypasses aten, so FlopCounterMode doesn't see these ops;
        # ComponentFlopCounter's forward hook reads this value. We must compute
        # it here because the hook can't see cu_seqlens/mask (they're kwargs).
        # For varlen/packed attention, cost is H * sum_i(L_i^2) * (4d + 3); only
        # the full (B, N, N) case uses B * N^2.
        H = self.num_heads
        d = self.head_dim
        per_pair_flops = 4 * d + 3
        if cu_seqlens is not None:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
            total_pairs = (lengths * lengths).sum()
        elif mask is not None and not mask.all():
            lengths = mask.sum(dim=1).to(torch.int64)
            total_pairs = (lengths * lengths).sum()
        else:
            total_pairs = torch.tensor(B * N * N, dtype=torch.int64, device=x.device)
        # Keep as a tensor to avoid a CUDA sync on the hot path; the hook calls
        # .item() only when FLOP counting is active.
        self._last_attn_flops = total_pairs * (H * per_pair_flops)

        x = x.to(orig_dtype)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def create_block(
    arch,
    d_model,
    norm_epsilon=1e-5,
    layer_idx=None,
    residual_in_fp32=True,
    device=None,
    dtype=None,
    dit_kwargs: Optional[dict] = None,
    stage_idx: int = 0,
    output_dim: Optional[int] = None,
    cond_dim: Optional[int] = None,
):
    factory_kwargs = {"device": device, "dtype": dtype}

    if arch in ("d", "D"):
        assert dit_kwargs is not None, "dit_kwargs must be provided for DiT blocks"
        has_mlp = arch == "D"
        return HNetDiTBlock(**dit_kwargs, **factory_kwargs, layer_idx=layer_idx, has_mlp=has_mlp)
    elif arch in ("c", "C"):
        has_mlp = arch == "C"
        use_conv2d = (stage_idx == 0)
        return HNetConvBlock(
            d_model,
            output_dim=output_dim,
            cond_dim=cond_dim,
            has_mlp=has_mlp,
            use_conv2d=use_conv2d,
            **factory_kwargs
        )
    else:
        raise NotImplementedError


def modulate(x, shift, scale):
    """Apply adaLN shift/scale. Accepts either broadcastable 2D (B, D) or
    per-token 3D (B, N, D) modulators, so the same block works for both
    padded (B, N, D) inputs and batch-packed (1, T, D) inputs.
    """
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class HNetDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, has_mlp=True, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = FlashAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        if has_mlp:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            self.norm2 = None
            self.mlp = None
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x,
        residual,
        cond,
        mask=None,
        mixer_kwargs=None,
        **kwargs,
    ):
        mixer_kwargs = mixer_kwargs or {}
        cu_seqlens = mixer_kwargs.get("cu_seqlens")
        max_seqlen = mixer_kwargs.get("max_seqlen")

        # adaLN_modulation is per-sample: Linear(D -> 6D) applied to cond (B, D).
        # For batch-packed inputs (x is (1, T, D)) we must broadcast the result
        # to per-token (1, T, 6D), but we do it AFTER the Linear via
        # repeat_interleave — running the Linear per-token would cost T/B more
        # FLOPs (huge for 28 DiT blocks at D=1152) for no benefit.
        ada = self.adaLN_modulation(cond)  # (B, 6D)
        if cu_seqlens is not None:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]  # (B,)
            ada = torch.repeat_interleave(ada, lengths, dim=0).unsqueeze(0)  # (1, T, 6D)
        else:
            # (B, 6D) -> (B, 1, 6D) so downstream modulators broadcast over N.
            ada = ada.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada.chunk(6, dim=-1)

        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            mask=mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if self.mlp is not None:
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, None


class HNetConvBlock(nn.Module):
    """
    A Stable Diffusion-style ResNet block for H-Net encoder/decoder.
    Supports both Conv2d (for outermost stage with 2D spatial structure) and 
    Conv1d (for inner stages with 1D sequence after chunking).
    """
    def __init__(self, input_dim, output_dim=None, cond_dim=None, has_mlp=True, use_conv2d=True, **kwargs):
        super().__init__()
        output_dim = output_dim or input_dim
        cond_dim = cond_dim or input_dim
        inner_dim = min(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_dim = inner_dim
        self.use_conv2d = use_conv2d
        
        # SD ResBlock: GroupNorm -> SiLU -> Conv -> [+temb] -> GroupNorm -> SiLU -> Conv
        self.norm1 = nn.GroupNorm(32, input_dim, **kwargs)
        self.norm2 = nn.GroupNorm(32, inner_dim, **kwargs)
        self.act = nn.SiLU()
        
        Conv = nn.Conv2d if use_conv2d else nn.Conv1d
        self.conv1 = Conv(input_dim, inner_dim, 3, padding=1, **kwargs)
        self.conv2 = Conv(inner_dim, output_dim, 3, padding=1, **kwargs)
        
        if input_dim != output_dim:
            self.skip_proj = Conv(input_dim, output_dim, 1, **kwargs)
        else:
            self.skip_proj = None
        
        # SD-style: project conditioning and add after first conv
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, inner_dim, **kwargs)
        )
    
    def forward(self, x, residual, cond, mixer_kwargs=None, **kwargs):
        B, L, D = x.shape
        
        if self.use_conv2d:
            if mixer_kwargs is None:
                mixer_kwargs = {}
            num_rows = mixer_kwargs.get("num_rows")
            num_cols = mixer_kwargs.get("num_cols")
            assert num_rows is not None and num_cols is not None, \
                "num_rows and num_cols required for Conv2d blocks"
            
            h = x.view(B, num_rows, num_cols, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        else:
            h = x.transpose(1, 2)  # (B, D, L)
        
        h_skip = h
        
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        if self.use_conv2d:
            temb = self.cond_proj(cond)[:, :, None, None]  # (B, inner_dim, 1, 1)
        else:
            temb = self.cond_proj(cond)[:, :, None]  # (B, inner_dim, 1)
        h = h + temb
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        if self.skip_proj is not None:
            h_skip = self.skip_proj(h_skip)
        h = h + h_skip
        
        if self.use_conv2d:
            x = h.permute(0, 2, 3, 1).view(B, L, self.output_dim)
        else:
            x = h.transpose(1, 2)  # (B, L, output_dim)
        
        return x, None
