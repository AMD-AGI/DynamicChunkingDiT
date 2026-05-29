# Copyright (c) Advanced Micro Devices, Inc.
# Portions copyright (c) 2025 brwa-cartesia, licensed under the MIT License.
# Originally based on https://github.com/goombalab/hnet
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.isotropic import Isotropic
from ..modules.dc import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
)

from config import HNetConfig
from contextlib import nullcontext


def _pack_sequence(hidden_states, mask, pos_embed):
    """Pack a padded (B, M, D) sequence into a (1, T, D) batch-packed tensor,
    where T = mask.sum(). Returns the packed hidden states, per-token pos_embed
    (or None), cu_seqlens (int32, (B+1,)), and max_seqlen.

    ``cond`` is NOT packed here. It stays as (B, cond_dim) and flows through
    the inner network unchanged — each DiT block runs adaLN_modulation
    per-sample (a Linear on (B, cond_dim)) and only expands the modulators to
    per-token via repeat_interleave. Packing cond would run adaLN per-token,
    costing ~T/B more FLOPs per block with no quality benefit.
    """
    lengths = mask.sum(dim=1)  # (B,)
    # cu_seqlens must be int32 for flash-attn's varlen API.
    cu_seqlens = F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
    max_seqlen = int(lengths.max().item())

    # Packed views: boolean indexing with mask produces the contiguous run of
    # valid tokens, in the same per-sample order as ``lengths``.
    packed_hidden = hidden_states[mask].unsqueeze(0)

    packed_pos_embed = None
    if pos_embed is not None:
        # pos_embed is (B, M, D) at this point (chunked in HNet.forward).
        packed_pos_embed = pos_embed[mask].unsqueeze(0)

    return packed_hidden, packed_pos_embed, cu_seqlens, max_seqlen


def _unpack_sequence(packed, mask, max_seqlen):
    """Inverse of ``_pack_sequence`` for hidden states. Scatters the packed
    (1, T, D) output back into a (B, M, D) tensor where M = max_seqlen. Padded
    positions are zero; DeChunkLayer ignores them via next_mask.
    """
    B, M = mask.shape[0], max_seqlen
    D = packed.shape[-1]
    out = packed.new_zeros((B, M, D))
    out[mask] = packed.squeeze(0)
    return out


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x

def ste_func(x):
    return STE.apply(x)


class HNet(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        stage_idx: int,
        dit_kwargs: dict,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]
        
        # enc_dec_dim: the reduced dimension this stage's encoder/decoder operate at
        factor = None
        if config.enc_dec_dim_factor and stage_idx < len(config.enc_dec_dim_factor):
            factor = config.enc_dec_dim_factor[stage_idx]
        self.enc_dec_dim = self.d_model // factor if factor else self.d_model

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]

        assert isinstance(arch_layout, list), f"Wrong arch_layout: {arch_layout}"
        if len(arch_layout) == 3:
            sub_model_names = ["encoder", "main_network", "decoder"]
            self.is_innermost = False
        elif len(arch_layout) == 1:
            sub_model_names = ["main_network"]
            self.is_innermost = True
        else:
            raise NotImplementedError

        # routing_dim: dimension at which routing/chunk/dechunk operate.
        # Equals the next stage's enc_dec_dim so tokens flow between stages
        # at the receiving stage's reduced dimension.
        if not self.is_innermost:
            next_idx = stage_idx + 1
            next_factor = None
            if config.enc_dec_dim_factor and next_idx < len(config.enc_dec_dim_factor):
                next_factor = config.enc_dec_dim_factor[next_idx]
            next_d_model = config.d_model[next_idx]
            self.routing_dim = next_d_model // next_factor if next_factor else next_d_model

        for _name, _layout in zip(sub_model_names, arch_layout):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = Isotropic
                _stage_idx = stage_idx
                _pos_idx = None
                _dim_kwargs = {}
                if _name == "encoder":
                    _pos_idx = 0
                    _dim_kwargs = {"input_dim": self.enc_dec_dim, "output_dim": self.routing_dim}
                elif self.is_innermost:
                    # if innermost, then len(layer_layout) == 1
                    _pos_idx = 0
                elif _name == "decoder":
                    _pos_idx = 2
                    _dim_kwargs = {"input_dim": self.routing_dim, "output_dim": self.enc_dec_dim}
                _pos_idx_dict = {"pos_idx": _pos_idx}
            else:
                SubModel = HNet
                _stage_idx = stage_idx + 1
                _pos_idx_dict = {}
                _dim_kwargs = {}

            _sub_model = SubModel(
                config=config,
                stage_idx=_stage_idx,
                dit_kwargs=dit_kwargs,
                **_pos_idx_dict,
                **_dim_kwargs,
                **factory_kwargs,
            )
            self.add_module(_name, _sub_model)

        if not self.is_innermost:
            self.routing_module = RoutingModule(
                self.routing_dim,
                fp32=config.fp32_router_dechunk,
                **factory_kwargs
            )
            self.chunk_layer = ChunkLayer(fp32=config.fp32_router_dechunk)
            # Outermost stage uses 2D spatial plug-back; inner stages use 1D
            plug_back_mode = "nearest_2d" if stage_idx == 0 else "nearest_1d"
            self.dechunk_layer = DeChunkLayer(
                self.routing_dim,
                plug_back_mode=plug_back_mode,
                kernel_sigma=config.dechunk_kernel_sigma,
                fp32=config.fp32_router_dechunk,
            )

            # do the residual in fp32
            self.residual_proj = nn.Linear(
                self.routing_dim, self.routing_dim, device=device, dtype=torch.float32
            )
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True

            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = nn.Parameter(
                torch.zeros(
                    self.d_model - config.d_model[stage_idx - 1], **factory_kwargs
                )
            )
        else:
            self.pad_dimension = None

    def forward(
        self,
        hidden_states,
        mask,
        cond=None,
        pos_embed=None,
        flop_counter=None,
        **mixer_kwargs,
    ):
        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))), dim=-1
            )

        if self.is_innermost:
            if pos_embed is not None:
                hidden_states = hidden_states + pos_embed
            ctx = flop_counter.track("main_network") if flop_counter else nullcontext()
            with ctx:
                hidden_states = self.main_network(
                    hidden_states,
                    mask=mask,
                    cond=cond,
                    **mixer_kwargs,
                )
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        # Encoder
        ctx = flop_counter.track("encoder") if flop_counter else nullcontext()
        with ctx:
            hidden_states = self.encoder(
                hidden_states,
                mask=mask,
                cond=cond,
                **mixer_kwargs,
            )

        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        # Routing
        ctx = flop_counter.track("routing") if flop_counter else nullcontext()
        with ctx:
            bpred_output = self.routing_module(
                hidden_states,
                mask=mask,
                num_rows=mixer_kwargs.get("num_rows"),
                num_cols=mixer_kwargs.get("num_cols"),
                tail_dropping_fraction=mixer_kwargs.get("tail_dropping_fraction", 0.0),
            )
        
        # Chunk
        ctx = flop_counter.track("chunk") if flop_counter else nullcontext()
        with ctx:
            hidden_states, next_mask, chunk_sort_indices = self.chunk_layer(
                hidden_states, bpred_output.boundary_mask, bpred_output.boundary_prob, mask=mask
            )

        # Chunk pos_embed using the same indices ChunkLayer used
        next_pos_embed = None
        if pos_embed is not None:
            B = hidden_states.shape[0]
            next_pos_embed = torch.gather(
                pos_embed.expand(B, -1, -1), dim=1,
                index=chunk_sort_indices.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1]),
            )

        # After chunking, the sequence no longer has the original 2D grid structure.
        # Clear num_rows/num_cols for inner blocks.
        inner_mixer_kwargs = {
            k: v for k, v in mixer_kwargs.items() 
            if k not in ("num_rows", "num_cols")
        }

        # Batch-pack the chunked sequences so the main network only sees real
        # tokens (no padding). Each sample's valid tokens become a contiguous
        # run in a single length-T sequence, with cu_seqlens describing the
        # boundaries for flash-attn's varlen kernel. Linear/MLP/LayerNorm layers
        # are pointwise and operate on the packed tensor for free; attention
        # uses cu_seqlens to keep samples from cross-attending.
        B, next_M, D_inner = hidden_states.shape
        packed_hidden, packed_pos_embed, cu_seqlens, max_seqlen = \
            _pack_sequence(hidden_states, next_mask, next_pos_embed)

        packed_mixer_kwargs = dict(inner_mixer_kwargs)
        packed_mixer_kwargs["cu_seqlens"] = cu_seqlens
        packed_mixer_kwargs["max_seqlen"] = max_seqlen

        # Main network (recursive) — runs on the packed (1, T, D) tensor.
        # cond stays (B, cond_dim): adaLN inside each DiT block is per-sample,
        # expanded to per-token only post-Linear (see HNetDiTBlock.forward).
        packed_out, prev_boundary_predictions = self.main_network(
            packed_hidden,
            mask=None,
            cond=cond,
            pos_embed=packed_pos_embed,
            flop_counter=flop_counter,
            **packed_mixer_kwargs,
        )

        # Unpack back to (B, next_max_seqlen, D). Padded positions are left as
        # zeros; they're never read because DeChunkLayer masks/plugs-back only
        # valid tokens using chunk_sort_indices and next_mask.
        hidden_states = _unpack_sequence(packed_out, next_mask, next_M)

        # Dechunk
        ctx = flop_counter.track("dechunk") if flop_counter else nullcontext()
        with ctx:
            hidden_states = self.dechunk_layer(
                hidden_states,
                bpred_output.boundary_mask,
                bpred_output.boundary_prob,
                mask=mask,
                chunk_sort_indices=chunk_sort_indices,
                next_mask=next_mask,
                num_rows=mixer_kwargs.get("num_rows"),
                num_cols=mixer_kwargs.get("num_cols"),
            )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, bpred_output.selected_probs
        ).to(hidden_states.dtype)

        # Decoder
        ctx = flop_counter.track("decoder") if flop_counter else nullcontext()
        with ctx:
            hidden_states = self.decoder(
                hidden_states,
                mask=mask,
                cond=cond,
                **mixer_kwargs,
            )

        hidden_states = hidden_states[..., :D]
        return hidden_states, [bpred_output, *prev_boundary_predictions]
