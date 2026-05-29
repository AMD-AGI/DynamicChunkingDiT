# Copyright (c) Advanced Micro Devices, Inc.
# Portions copyright (c) 2025 brwa-cartesia, licensed under the MIT License.
# Originally based on https://github.com/goombalab/hnet
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor
    # Pre-drop argmax mask (identical to boundary_mask when tail_dropping_fraction=0).
    # Used by the ratio loss so the target N is anchored to the natural mask and is
    # unaffected by the sampled drop fraction during multi-budget training.
    boundary_mask_natural: torch.Tensor = None


class RoutingModule(nn.Module):

    def __init__(
        self,
        d_model,
        fp32=True,
        device=None,
        dtype=None,
    ):
        self.d_model = d_model
        self.fp32 = fp32
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Local prediction residuals score which spatial tokens are worth keeping.
        bottleneck = d_model // 4
        self.proj = nn.Conv2d(d_model, bottleneck, kernel_size=1, **factory_kwargs)
        self.context_predictor = nn.Conv2d(
            bottleneck, bottleneck, kernel_size=3, padding=1, **factory_kwargs
        )
        self.residual_scorer = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, **factory_kwargs),
            nn.GroupNorm(8, bottleneck, **factory_kwargs),
            nn.SiLU(),
            nn.Conv2d(bottleneck, 1, 1, **factory_kwargs),
        )
        nn.init.constant_(self.residual_scorer[-1].bias, -1.1)

    def _compute_spatial_predictability_prob(self, hidden_states, num_rows, num_cols):
        B, L, D = hidden_states.shape
        x = hidden_states.transpose(1, 2).reshape(B, D, num_rows, num_cols)
        x = self.proj(x)
        predicted = self.context_predictor(x)
        residual = x - predicted
        keep_logit = self.residual_scorer(residual).view(B, L)
        # .clone() is required: the caller writes boundary_prob[:, 0/-1] = 1.0
        # in-place, and SigmoidBackward needs the original sigmoid output for
        # its gradient computation.
        return torch.sigmoid(keep_logit).clone()

    def forward(self, hidden_states, mask, num_rows=None, num_cols=None, tail_dropping_fraction=0.0):
        assert num_rows is not None and num_cols is not None, \
            "num_rows and num_cols required for spatial-predictability routing"

        ctx = torch.amp.autocast(device_type=hidden_states.device.type, enabled=False) if self.fp32 else nullcontext()
        with ctx:
            if self.fp32:
                hidden_states = hidden_states.float()

            boundary_prob = self._compute_spatial_predictability_prob(
                hidden_states, num_rows, num_cols
            )

            boundary_prob[:, 0] = 1.0
            boundary_prob[:, -1] = 1.0

            boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

            selected_idx = torch.argmax(boundary_prob, dim=-1)

            boundary_mask = selected_idx == 1
            if mask is not None:
                boundary_mask = boundary_mask & mask

            boundary_mask_natural = boundary_mask.clone()

            # Accept either one global drop fraction or per-sample fractions for Lite-CFG.
            is_tensor_fraction = torch.is_tensor(tail_dropping_fraction)
            if is_tensor_fraction or tail_dropping_fraction > 0.0:
                prob1 = boundary_prob[..., 1]
                sort_probs = prob1.masked_fill(~boundary_mask.bool(), float("inf"))
                ranks = sort_probs.argsort(dim=-1).argsort(dim=-1)
                n_boundary = boundary_mask.sum(-1, keepdim=True).long()
                if is_tensor_fraction:
                    frac = tail_dropping_fraction.to(
                        device=boundary_prob.device, dtype=boundary_prob.dtype
                    ).view(-1, 1)
                else:
                    frac = tail_dropping_fraction
                n_drop = (n_boundary.float() * frac).long()
                drop = (ranks < n_drop) & boundary_mask
                boundary_mask = boundary_mask & ~drop

            selected_probs = boundary_prob.gather(
                dim=-1, index=selected_idx.unsqueeze(-1)
            )

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
            boundary_mask_natural=boundary_mask_natural,
        )


class ChunkLayer(nn.Module):

    def __init__(self, fp32=True):
        super().__init__()
        self.fp32 = fp32

    @torch._dynamo.disable
    def forward(self, hidden_states, boundary_mask, boundary_prob, mask):
        device = hidden_states.device

        ctx = torch.amp.autocast(device_type=device.type, enabled=False) if self.fp32 else nullcontext()
        with ctx:
            if self.fp32:
                boundary_prob = boundary_prob.float()

            num_tokens = boundary_mask.sum(dim=-1)
            next_max_seqlen = int(num_tokens.max())

            L = hidden_states.shape[1]

            prob = boundary_prob[..., 1]  # (B, L)

            boundary_mask = boundary_mask.bool()
            non_boundary_prob = prob.clone()
            non_boundary_prob[boundary_mask] = float('-inf')
            non_boundary_rank = torch.argsort(torch.argsort(-non_boundary_prob, dim=1), dim=1)

            token_idx = torch.where(
                boundary_mask,
                torch.arange(L, device=device)[None, :],
                L + non_boundary_rank,
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

        next_hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                -1, -1, hidden_states.shape[-1]
            ),
        )

        next_mask = (
            torch.arange(next_max_seqlen, device=device)[None, :]
            < num_tokens[:, None]
        )

        return next_hidden_states, next_mask, seq_sorted_indices[:, :next_max_seqlen]


def compute_nearest_boundary_idx(boundary_mask):
    B, L = boundary_mask.shape
    device = boundary_mask.device
    
    left_chunk_idx = torch.cumsum(boundary_mask, dim=1) - 1
    
    right_chunk_idx_from_right = torch.cumsum(boundary_mask.flip(1), dim=1).flip(1) - 1
    num_chunks = boundary_mask.sum(dim=1, keepdim=True)
    right_chunk_idx = num_chunks - 1 - right_chunk_idx_from_right
    
    ones = torch.ones(B, L, device=device)
    cumsum_ones = torch.cumsum(ones, dim=1)
    
    boundary_cumsum = cumsum_ones * boundary_mask.float()
    boundary_cumsum_filled = torch.cummax(boundary_cumsum, dim=1)[0]
    dist_to_left = cumsum_ones - boundary_cumsum_filled
    
    cumsum_ones_flip = torch.cumsum(ones.flip(1), dim=1)
    boundary_cumsum_flip = cumsum_ones_flip * boundary_mask.flip(1).float()
    boundary_cumsum_filled_flip = torch.cummax(boundary_cumsum_flip, dim=1)[0]
    dist_to_right = (cumsum_ones_flip - boundary_cumsum_filled_flip).flip(1)
    
    use_right = dist_to_right < dist_to_left
    nearest_chunk_idx = torch.where(use_right, right_chunk_idx, left_chunk_idx)
    
    nearest_chunk_idx = nearest_chunk_idx.clamp(min=0, max=num_chunks.max().item() - 1)
    
    return nearest_chunk_idx.long()


def compute_spatial_nearest_boundary_idx(boundary_mask, num_rows, num_cols):
    B, L = boundary_mask.shape
    device = boundary_mask.device
    
    row_coords = torch.arange(num_rows, device=device).view(-1, 1).expand(-1, num_cols).flatten().float()
    col_coords = torch.arange(num_cols, device=device).view(1, -1).expand(num_rows, -1).flatten().float()
    
    num_boundaries = boundary_mask.sum(dim=1)
    M_max = num_boundaries.max().item()
    
    INF_DIST = float('inf')
    
    sort_key = (~boundary_mask).long() * L + torch.arange(L, device=device).unsqueeze(0)
    sorted_indices = torch.argsort(sort_key, dim=1)
    boundary_indices = sorted_indices[:, :M_max]
    
    boundary_rows = row_coords[boundary_indices]
    boundary_cols = col_coords[boundary_indices]
    
    valid_boundary_mask = torch.arange(M_max, device=device).unsqueeze(0) < num_boundaries.unsqueeze(1)
    
    row_diff = row_coords.view(1, L, 1) - boundary_rows.unsqueeze(1)
    col_diff = col_coords.view(1, L, 1) - boundary_cols.unsqueeze(1)
    sq_dist = row_diff**2 + col_diff**2
    
    sq_dist = sq_dist.masked_fill(~valid_boundary_mask.unsqueeze(1), INF_DIST)
    
    plug_back_idx = torch.argmin(sq_dist, dim=2)
    
    return plug_back_idx


class DeChunkLayer(nn.Module):
    """Dechunk with spatial-kernel smoothing and nearest plug-back."""

    def __init__(
        self,
        d_model,
        plug_back_mode: str = "nearest_2d",
        kernel_sigma: float = 1.0,
        fp32: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.plug_back_mode = plug_back_mode
        self.kernel_sigma = kernel_sigma
        self.fp32 = fp32

    def _smooth_spatial_kernel(self, hidden_states, p, chunk_sort_indices, next_mask, num_rows=None, num_cols=None):
        B, M, D = hidden_states.shape
        
        boundary_positions = chunk_sort_indices.float()
        
        if num_rows is not None and num_cols is not None:
            rows = boundary_positions // num_cols
            cols = boundary_positions % num_cols
            row_diff = rows.unsqueeze(2) - rows.unsqueeze(1)
            col_diff = cols.unsqueeze(2) - cols.unsqueeze(1)
            dist_sq = row_diff**2 + col_diff**2
        else:
            pos_diff = boundary_positions.unsqueeze(2) - boundary_positions.unsqueeze(1)
            dist_sq = pos_diff**2
        
        kernel_weights = torch.exp(-dist_sq / (2 * self.kernel_sigma**2))
        
        confidence_weights = p.unsqueeze(1)
        weights = kernel_weights * confidence_weights
        
        valid_mask_2d = next_mask.unsqueeze(1) & next_mask.unsqueeze(2)
        weights = weights * valid_mask_2d.float()
        
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        smoothed = torch.bmm(weights, hidden_states)
        
        p_self = p.unsqueeze(-1)
        output = p_self * hidden_states + (1 - p_self) * smoothed
        
        return output

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        mask,
        chunk_sort_indices,
        next_mask,
        num_rows=None,
        num_cols=None,
    ):
        original_dtype = hidden_states.dtype

        ctx = torch.amp.autocast(device_type=hidden_states.device.type, enabled=False) if self.fp32 else nullcontext()
        with ctx:
            if self.fp32:
                hidden_states = hidden_states.float()
                boundary_prob = boundary_prob.float()

            B, L = boundary_mask.shape

            p = torch.clamp(boundary_prob[..., -1], min=1e-4, max=1 - (1e-4))

            p = torch.gather(
                p, dim=1, index=chunk_sort_indices
            )

            out = self._smooth_spatial_kernel(
                hidden_states, p, chunk_sort_indices, next_mask, num_rows, num_cols
            )

            if self.plug_back_mode == "nearest_2d":
                assert num_rows is not None and num_cols is not None, \
                    "num_rows and num_cols required for nearest_2d plug_back_mode"
                plug_back_idx = compute_spatial_nearest_boundary_idx(boundary_mask, num_rows, num_cols)
            elif self.plug_back_mode == "nearest_1d":
                plug_back_idx = compute_nearest_boundary_idx(boundary_mask)
            else:
                raise ValueError(f"plug_back_mode must be 'nearest_1d' or 'nearest_2d', got {self.plug_back_mode!r}")

            out = torch.gather(
                out,
                dim=1,
                index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
            )

        return out.to(original_dtype)
