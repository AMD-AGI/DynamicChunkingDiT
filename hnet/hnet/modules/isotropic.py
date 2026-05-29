# Copyright (c) Advanced Micro Devices, Inc.
# Portions copyright (c) 2025 brwa-cartesia, licensed under the MIT License.
# Originally based on https://github.com/goombalab/hnet
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import torch.nn as nn

from flash_attn.ops.triton.layer_norm import RMSNorm

from ..modules.block import create_block

from config import HNetConfig


class Isotropic(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        pos_idx: int,
        stage_idx: int,
        dit_kwargs: dict,
        input_dim: int = None,
        output_dim: int = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.stage_idx = stage_idx
        self.d_model = config.d_model[self.stage_idx]
        self.num_heads = config.num_heads[self.stage_idx]

        # Per-block dim control for enc/dec with reduced hidden dim
        input_dim = input_dim if input_dim is not None else self.d_model
        output_dim = output_dim if output_dim is not None else self.d_model

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]
        layout_parse = re.findall(r"([tTdDcC])(\d+)", arch_layout)

        # Count total blocks to determine per-block dims
        total_blocks = sum(int(n) for _, n in layout_parse)

        # Update dit_kwargs with this stage's hidden_size and num_heads
        stage_dit_kwargs = dict(dit_kwargs) if dit_kwargs else {}
        stage_dit_kwargs["hidden_size"] = self.d_model
        stage_dit_kwargs["num_heads"] = self.num_heads

        layers = []
        layer_idx = 0
        self.arch_full = []

        # self.height counts the number of things that get added to the residual stream
        self.height = 0
        for arch, n_layer in layout_parse:
            assert arch in ("d", "D", "c", "C")
            assert n_layer.isdigit()
            for i in range(int(n_layer)):
                global_idx = layer_idx + i
                # Compute per-block input/output dims for conv blocks
                block_input_dim, block_output_dim, block_cond_dim = self._compute_block_dims(
                    global_idx, total_blocks, input_dim, output_dim, pos_idx, self.d_model
                )
                layers.append(
                    create_block(
                        arch,
                        block_input_dim,
                        layer_idx=global_idx,
                        dit_kwargs=stage_dit_kwargs,
                        stage_idx=stage_idx,
                        output_dim=block_output_dim if arch in ("c", "C") else None,
                        cond_dim=block_cond_dim if arch in ("c", "C") else None,
                        **factory_kwargs,
                    )
                )
            if arch.islower():
                self.height += int(n_layer)
            else:
                self.height += 2 * int(n_layer)
            self.arch_full.extend([arch for _ in range(int(n_layer))])
            layer_idx += int(n_layer)

        self.layers = nn.ModuleList(layers)

        self.rmsnorm = RMSNorm(output_dim, eps=1e-5, **factory_kwargs)

    @staticmethod
    def _compute_block_dims(block_idx, total_blocks, input_dim, output_dim, pos_idx, d_model):
        """Compute (block_input_dim, block_output_dim, cond_dim) for a block based on its position.
        
        For encoder (pos_idx=0): input_dim=enc_dec_dim, output_dim=routing_dim
          - All blocks at input_dim, last block transitions to output_dim
        For decoder (pos_idx=2): input_dim=routing_dim, output_dim=enc_dec_dim
          - First block transitions from input_dim to output_dim, rest at output_dim
        """
        cond_dim = d_model  # conditioning is always at full d_model
        if input_dim == output_dim:
            return input_dim, output_dim, cond_dim
        if pos_idx == 0:  # encoder: last block transitions
            if block_idx == total_blocks - 1:
                return input_dim, output_dim, cond_dim
            return input_dim, input_dim, cond_dim
        else:  # decoder: first block transitions, rest at output_dim
            if block_idx == 0:
                return input_dim, output_dim, cond_dim
            return output_dim, output_dim, cond_dim

    def forward(
        self,
        hidden_states,
        mask,
        cond=None,
        **mixer_kwargs,
    ):
        assert (
            hidden_states.dim() == 3
        ), "Hidden states must be (B, L, D)"

        # Shallow copy: mixer_kwargs can contain tensors (e.g. cu_seqlens) and
        # we don't want to clone them per-forward. We never mutate this dict.
        attn_mixer_kwargs = dict(mixer_kwargs)

        residual = None
        for layer, arch in zip(self.layers, self.arch_full):
            if arch in ("t", "T", "d", "D"):
                layer_mixer_kwargs = attn_mixer_kwargs
            elif arch in ("c", "C"):
                layer_mixer_kwargs = attn_mixer_kwargs
            else:
                raise NotImplementedError(f"Unsupported arch: {arch}")
                
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cond=cond,
                mask=mask,
                mixer_kwargs=layer_mixer_kwargs,
            )

        # Setting prenorm=False ignores the residual
        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )

        return hidden_states
