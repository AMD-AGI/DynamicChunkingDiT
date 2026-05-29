# Copyright (c) Advanced Micro Devices, Inc.
# Portions copyright (c) 2025 brwa-cartesia, licensed under the MIT License.
# Originally based on https://github.com/goombalab/hnet
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains utility functions for training.

NOTE: This file is not used inside the HNet package, but contains useful utilities for training the model itself.
"""

import torch

from ..hnet.modules.dc import RoutingModuleOutput


def _switch_loss(avg_prob: torch.Tensor, true_ratio: torch.Tensor, N: float) -> torch.Tensor:
    """Switch-Transformer-style load-balancing loss."""
    return (
        (1 - true_ratio) * (1 - avg_prob) +
        true_ratio * avg_prob * (N - 1)
    ) * N / (N - 1)


def load_balancing_loss(
    router_output: RoutingModuleOutput,
    N: float,
    ratio_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Compute the ratio loss.

    Args:
        router_output: The output of the routing module.
        N: The target downsampling factor (must be > 1).
        ratio_batch_size: The batch size for computing the ratio loss. If None,
                          computes over the local batch.

    Returns:
        A single tensor, the ratio loss.
    """
    tokenized_prob = router_output.boundary_prob[..., -1]
    
    if router_output.boundary_mask_natural is not None:
        boundary_mask = router_output.boundary_mask_natural
    else:
        boundary_mask = router_output.boundary_mask
    batch_size = boundary_mask.shape[0]

    if ratio_batch_size is None:
        avg_prob = tokenized_prob.float().mean()
        true_ratio = boundary_mask.float().mean()
        return _switch_loss(avg_prob, true_ratio, N)

    num_chunks = batch_size // ratio_batch_size
    prob_chunks = tokenized_prob.view(num_chunks, ratio_batch_size, *tokenized_prob.shape[1:])
    mask_chunks = boundary_mask.view(num_chunks, ratio_batch_size, *boundary_mask.shape[1:])

    avg_prob = prob_chunks.float().view(num_chunks, -1).mean(dim=-1)
    true_ratio = mask_chunks.float().view(num_chunks, -1).mean(dim=-1)

    return _switch_loss(avg_prob, true_ratio, N).mean()
