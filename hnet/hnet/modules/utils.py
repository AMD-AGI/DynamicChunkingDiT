# Copyright (c) Advanced Micro Devices, Inc.
# Portions copyright (c) 2025 brwa-cartesia, licensed under the MIT License.
# Originally based on https://github.com/goombalab/hnet
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict

import torch


def get_stage_cfg(cfg, stage_idx):
    return {
        k: v[stage_idx] if isinstance(v, list) else v for k, v in asdict(cfg).items()
    }


def apply_optimization_params(
    param: torch.Tensor,
    **kwargs,
) -> None:
    """
    Annotates a parameter with optimization parameters.

    Specifically, updates the parameter's `_optim` attribute with the given kwargs.
    """

    if hasattr(param, "_optim"):
        param._optim.update(kwargs)
    else:
        param._optim = kwargs
