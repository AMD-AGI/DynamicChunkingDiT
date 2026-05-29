# Copyright (c) Advanced Micro Devices, Inc.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import torch


# Only to maintain backwards compatibility with old checkpoints.
# This unpickler stubs any missing class so torch.load can still deserialize them.
class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except AttributeError:
            return type(name, (), {"__init__": lambda self, *a, **kw: None})


def load_checkpoint(path, **kwargs):
    """torch.load wrapper that tolerates removed classes in old checkpoints."""
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except AttributeError:
        _m = type("_m", (), {"Unpickler": _CompatUnpickler, "UnpicklingError": pickle.UnpicklingError})
        return torch.load(path, weights_only=False, pickle_module=_m, **kwargs)


def find_model(model_name):
    """Load EMA weights from a DiT checkpoint."""
    assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
    ckpt = load_checkpoint(model_name, map_location=lambda storage, loc: storage)
    return ckpt["ema"] if "ema" in ckpt else ckpt
