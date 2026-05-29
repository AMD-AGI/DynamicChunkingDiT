# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
FLOP counting utilities using PyTorch's native FlopCounterMode, plus a
forward-hook tracker for custom ops (FlashAttention) that bypass aten and
aren't visible to FlopCounterMode.
"""
import torch
from torch.utils.flop_counter import FlopCounterMode
from typing import Optional
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class FlopBreakdown:
    """Per-component FLOP breakdown for a single forward pass."""
    total: int = 0
    embeddings: int = 0  # x_embedder + t_embedder + y_embedder
    final_layer: int = 0
    # HNet components (aggregated across stages)
    encoder: int = 0
    routing: int = 0
    chunk: int = 0
    main_network: int = 0
    dechunk: int = 0
    decoder: int = 0
    
    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "embeddings": self.embeddings,
            "final_layer": self.final_layer,
            "encoder": self.encoder,
            "routing": self.routing,
            "chunk": self.chunk,
            "main_network": self.main_network,
            "dechunk": self.dechunk,
            "decoder": self.decoder,
        }
    
    def __add__(self, other: "FlopBreakdown") -> "FlopBreakdown":
        return FlopBreakdown(
            total=self.total + other.total,
            embeddings=self.embeddings + other.embeddings,
            final_layer=self.final_layer + other.final_layer,
            encoder=self.encoder + other.encoder,
            routing=self.routing + other.routing,
            chunk=self.chunk + other.chunk,
            main_network=self.main_network + other.main_network,
            dechunk=self.dechunk + other.dechunk,
            decoder=self.decoder + other.decoder,
        )


class CustomOpFlopTracker:
    """
    Tracks FLOPs for custom ops (e.g. FlashAttention) using forward hooks.
    """
    
    def __init__(self):
        self._custom_flops = 0
        self._hooks = []
    
    def _attention_hook(self, module, input, output):
        """Hook to count FlashAttention FLOPs (only the attention kernel, not projections).

        FlashAttention.forward computes the exact attention-kernel FLOPs for the
        current call (accounting for varlen / batch-packed sequences where cost
        is H * sum_i(L_i^2) * (4d + 3), not B * N^2) and stashes the result on
        the module. We just read it here — the hook can't see ``cu_seqlens`` /
        ``mask`` directly because those are keyword args.
        """
        flops = getattr(module, '_last_attn_flops', None)
        if flops is None:
            return
        if torch.is_tensor(flops):
            flops = int(flops.item())
        self._custom_flops += int(flops)
    
    def register_hooks(self, model: torch.nn.Module):
        """Register hooks on FlashAttention and other custom modules."""
        for name, module in model.named_modules():
            if type(module).__name__ == 'FlashAttention':
                hook = module.register_forward_hook(self._attention_hook)
                self._hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def get_flops(self) -> int:
        return self._custom_flops
    
    def reset(self):
        self._custom_flops = 0


class ComponentFlopCounter:
    """
    Tracks FLOPs per named component during a forward pass.
    Includes both native PyTorch ops (via FlopCounterMode) and custom ops (via hooks).
    
    Usage:
        counter = ComponentFlopCounter()
        with counter.track("encoder"):
            output = encoder(x)
        with counter.track("decoder"):
            output = decoder(output)
        breakdown = counter.get_breakdown()
    """
    
    def __init__(self):
        self._components: dict[str, int] = {}
        self._current_counter: Optional[FlopCounterMode] = None
        self._custom_tracker: Optional[CustomOpFlopTracker] = None
        self._model: Optional[torch.nn.Module] = None
    
    def set_model(self, model: torch.nn.Module):
        """Set the model to track custom ops for."""
        self._model = model
    
    @contextmanager
    def track(self, component_name: str):
        """Track FLOPs for a named component (native + custom ops)."""
        # Native ops counter
        counter = FlopCounterMode(display=False)
        
        # Custom ops tracker (if model is set)
        custom_tracker = None
        if self._model is not None:
            custom_tracker = CustomOpFlopTracker()
            custom_tracker.register_hooks(self._model)
        
        try:
            with counter:
                yield
        finally:
            # Get native FLOPs
            native_flops = counter.get_total_flops()
            
            # Get custom FLOPs
            custom_flops = 0
            if custom_tracker is not None:
                custom_flops = custom_tracker.get_flops()
                custom_tracker.remove_hooks()
            
            total_flops = native_flops + custom_flops
            self._components[component_name] = self._components.get(component_name, 0) + total_flops
    
    def add_flops(self, component_name: str, flops: int):
        """Manually add FLOPs to a component."""
        self._components[component_name] = self._components.get(component_name, 0) + flops
    
    def get_component_flops(self, component_name: str) -> int:
        """Get FLOPs for a specific component."""
        return self._components.get(component_name, 0)
    
    def get_all_components(self) -> dict[str, int]:
        """Get all component FLOPs."""
        return dict(self._components)
    
    def get_total_flops(self) -> int:
        """Get total FLOPs across all components."""
        return sum(self._components.values())
    
    def to_breakdown(self) -> FlopBreakdown:
        """Convert to FlopBreakdown dataclass."""
        return FlopBreakdown(
            total=self.get_total_flops(),
            embeddings=self._components.get("embeddings", 0),
            final_layer=self._components.get("final_layer", 0),
            encoder=self._components.get("encoder", 0),
            routing=self._components.get("routing", 0),
            chunk=self._components.get("chunk", 0),
            main_network=self._components.get("main_network", 0),
            dechunk=self._components.get("dechunk", 0),
            decoder=self._components.get("decoder", 0),
        )
    
    def reset(self):
        """Reset all counters."""
        self._components.clear()


def format_flops(flops: int) -> str:
    """Format FLOPs count to human-readable string."""
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    else:
        return f"{flops} FLOPs"

