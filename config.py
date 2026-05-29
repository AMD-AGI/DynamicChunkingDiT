# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path
import yaml


@dataclass
class HNetConfig:
    # Architecture layout, e.g. ["c2", ["D12"], "c2"] = 2 conv blocks, 12 DiT blocks (inner), 2 conv blocks (d/D=DiT, c/C=conv)
    arch_layout: List[Union[str, List]] = field(default_factory=lambda: ["c2", ["D12"], "c2"])
    # Hidden dimension per stage
    d_model: List[int] = field(default_factory=lambda: [768, 768])
    # Attention heads per stage (for DiT blocks)
    num_heads: List[int] = field(default_factory=lambda: [12, 12])
    tie_embeddings: bool = False
    # Per-stage dim reduction factor for encoder/decoder conv blocks (None = no reduction)
    # e.g. [4, null] with d_model=[768,768] gives enc/dec dim 192 at stage 0
    enc_dec_dim_factor: Optional[List[Optional[int]]] = None
    # Gaussian sigma for DeChunkLayer spatial smoothing (default 1.0 for 256px; use 2.0 for 512px)
    dechunk_kernel_sigma: float = 1.0
    # Force RoutingModule and DeChunkLayer to run in FP32 (disable autocast) for numerical stability
    fp32_router_dechunk: bool = True


@dataclass
class ModelConfig:
    name: str = "DC-DiT"
    hnet: HNetConfig = field(default_factory=HNetConfig)
    mlp_ratio: float = 4.0


@dataclass
class TrainingConfig:
    epochs: int = 80
    global_batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    class_dropout_prob: float = 0.1
    learn_sigma: bool = True
    ratio_loss_weight: float = 0.03
    ratio_batch_size: Optional[int] = 16
    downsample_factor: Union[float, List[float]] = 4.0
    downsample_factor_start: Optional[Union[float, List[float]]] = None
    downsample_factor_warmup_steps: int = 0
    multi_budget_training: bool = False
    multi_budget_drop_fractions: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])
    multi_budget_start_step: Optional[int] = None
    profile: bool = False
    profile_steps: List[int] = field(default_factory=lambda: [10, 20])
    resume_from_ckpt: Optional[str] = None


@dataclass
class DataConfig:
    feature_path: str = "features"
    image_size: int = 256
    num_classes: int = 1000
    num_workers: int = 32


@dataclass
class DiffusionConfig:
    noise_schedule: str = "linear"
    diffusion_steps: int = 1000


@dataclass
class LoggingConfig:
    results_dir: str = "results"
    log_every: int = 100
    ckpt_every: int = 10000
    wandb_project: str = "Dynamic-Chunking-DiT"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ckpt: Optional[str] = None


#---------------------------------------------------------#
# Helper functions for loading and saving config
#---------------------------------------------------------#

def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def _load_yaml_with_inheritance(path: str) -> dict:
    path = Path(path)
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    if '_base_' in d:
        base = d.pop('_base_')
        if not Path(base).is_absolute():
            base = path.parent / base
        d = _deep_merge(_load_yaml_with_inheritance(str(base)), d)
    return d

def _dict_to_dataclass(cls, data: dict):
    if data is None:
        return cls()
    fields = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in data.items():
        if k in fields:
            ft = fields[k]
            kwargs[k] = _dict_to_dataclass(ft, v) if hasattr(ft, '__dataclass_fields__') else v
    return cls(**kwargs)

def load_config(config_path: str) -> Config:
    d = _load_yaml_with_inheritance(config_path)
    if not d:
        return Config()
    return Config(
        model=_dict_to_dataclass(ModelConfig, d.get('model')),
        training=_dict_to_dataclass(TrainingConfig, d.get('training')),
        data=_dict_to_dataclass(DataConfig, d.get('data')),
        diffusion=_dict_to_dataclass(DiffusionConfig, d.get('diffusion')),
        logging=_dict_to_dataclass(LoggingConfig, d.get('logging')),
        ckpt=d.get('ckpt'),
    )

def save_config(config: Config, config_path: str) -> None:
    from dataclasses import asdict
    with open(config_path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
