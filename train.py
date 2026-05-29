# Copyright (c) Advanced Micro Devices, Inc.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator
import wandb

from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from models import create_dit
from diffusion import create_diffusion
from config import load_config, Config, save_config
from utils import load_checkpoint
from hnet.utils.train import load_balancing_loss


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def _strip_prefix(name):
    """Strip all wrapper prefixes from parameter names."""
    prefixes = ("module.", "_orig_mod.", "_fsdp_wrapped_module.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = {_strip_prefix(k): v for k, v in ema_model.named_parameters()}
    for name, param in model.named_parameters():
        name = _strip_prefix(name)
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def _build_wandb_config(config: Config, accelerator: Accelerator, experiment_dir: str, resumed_from: str = None) -> dict:
    wandb_cfg = {
        "model": config.model.name,
        "hnet_config": str(config.model.hnet),
        "class_dropout_prob": config.training.class_dropout_prob,
        "learn_sigma": config.training.learn_sigma,
        "image_size": config.data.image_size,
        "num_classes": config.data.num_classes,
        "epochs": config.training.epochs,
        "global_batch_size": config.training.global_batch_size,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "ema_decay": config.training.ema_decay,
        "num_workers": config.data.num_workers,
        "log_every": config.logging.log_every,
        "ckpt_every": config.logging.ckpt_every,
        "max_grad_norm": config.training.max_grad_norm,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "ratio_loss_weight": config.training.ratio_loss_weight,
        "ratio_batch_size": config.training.ratio_batch_size,
        "downsample_factor": config.training.downsample_factor,
        "feature_path": config.data.feature_path,
        "results_dir": config.logging.results_dir,
        "experiment_dir": experiment_dir,
        "num_processes": accelerator.num_processes,
        "diffusion_steps": config.diffusion.diffusion_steps,
        "noise_schedule": config.diffusion.noise_schedule,
        "profile": config.training.profile,
        "profile_steps": config.training.profile_steps,
        "dechunk_kernel_sigma": config.model.hnet.dechunk_kernel_sigma,
        "fp32_router_dechunk": config.model.hnet.fp32_router_dechunk,
        "multi_budget_training": config.training.multi_budget_training,
        "multi_budget_drop_fractions": config.training.multi_budget_drop_fractions,
        "multi_budget_start_step": config.training.multi_budget_start_step,
    }
    if resumed_from is not None:
        wandb_cfg["resumed_from"] = resumed_from
    return wandb_cfg


def _init_wandb_run(
    config: Config,
    accelerator: Accelerator,
    experiment_dir: str,
    run_name: str,
    resumed_from: str = None,
) -> None:
    wandb.init(
        project=config.logging.wandb_project,
        name=run_name,
        config=_build_wandb_config(config, accelerator, experiment_dir, resumed_from=resumed_from),
        dir=experiment_dir,
    )

class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features).squeeze(dim=0), torch.from_numpy(labels).squeeze(dim=0)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(config: Config, resume_from: str = None, checkpoint_path: str = None, wandb_run_id: str = None):
    """
    Trains a new DiT model, or resumes training from an existing experiment.
    
    Args:
        config: Training configuration
        resume_from: Path to experiment directory to resume from (optional)
        checkpoint_path: Path to checkpoint file to load (optional, used with resume_from)
        wandb_run_id: Wandb run ID to resume (optional, used with resume_from)
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    device = accelerator.device

    # Setup an experiment folder:
    if resume_from:
        # Resume from existing experiment
        experiment_dir = resume_from
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        if accelerator.is_main_process:
            logger = create_logger(experiment_dir)
            logger.info(f"Resuming experiment from {experiment_dir}")
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            if wandb_run_id:
                # Resume existing wandb run
                logger.info(f"Resuming wandb run {wandb_run_id}")
                wandb.init(
                    project=config.logging.wandb_project,
                    id=wandb_run_id,
                    resume="must",
                    dir=experiment_dir,
                )
            else:
                # Create new wandb run for resumed training
                model_string_name = config.model.name.replace("/", "-")
                experiment_name = os.path.basename(experiment_dir)
                logger.info(f"Creating new wandb run for resumed training")
                _init_wandb_run(
                    config=config,
                    accelerator=accelerator,
                    experiment_dir=experiment_dir,
                    run_name=f"{experiment_name}-resumed",
                    resumed_from=checkpoint_path,
                )
    else:
        # Create new experiment
        if accelerator.is_main_process:
            os.makedirs(config.logging.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            experiment_index = len(glob(f"{config.logging.results_dir}/*"))
            model_string_name = config.model.name.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
            experiment_dir = f"{config.logging.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
            checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")
            
            # Save config to experiment directory
            save_config(config, f"{experiment_dir}/config.yaml")
            logger.info(f"Configuration saved to {experiment_dir}/config.yaml")
            
            # Initialize wandb
            _init_wandb_run(
                config=config,
                accelerator=accelerator,
                experiment_dir=experiment_dir,
                run_name=f"{experiment_index:03d}-{model_string_name}",
            )


    # Create model:
    assert config.data.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = config.data.image_size // 8
    in_channels = 4
    model = create_dit(
        model_config=config.model,
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=config.data.num_classes,
        class_dropout_prob=config.training.class_dropout_prob,
        learn_sigma=config.training.learn_sigma,
    )
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule=config.diffusion.noise_schedule,
        diffusion_steps=config.diffusion.diffusion_steps,
        learn_sigma=config.training.learn_sigma,
    )
    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"DiT Trainable Parameters: {num_params:,}")
        wandb.config.update({"num_parameters": num_params})

    # Setup optimizer:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=config.training.weight_decay,
    )

    # Setup data:
    features_dir = f"{config.data.feature_path}/imagenet{config.data.image_size}_features"
    labels_dir = f"{config.data.feature_path}/imagenet{config.data.image_size}_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(config.training.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        dataset_size = len(dataset)
        logger.info(f"Dataset contains {dataset_size:,} images ({config.data.feature_path})")
        wandb.config.update({"dataset_size": dataset_size})

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    requires_grad(ema, False)

    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Load checkpoint if resuming
    train_steps = 0
    start_epoch = 0
    if checkpoint_path is not None:
        if accelerator.is_main_process:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        
        # Load model state
        model_state = checkpoint["model"]
        accelerator.unwrap_model(model).load_state_dict(model_state)
        
        # Load EMA state
        ema_state = checkpoint["ema"]
        accelerator.unwrap_model(ema).load_state_dict(ema_state)
        
        # Load optimizer state
        opt.load_state_dict(checkpoint["opt"])
        
        # Extract step number from checkpoint filename (format: 0050000.pt)
        train_steps = int(os.path.basename(checkpoint_path).split('.')[0])
        
        # Estimate epoch from steps (approximate, will be refined during training)
        steps_per_epoch = len(loader) // config.training.gradient_accumulation_steps
        start_epoch = train_steps // steps_per_epoch
        
        if accelerator.is_main_process:
            logger.info(f"Resumed from step {train_steps}, starting at epoch {start_epoch}")
        
        del checkpoint  # Free memory

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_diffusion_loss = 0
    running_ratio_loss = 0
    running_compression_ratio = 0.0
    running_compression_ratio_std = 0.0
    running_compression_ratio_min = float('inf')
    running_compression_ratio_max = 0.0
    # Multi-budget metrics (only meaningful when multi_budget_training is on)
    running_effective_compression_ratio = 0.0
    running_tail_drop_fraction = 0.0
    start_time = time()
    
    # Setup profiler if enabled
    profile_steps = config.training.profile_steps
    if len(profile_steps) == 1:
        profile_start, profile_end = profile_steps[0], profile_steps[0]
    else:
        profile_start, profile_end = profile_steps[0], profile_steps[1]
    
    profiler = None
    if config.training.profile:
        profile_dir = f"{experiment_dir}/profiler" if accelerator.is_main_process else None
        if profile_dir:
            os.makedirs(profile_dir, exist_ok=True)
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=max(0, profile_start - 1),
                warmup=1,
                active=profile_end - profile_start + 1,
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(profile_dir, use_gzip=True) if profile_dir else None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        if accelerator.is_main_process:
            logger.info(f"Profiling enabled for steps {profile_start} to {profile_end}")
            logger.info(f"Profiler traces will be saved to {profile_dir}")
    
    if accelerator.is_main_process:
        logger.info(f"Training for {config.training.epochs} epochs (starting from epoch {start_epoch})...")
        if not resume_from:
            wandb.log({"epoch": 0, "train_steps": 0})  # Initialize epoch tracking only for new runs
    
    # Resolve the step at which multi-budget sampling begins.
    # Default: after the N-curriculum warmup, so the router first learns its target
    # ratio before being asked to importance-rank its boundaries.
    _mb_start_step = config.training.multi_budget_start_step
    if _mb_start_step is None:
        _mb_start_step = config.training.downsample_factor_warmup_steps
    _mb_fractions = config.training.multi_budget_drop_fractions
    
    if profiler:
        profiler.__enter__()
    
    for epoch in range(start_epoch, config.training.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            with accelerator.accumulate(model):
                x = x.to(device)
                y = y.to(device)
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

                # Multi-budget training: sample a tail-drop fraction uniformly from
                # the discrete set, using train_steps as the seed so every rank picks
                # the same index without a cross-rank broadcast.
                tail_drop_f = 0.0
                if config.training.multi_budget_training and train_steps >= _mb_start_step:
                    g = torch.Generator()
                    g.manual_seed(train_steps)
                    idx = torch.randint(0, len(_mb_fractions), (1,), generator=g).item()
                    tail_drop_f = _mb_fractions[idx]

                model_kwargs = dict(y=y, tail_dropping_fraction=tail_drop_f)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                diffusion_loss = loss_dict["loss"].mean()
                
                # Compute ratio loss from cached boundary predictions
                ratio_loss = torch.tensor(0.0, device=device)
                if config.training.ratio_loss_weight > 0:
                    downsample_factors = config.training.downsample_factor
                    if not isinstance(downsample_factors, list):
                        downsample_factors = [downsample_factors]

                    # Curriculum: linearly ramp N from start to target
                    if config.training.downsample_factor_start is not None and config.training.downsample_factor_warmup_steps > 0:
                        start_factors = config.training.downsample_factor_start
                        if not isinstance(start_factors, list):
                            start_factors = [start_factors] * len(downsample_factors)
                        progress = min(1.0, train_steps / config.training.downsample_factor_warmup_steps)
                        effective_factors = [
                            s + (e - s) * progress
                            for s, e in zip(start_factors, downsample_factors)
                        ]
                    else:
                        effective_factors = downsample_factors

                    assert all(n > 1 for n in effective_factors), "All downsample factors must be greater than 1"
                    unwrapped_model = accelerator.unwrap_model(model)
                    if unwrapped_model._last_boundary_predictions:
                        for i, bpred in enumerate(unwrapped_model._last_boundary_predictions):
                            N = effective_factors[i]
                            ratio_loss = ratio_loss + load_balancing_loss(bpred, N=N, ratio_batch_size=config.training.ratio_batch_size)
                        ratio_loss = ratio_loss / len(unwrapped_model._last_boundary_predictions)
                
                loss = diffusion_loss + config.training.ratio_loss_weight * ratio_loss
                accelerator.backward(loss)
                
                # Only step optimizer and update EMA when gradients are synced
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    update_ema(ema, model, decay=config.training.ema_decay)
                    train_steps += 1

            # Log loss values (accumulate every micro-batch):
            running_loss += loss.detach().item()
            running_diffusion_loss += diffusion_loss.detach().item()
            running_ratio_loss += ratio_loss.detach().item()
            running_tail_drop_fraction += tail_drop_f
            log_steps += 1

            # Accumulate compression ratio from boundary predictions.
            # Natural CR uses boundary_mask_natural (what the router chose before any
            # tail-drop) so the metric's meaning is stable across training modes.
            # Effective CR uses boundary_mask (post-drop, actual compute used).
            _unwrapped = accelerator.unwrap_model(model)
            if _unwrapped._last_boundary_predictions:
                _total_ratio = 0.0
                _total_std = 0.0
                _total_eff_ratio = 0.0
                for _bpred in _unwrapped._last_boundary_predictions:
                    _L = _bpred.boundary_mask.shape[1]
                    # Natural mask (pre-drop): fall back to boundary_mask for safety
                    _natural_mask = _bpred.boundary_mask_natural if _bpred.boundary_mask_natural is not None else _bpred.boundary_mask
                    _num_natural = _natural_mask.float().sum(dim=-1).clamp(min=1)
                    _per_sample_natural_ratio = _L / _num_natural  # (B,)
                    _total_ratio += _per_sample_natural_ratio.log().mean().item()
                    _total_std += _per_sample_natural_ratio.std().item()
                    running_compression_ratio_min = min(running_compression_ratio_min, _per_sample_natural_ratio.min().item())
                    running_compression_ratio_max = max(running_compression_ratio_max, _per_sample_natural_ratio.max().item())
                    # Effective mask (post-drop)
                    _num_eff = _bpred.boundary_mask.float().sum(dim=-1).clamp(min=1)
                    _per_sample_eff_ratio = _L / _num_eff  # (B,)
                    _total_eff_ratio += _per_sample_eff_ratio.log().mean().item()
                _n_stages = len(_unwrapped._last_boundary_predictions)
                running_compression_ratio += _total_ratio / _n_stages
                running_compression_ratio_std += _total_std / _n_stages
                running_effective_compression_ratio += _total_eff_ratio / _n_stages
            
            if accelerator.sync_gradients and train_steps % config.logging.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes

                avg_loss = torch.tensor(avg_loss, device=accelerator.device)
                avg_loss = accelerator.reduce(avg_loss, reduction="sum")

                avg_diffusion_loss = running_diffusion_loss / log_steps
                avg_ratio_loss = running_ratio_loss / log_steps
                avg_compression_ratio = float(np.exp(running_compression_ratio / log_steps))
                avg_compression_ratio_std = running_compression_ratio_std / log_steps
                avg_effective_compression_ratio = float(np.exp(running_effective_compression_ratio / log_steps))
                avg_tail_drop_fraction = running_tail_drop_fraction / log_steps
                
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Diffusion: {avg_diffusion_loss:.4f}, Ratio: {avg_ratio_loss:.4f}, Compression: {avg_compression_ratio:.3f}x (std={avg_compression_ratio_std:.3f}, min={running_compression_ratio_min:.3f}, max={running_compression_ratio_max:.3f}), Steps/Sec: {steps_per_sec:.2f}")
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/diffusion_loss": avg_diffusion_loss,
                        "train/ratio_loss": avg_ratio_loss,
                        "train/avg_compression_ratio": avg_compression_ratio,
                        "train/compression_ratio_std": avg_compression_ratio_std,
                        "train/compression_ratio_min": running_compression_ratio_min,
                        "train/compression_ratio_max": running_compression_ratio_max,
                        "train/grad_norm": grad_norm.item(),
                        "train/steps_per_sec": steps_per_sec,
                        "train_steps": train_steps,
                        "epoch": epoch,
                    }
                    if config.training.multi_budget_training:
                        log_dict["train/tail_drop_fraction"] = avg_tail_drop_fraction
                        log_dict["train/effective_compression_ratio"] = avg_effective_compression_ratio
                    wandb.log(log_dict, step=train_steps // config.logging.log_every)
                # Reset monitoring variables:
                running_loss = 0
                running_diffusion_loss = 0
                running_ratio_loss = 0
                running_compression_ratio = 0.0
                running_compression_ratio_std = 0.0
                running_compression_ratio_min = float('inf')
                running_compression_ratio_max = 0.0
                running_effective_compression_ratio = 0.0
                running_tail_drop_fraction = 0.0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if accelerator.sync_gradients and train_steps % config.logging.ckpt_every == 0 and train_steps > 0:
                opt_state_dict = opt.state_dict()
                checkpoint = {
                    "model": accelerator.get_state_dict(model),
                    "ema": accelerator.get_state_dict(ema),
                    "opt": opt_state_dict,
                    "config": config
                }
                if accelerator.is_main_process:
                    ckpt_save_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, ckpt_save_path)
                    logger.info(f"Saved checkpoint to {ckpt_save_path}")
            
            # Step profiler and check if profiling is complete
            if profiler:
                profiler.step()
                if train_steps > profile_end:
                    profiler.__exit__(None, None, None)
                    profiler = None
                    if accelerator.is_main_process:
                        logger.info(f"Profiling complete. Traces saved to {experiment_dir}/profiler")

    # Cleanup profiler if still active
    if profiler:
        profiler.__exit__(None, None, None)
        if accelerator.is_main_process:
            logger.info(f"Profiling complete. Traces saved to {experiment_dir}/profiler")
    
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")
        wandb.finish()


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint file in a directory based on step number."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = glob(f"{checkpoint_dir}/*.pt")
    if not checkpoints:
        return None
    # Sort by step number (filename format: 0050000.pt)
    checkpoints.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    return checkpoints[-1]


def find_wandb_run_id(experiment_dir: str) -> str | None:
    """Extract the wandb run ID from an experiment directory."""
    wandb_dir = os.path.join(experiment_dir, "wandb")
    if not os.path.exists(wandb_dir):
        return None
    # Look for run directories (format: run-YYYYMMDD_HHMMSS-RUNID)
    run_dirs = glob(f"{wandb_dir}/run-*")
    if not run_dirs:
        return None
    # Get the most recent run directory
    run_dirs.sort(key=os.path.getmtime)
    latest_run = run_dirs[-1]
    # Extract run ID from directory name
    run_id = os.path.basename(latest_run).split('-')[-1]
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiT with YAML configuration")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, help="Path to experiment directory to resume from")
    parser.add_argument("--same_wandb", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to resume the same wandb run (true) or create a new one (false). Default: true")
    args = parser.parse_args()
    
    if args.resume:
        # Resume from existing experiment
        experiment_dir = args.resume.rstrip('/')
        
        # Use provided config if given, otherwise load from experiment dir
        if args.config:
            config = load_config(args.config)
        else:
            config_path = os.path.join(experiment_dir, "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            config = load_config(config_path)
        
        # Find latest checkpoint
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        # Find wandb run ID if resuming same run
        wandb_run_id = find_wandb_run_id(experiment_dir) if args.same_wandb else None
        
        main(config, resume_from=experiment_dir, checkpoint_path=latest_ckpt, wandb_run_id=wandb_run_id)
    elif args.config:
        config = load_config(args.config)
        main(config, checkpoint_path=config.training.resume_from_ckpt)
    else:
        parser.error("Either --config or --resume must be specified")
