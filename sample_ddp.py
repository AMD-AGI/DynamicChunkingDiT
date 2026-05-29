# Copyright (c) Advanced Micro Devices, Inc.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import create_dit
from utils import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from config import load_config, Config
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from datetime import timedelta

from sample import _make_lite_cfg_sample_fn


def main(
    config: Config,
    ckpt_path: str = None,
    output_dir: str = None,
    cfg_scale: float = 1.5,
    num_sampling_steps: int = 250,
    seed: int = 0,
    vae_variant: str = "ema",
    per_proc_batch_size: int = 32,
    num_fid_samples: int = 50000,
    tail_dropping_fraction: float = 0.0,
    lite_cfg: bool = False,
    uncond_tail_dropping_fraction: float = 0.0,
):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl", timeout=timedelta(hours=24))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    ckpt = ckpt_path or config.ckpt
    assert ckpt is not None, "Checkpoint path must be provided via --ckpt or config.ckpt"

    latent_size = config.data.image_size // 8
    latent_channels = 4

    model = create_dit(
        model_config=config.model,
        input_size=latent_size,
        in_channels=latent_channels,
        num_classes=config.data.num_classes,
        class_dropout_prob=config.training.class_dropout_prob,
        learn_sigma=config.training.learn_sigma,
    )
    model = model.to(device)

    state_dict = find_model(ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    
    diffusion = create_diffusion(
        timestep_respacing=str(num_sampling_steps),
        noise_schedule=config.diffusion.noise_schedule,
        diffusion_steps=config.diffusion.diffusion_steps,
        learn_sigma=config.training.learn_sigma,
    )

    if rank == 0:
        if tail_dropping_fraction > 0.0:
            print(f"Tail dropping fraction: {tail_dropping_fraction} "
                  f"(dropping weakest {tail_dropping_fraction:.0%} of boundary tokens each step)")
        if lite_cfg:
            print(f"Lite-CFG enabled: cond tail_dropping_fraction={tail_dropping_fraction}, "
                  f"uncond tail_dropping_fraction={uncond_tail_dropping_fraction}")

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_variant}").to(device)
    
    assert cfg_scale >= 1.0, "In almost all cases, cfg_scale should be >= 1.0"
    using_cfg = cfg_scale > 1.0

    sample_folder_dir = output_dir if output_dir else "samples"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        z = torch.randn(n, latent_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, config.data.num_classes, (n,), device=device)

        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            if lite_cfg:
                model_kwargs = dict(y=y, cfg_scale=cfg_scale)
                sample_fn = _make_lite_cfg_sample_fn(
                    model,
                    cond_tail_dropping_fraction=tail_dropping_fraction,
                    uncond_tail_dropping_fraction=uncond_tail_dropping_fraction,
                )
            else:
                model_kwargs = dict(y=y, cfg_scale=cfg_scale, tail_dropping_fraction=tail_dropping_fraction)
                sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y, tail_dropping_fraction=tail_dropping_fraction)
            sample_fn = model.forward

        samples, _ = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
    
    if rank == 0:
        print(f"Sampling complete. Total samples across all ranks: {total_samples}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from DiT with DDP using YAML configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint (overrides config.ckpt)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to output directory")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="CFG scale")
    parser.add_argument("--num-sampling-steps", type=int, default=250,
                        help="Number of DDPM sampling steps")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed (each rank offsets from this)")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema",
                        help="VAE variant to use")
    parser.add_argument("--per-proc-batch-size", type=int, default=32,
                        help="Batch size per GPU")
    parser.add_argument("--num-fid-samples", type=int, default=50000,
                        help="Total number of samples to generate for FID")
    parser.add_argument("--tail-dropping-fraction", type=float, default=0.0,
                        help="Fraction of lowest-probability boundary tokens to drop each step "
                             "(0.0 = disabled, 0.2 = drop weakest 20%%). Default: 0.0.")
    parser.add_argument("--lite-cfg", action="store_true",
                        help="Run conditional and unconditional CFG paths as separate forward "
                             "calls, using --tail-dropping-fraction for the conditional path and "
                             "--uncond-tail-dropping-fraction for the unconditional path.")
    parser.add_argument("--uncond-tail-dropping-fraction", type=float, default=0.0,
                        help="Tail-dropping fraction for the unconditional CFG path when "
                             "--lite-cfg is set (typically higher than --tail-dropping-fraction). "
                             "Ignored otherwise. Default: 0.0.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    main(
        config,
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.num_sampling_steps,
        seed=args.seed,
        vae_variant=args.vae,
        per_proc_batch_size=args.per_proc_batch_size,
        num_fid_samples=args.num_fid_samples,
        tail_dropping_fraction=args.tail_dropping_fraction,
        lite_cfg=args.lite_cfg,
        uncond_tail_dropping_fraction=args.uncond_tail_dropping_fraction,
    )
