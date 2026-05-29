# Copyright (c) Advanced Micro Devices, Inc.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils import find_model
from models import create_dit
from config import load_config, Config
from flop_counter import format_flops
import argparse
import matplotlib.pyplot as plt
import numpy as np
from contextlib import nullcontext
from torch.profiler import profile, ProfilerActivity


def visualize_chunking(all_chunking_info, num_samples, vae=None, save_path="chunking_viz.png", lite_cfg=False):
    """
    Visualize boundary predictions across diffusion timesteps, correlated with intermediate images.
    
    Args:
        all_chunking_info: List of dicts with 'timestep', 'boundary_predictions', and 'x_t' keys.
        num_samples: Number of samples in the batch
        vae: Optional VAE decoder for latent space models
        save_path: Path to save the visualization
    """
    # Filter to get evenly spaced timesteps for visualization
    num_timesteps = len(all_chunking_info)
    num_vis_steps = min(8, num_timesteps)
    step_indices = np.linspace(0, num_timesteps - 1, num_vis_steps, dtype=int)
    
    num_stages = 0
    for info in all_chunking_info:
        if info["boundary_predictions"] is not None and len(info["boundary_predictions"]) > 0:
            num_stages = len(info["boundary_predictions"])
            break
    
    if num_stages == 0:
        print("No boundary predictions found. Model may not have dynamic chunking enabled.")
        return
    
    for sample_idx in range(num_samples):
        num_rows = 1 + num_stages  # 1 row for images, rest for boundary maps per stage
        fig, axes = plt.subplots(num_rows, num_vis_steps, figsize=(3 * num_vis_steps, 3 * num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Sample {sample_idx}: Boundary Predictions Correlated with Intermediate Images', 
                     fontsize=14, y=1.02)
        
        for col_idx, step_idx in enumerate(step_indices):
            info = all_chunking_info[step_idx]
            timestep = info["timestep"]
            boundary_predictions = info["boundary_predictions"]
            x_t = info.get("x_t")

            # Row 0: show x_t (the noisy network input) for this sample.
            ax_img = axes[0, col_idx]
            if x_t is not None:
                # The first num_samples entries are always the conditional samples
                # (doubled for CFG, or all we have when CFG is off).
                img = x_t[:num_samples][sample_idx]
                
                # Decode if using VAE
                if vae is not None:
                    with torch.no_grad():
                        img_decoded = vae.decode(img.unsqueeze(0) / 0.18215).sample[0]
                    img_np = img_decoded.permute(1, 2, 0).float().cpu().numpy()
                else:
                    img_np = img.permute(1, 2, 0).float().cpu().numpy()
                
                # Normalize to [0, 1] for display
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                ax_img.imshow(img_np)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title(f't={timestep}', fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax_img.set_ylabel('Network\nInput (x_t)', fontsize=10)
            
            if boundary_predictions is None or len(boundary_predictions) == 0:
                for stage_idx in range(num_stages):
                    axes[1 + stage_idx, col_idx].axis('off')
                continue
            
            for stage_idx, bpred in enumerate(boundary_predictions):
                ax = axes[1 + stage_idx, col_idx]

                boundary_mask = bpred.boundary_mask[:num_samples]  # (num_samples, L)

                sample_mask = boundary_mask[sample_idx].float()
                compression_ratio = sample_mask.mean().item()

                boundary_prob = bpred.boundary_prob[..., 1][:num_samples][sample_idx]  # (L,)
                
                # Reshape to grid
                L = boundary_prob.shape[0]
                side = int(np.sqrt(L))
                if side * side == L:
                    prob_map_2d = boundary_prob.float().cpu().numpy().reshape(side, side)
                else:
                    prob_map_2d = boundary_prob.float().cpu().numpy().reshape(1, -1)
                
                im = ax.imshow(prob_map_2d, cmap='RdYlGn', vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                
                ax.text(0.5, -0.15, f'{compression_ratio:.1%}', transform=ax.transAxes, 
                       ha='center', va='top', fontsize=9, color='darkblue', fontweight='bold')
                
                if col_idx == 0:
                    ax.set_ylabel(f'Stage {stage_idx}\nBoundary', fontsize=10)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.5])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), cax=cbar_ax)
        cbar.set_label('Boundary Probability', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        # Save with sample index in filename
        sample_save_path = save_path.replace('.png', f'_sample{sample_idx}.png')
        plt.savefig(sample_save_path, dpi=150, bbox_inches='tight')
        print(f"Chunking visualization for sample {sample_idx} saved to {sample_save_path}")
    plt.close()
    
    # Also create summary plots (aggregated across all samples)
    visualize_chunking_summary(all_chunking_info, num_stages, num_samples, save_path.replace('.png', '_summary.png'), lite_cfg=lite_cfg)


def _compression_stats_for_slice(all_chunking_info, num_stages, row_start, row_end):
    """Compute per-step compression ratios over a batch-row slice ``[row_start:row_end]``.

    Returns ``(boundary_ratios_avg, compression_ratios_avg)``, each a
    ``{stage_idx: [per-timestep values]}`` dict."""
    boundary_ratios_avg = {stage: [] for stage in range(num_stages)}
    compression_ratios_avg = {stage: [] for stage in range(num_stages)}
    for info in all_chunking_info:
        bpreds = info.get("boundary_predictions")
        if bpreds is None or len(bpreds) == 0:
            continue
        for stage_idx, bpred in enumerate(bpreds):
            mask_slice = bpred.boundary_mask[row_start:row_end].float()
            if mask_slice.numel() == 0:
                continue
            ratio = mask_slice.mean().item()
            boundary_ratios_avg[stage_idx].append(ratio)
            compression_ratios_avg[stage_idx].append(1.0 / ratio if ratio > 0 else float("inf"))
    return boundary_ratios_avg, compression_ratios_avg


def visualize_chunking_summary(all_chunking_info, num_stages, num_samples, save_path, lite_cfg=False):
    """
    Create a summary plot showing compression ratio and boundary statistics over timesteps.
    Shows per-sample statistics as well as aggregate.
    """
    timesteps = []
    # Per-sample tracking: boundary_ratios_per_sample[stage][sample] = list of ratios over time
    boundary_ratios_per_sample = {stage: {s: [] for s in range(num_samples)} for stage in range(num_stages)}
    compression_ratios_per_sample = {stage: {s: [] for s in range(num_samples)} for stage in range(num_stages)}
    # Aggregate (mean across samples)
    boundary_ratios_avg = {stage: [] for stage in range(num_stages)}
    compression_ratios_avg = {stage: [] for stage in range(num_stages)}
    avg_boundary_probs = {stage: [] for stage in range(num_stages)}
    
    for info in all_chunking_info:
        timestep = info["timestep"]
        boundary_predictions = info["boundary_predictions"]
        
        if boundary_predictions is None or len(boundary_predictions) == 0:
            continue
            
        timesteps.append(timestep)
        
        for stage_idx, bpred in enumerate(boundary_predictions):
            # Boundary mask: which tokens are selected as boundaries
            boundary_mask = bpred.boundary_mask  # (B, L)
            # Boundary probability
            boundary_prob = bpred.boundary_prob[..., 1]  # (B, L)

            # First num_samples entries are conditional (doubled for CFG, else all we have)
            mask_cond = boundary_mask[:num_samples].float()

            # Per-sample ratios
            for sample_idx in range(num_samples):
                sample_ratio = mask_cond[sample_idx].mean().item()
                boundary_ratios_per_sample[stage_idx][sample_idx].append(sample_ratio)
                sample_compression = 1.0 / sample_ratio if sample_ratio > 0 else float('inf')
                compression_ratios_per_sample[stage_idx][sample_idx].append(sample_compression)

            # Aggregate ratio (mean across samples)
            ratio = mask_cond.mean().item()
            boundary_ratios_avg[stage_idx].append(ratio)

            # Compression ratio = 1 / boundary_ratio (how much we compress)
            compression = 1.0 / ratio if ratio > 0 else float('inf')
            compression_ratios_avg[stage_idx].append(compression)

            avg_prob = boundary_prob[:num_samples].mean().item()
            avg_boundary_probs[stage_idx].append(avg_prob)
    
    if len(timesteps) == 0:
        return
    
    step_times = []
    step_flops = []
    flop_breakdowns = []
    for info in all_chunking_info:
        if "step_time" in info:
            step_times.append(info["step_time"])
        if "step_flops" in info and info["step_flops"] is not None:
            step_flops.append(info["step_flops"])
        if "flop_breakdown" in info and info["flop_breakdown"] is not None:
            flop_breakdowns.append(info["flop_breakdown"])
    
    has_flops = len(step_flops) > 0 and len(step_flops) == len(timesteps)
    has_breakdown = len(flop_breakdowns) > 0 and len(flop_breakdowns) == len(timesteps)
    
    num_rows = 4 if has_flops else 3
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, 5 * num_rows))
    
    stage_colors = plt.cm.tab10(np.linspace(0, 1, max(num_stages, 10)))[:num_stages]
    sample_colors = plt.cm.Set2(np.linspace(0, 1, max(num_samples, 8)))[:num_samples]
    
    ax1 = axes[0, 0]
    for stage_idx in range(num_stages):
        ax1.plot(timesteps, compression_ratios_avg[stage_idx], 
                label=f'Stage {stage_idx}', color=stage_colors[stage_idx], linewidth=2.5, marker='o', markersize=2)
    ax1.set_xlabel('Diffusion Timestep (t)', fontsize=12)
    ax1.set_ylabel('Compression Ratio (1/boundary_ratio)', fontsize=12)
    ax1.set_title('Compression Ratio (Avg Across Samples)\n(Higher = More Compression)', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=1)
    ax1.annotate('← Noisy', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9, color='gray', ha='left', va='top')
    ax1.annotate('Clean →', xy=(0.98, 0.98), xycoords='axes fraction', fontsize=9, color='gray', ha='right', va='top')
    
    ax2 = axes[0, 1]
    for stage_idx in range(num_stages):
        ax2.plot(timesteps, boundary_ratios_avg[stage_idx], 
                label=f'Stage {stage_idx}', color=stage_colors[stage_idx], linewidth=2.5)
        ax2.fill_between(timesteps, 0, boundary_ratios_avg[stage_idx], 
                        color=stage_colors[stage_idx], alpha=0.15)
    ax2.set_xlabel('Diffusion Timestep (t)', fontsize=12)
    ax2.set_ylabel('Boundary Token Ratio', fontsize=12)
    ax2.set_title('Fraction of Tokens Kept (Avg Across Samples)\n(Lower = More Compression)', fontsize=13)
    ax2.legend(loc='upper right')
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    ax3 = axes[0, 2]
    for stage_idx in range(num_stages):
        ax3.plot(timesteps, avg_boundary_probs[stage_idx],
                label=f'Stage {stage_idx}', color=stage_colors[stage_idx], linewidth=2.5)
    ax3.set_xlabel('Diffusion Timestep (t)', fontsize=12)
    ax3.set_ylabel('Average Boundary Probability', fontsize=12)
    ax3.set_title('Average Boundary Probability\n(Soft Routing Confidence)', fontsize=13)
    ax3.legend(loc='upper right')
    ax3.invert_xaxis()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    stage_to_show = 0
    
    ax4 = axes[1, 0]
    for sample_idx in range(num_samples):
        ax4.plot(timesteps, compression_ratios_per_sample[stage_to_show][sample_idx], 
                label=f'Sample {sample_idx}', color=sample_colors[sample_idx], linewidth=2, alpha=0.8)
    ax4.set_xlabel('Diffusion Timestep (t)', fontsize=12)
    ax4.set_ylabel('Compression Ratio', fontsize=12)
    ax4.set_title(f'Per-Sample Compression Ratio (Stage {stage_to_show})', fontsize=13)
    ax4.legend(loc='upper right', fontsize=8, ncol=2)
    ax4.invert_xaxis()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=1)
    
    ax5 = axes[1, 1]
    for sample_idx in range(num_samples):
        ax5.plot(timesteps, boundary_ratios_per_sample[stage_to_show][sample_idx], 
                label=f'Sample {sample_idx}', color=sample_colors[sample_idx], linewidth=2, alpha=0.8)
    ax5.set_xlabel('Diffusion Timestep (t)', fontsize=12)
    ax5.set_ylabel('Boundary Token Ratio', fontsize=12)
    ax5.set_title(f'Per-Sample Boundary Ratio (Stage {stage_to_show})', fontsize=13)
    ax5.legend(loc='upper right', fontsize=8, ncol=2)
    ax5.invert_xaxis()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    
    ax6 = axes[1, 2]
    for stage_idx in range(num_stages):
        stds = []
        for t_idx in range(len(timesteps)):
            sample_ratios = [boundary_ratios_per_sample[stage_idx][s][t_idx] for s in range(num_samples)]
            stds.append(np.std(sample_ratios))
        ax6.plot(timesteps, stds, label=f'Stage {stage_idx}', color=stage_colors[stage_idx], linewidth=2.5)
    ax6.set_xlabel('Diffusion Timestep (t)', fontsize=12)
    ax6.set_ylabel('Std Dev of Boundary Ratio', fontsize=12)
    ax6.set_title('Sample Variability in Boundary Decisions\n(Higher = More Variation Between Samples)', fontsize=13)
    ax6.legend(loc='upper right')
    ax6.invert_xaxis()
    ax6.grid(True, alpha=0.3)
    
    if len(step_times) > 1 and len(step_times) == len(timesteps):
        timesteps_plot = timesteps[1:]
        step_times_plot = step_times[1:]
        steps_per_sec = [1.0 / t for t in step_times_plot]
        compression_ratios_plot = compression_ratios_avg[0][1:] if len(compression_ratios_avg[0]) > 1 else []
        
        ax7 = axes[2, 0]
        ax7.plot(timesteps_plot, steps_per_sec, color='#E74C3C', linewidth=2.5, marker='o', markersize=3)
        ax7.fill_between(timesteps_plot, 0, steps_per_sec, color='#E74C3C', alpha=0.2)
        ax7.set_xlabel('Diffusion Timestep (t)', fontsize=12)
        ax7.set_ylabel('Steps per Second', fontsize=12)
        ax7.set_title('Throughput per Diffusion Step\n(Excluding first step)', fontsize=13)
        ax7.invert_xaxis()
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim(bottom=0)
        ax7.annotate('← Noisy', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9, color='gray', ha='left', va='top')
        ax7.annotate('Clean →', xy=(0.98, 0.98), xycoords='axes fraction', fontsize=9, color='gray', ha='right', va='top')
        
        ax8 = axes[2, 1]
        cumulative_times = np.cumsum(step_times_plot)
        ax8.plot(timesteps_plot, cumulative_times, color='#3498DB', linewidth=2.5, marker='o', markersize=3)
        ax8.fill_between(timesteps_plot, 0, cumulative_times, color='#3498DB', alpha=0.2)
        ax8.set_xlabel('Diffusion Timestep (t)', fontsize=12)
        ax8.set_ylabel('Cumulative Time (seconds)', fontsize=12)
        ax8.set_title(f'Cumulative Sampling Time\n(Total: {cumulative_times[-1]:.2f}s, excl. first step)', fontsize=13)
        ax8.invert_xaxis()
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(bottom=0)
        
        ax9 = axes[2, 2]
        if len(compression_ratios_plot) == len(steps_per_sec):
            ax9.scatter(compression_ratios_plot, steps_per_sec, c=timesteps_plot, cmap='viridis', s=50, alpha=0.7)
            ax9.set_xlabel('Compression Ratio (Stage 0)', fontsize=12)
            ax9.set_ylabel('Steps per Second', fontsize=12)
            ax9.set_title('Throughput vs Compression\n(Color = Timestep)', fontsize=13)
            ax9.grid(True, alpha=0.3)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(timesteps_plot), vmax=max(timesteps_plot)))
            cbar = plt.colorbar(sm, ax=ax9)
            cbar.set_label('Timestep', fontsize=10)
        else:
            ax9.axis('off')
            ax9.text(0.5, 0.5, 'Timing data mismatch', ha='center', va='center', transform=ax9.transAxes)
    else:
        axes[2, 0].axis('off')
        axes[2, 0].text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=axes[2, 0].transAxes, fontsize=12)
        axes[2, 1].axis('off')
        axes[2, 1].text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=axes[2, 1].transAxes, fontsize=12)
        axes[2, 2].axis('off')
        axes[2, 2].text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=axes[2, 2].transAxes, fontsize=12)
    
    if has_flops:
        timesteps_flops = timesteps[1:] if len(timesteps) > 1 else timesteps
        step_flops_plot = step_flops[1:] if len(step_flops) > 1 else step_flops
        compression_ratios_plot = compression_ratios_avg[0][1:] if len(compression_ratios_avg[0]) > 1 else compression_ratios_avg[0]
        
        gflops_per_step = [f / 1e9 for f in step_flops_plot]
        
        ax10 = axes[3, 0]
        ax10.plot(timesteps_flops, gflops_per_step, color='#9B59B6', linewidth=2.5, marker='o', markersize=3)
        ax10.fill_between(timesteps_flops, 0, gflops_per_step, color='#9B59B6', alpha=0.2)
        ax10.set_xlabel('Diffusion Timestep (t)', fontsize=12)
        ax10.set_ylabel('GFLOPs per Step', fontsize=12)
        ax10.set_title('FLOPs per Diffusion Step\n(Excluding first step)', fontsize=13)
        ax10.invert_xaxis()
        ax10.grid(True, alpha=0.3)
        ax10.set_ylim(bottom=0)
        ax10.annotate('← Noisy', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9, color='gray', ha='left', va='top')
        ax10.annotate('Clean →', xy=(0.98, 0.98), xycoords='axes fraction', fontsize=9, color='gray', ha='right', va='top')
        
        ax11 = axes[3, 1]
        cumulative_flops = np.cumsum(step_flops_plot)
        cumulative_gflops = [f / 1e9 for f in cumulative_flops]
        ax11.plot(timesteps_flops, cumulative_gflops, color='#E67E22', linewidth=2.5, marker='o', markersize=3)
        ax11.fill_between(timesteps_flops, 0, cumulative_gflops, color='#E67E22', alpha=0.2)
        ax11.set_xlabel('Diffusion Timestep (t)', fontsize=12)
        ax11.set_ylabel('Cumulative GFLOPs', fontsize=12)
        total_tflops = cumulative_flops[-1] / 1e12 if cumulative_flops[-1] >= 1e12 else cumulative_flops[-1] / 1e9
        unit = 'TFLOPs' if cumulative_flops[-1] >= 1e12 else 'GFLOPs'
        ax11.set_title(f'Cumulative FLOPs\n(Total: {total_tflops:.2f} {unit})', fontsize=13)
        ax11.invert_xaxis()
        ax11.grid(True, alpha=0.3)
        ax11.set_ylim(bottom=0)
        
        ax12 = axes[3, 2]
        if has_breakdown and len(flop_breakdowns) > 1:
            breakdowns_plot = flop_breakdowns[1:] if len(flop_breakdowns) > 1 else flop_breakdowns
            
            component_names = ['embeddings', 'encoder', 'routing', 'chunk', 'main_network', 'dechunk', 'decoder', 'final_layer']
            component_colors = plt.cm.Set3(np.linspace(0, 1, len(component_names)))
            
            total_by_component = {name: 0 for name in component_names}
            for bd in breakdowns_plot:
                for name in component_names:
                    total_by_component[name] = total_by_component.get(name, 0) + bd.get(name, 0)
            
            labels = [n for n in component_names if total_by_component[n] > 0]
            sizes = [total_by_component[n] / 1e9 for n in labels]
            colors = [component_colors[component_names.index(n)] for n in labels]
            
            if sum(sizes) > 0:
                ax12.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax12.set_title('FLOP Distribution by Component\n(Excluding first step)', fontsize=13)
            else:
                ax12.axis('off')
                ax12.text(0.5, 0.5, 'No component data', ha='center', va='center', transform=ax12.transAxes)
        elif len(compression_ratios_plot) == len(gflops_per_step):
            scatter = ax12.scatter(compression_ratios_plot, gflops_per_step, c=timesteps_flops, cmap='plasma', s=50, alpha=0.7)
            ax12.set_xlabel('Compression Ratio (Stage 0)', fontsize=12)
            ax12.set_ylabel('GFLOPs per Step', fontsize=12)
            ax12.set_title('FLOPs vs Compression\n(Color = Timestep)', fontsize=13)
            ax12.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax12)
            cbar.set_label('Timestep', fontsize=10)
        else:
            ax12.axis('off')
            ax12.text(0.5, 0.5, 'Data mismatch', ha='center', va='center', transform=ax12.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chunking summary saved to {save_path}")
    
    cond_label = "cond" if lite_cfg else "all"
    print(f"\n=== Compression Ratio Summary (Aggregate, {cond_label} path) ===")
    for stage_idx in range(num_stages):
        ratios = compression_ratios_avg[stage_idx]
        print(f"Stage {stage_idx}: min={min(ratios):.2f}x, max={max(ratios):.2f}x, avg={np.exp(np.mean(np.log(ratios))):.2f}x")

    print(f"\n=== Per-Sample Compression Statistics ({cond_label} path) ===")
    for sample_idx in range(num_samples):
        for stage_idx in range(num_stages):
            ratios = compression_ratios_per_sample[stage_idx][sample_idx]
            print(f"Sample {sample_idx}, Stage {stage_idx}: avg={np.exp(np.mean(np.log(ratios))):.2f}x")

    if lite_cfg:
        _, uncond_compression = _compression_stats_for_slice(
            all_chunking_info, num_stages,
            row_start=num_samples, row_end=2 * num_samples,
        )
        print("\n=== Compression Ratio Summary (Aggregate, uncond path) ===")
        for stage_idx in range(num_stages):
            ratios = uncond_compression[stage_idx]
            if not ratios:
                continue
            print(f"Stage {stage_idx}: min={min(ratios):.2f}x, max={max(ratios):.2f}x, avg={np.exp(np.mean(np.log(ratios))):.2f}x")


def print_timing_and_flop_stats(all_chunking_info):
    """Print per-step timing and FLOP statistics from collected chunking info.

    Safe to call whenever ``all_chunking_info`` was populated (i.e. whenever
    ``return_chunking_info=True`` was passed to the sampler), independent of
    whether visualization is enabled.
    """
    if not all_chunking_info:
        return

    step_times = [info["step_time"] for info in all_chunking_info if "step_time" in info]
    step_flops = [info["step_flops"] for info in all_chunking_info if info.get("step_flops") is not None]
    flop_breakdowns = [info["flop_breakdown"] for info in all_chunking_info if info.get("flop_breakdown") is not None]

    # Timing statistics (excluding first step due to startup latency)
    if len(step_times) > 1:
        step_times_excl_first = step_times[1:]
        steps_per_sec_stats = [1.0 / t for t in step_times_excl_first]
        print("\n=== Timing Statistics (excluding first step) ===")
        print(f"Total sampling time: {sum(step_times_excl_first):.2f}s")
        print(f"Average steps/sec: {np.mean(steps_per_sec_stats):.2f}")
        print(f"Min steps/sec: {min(steps_per_sec_stats):.2f}")
        print(f"Max steps/sec: {max(steps_per_sec_stats):.2f}")
        print(f"Std dev steps/sec: {np.std(steps_per_sec_stats):.2f}")

    # FLOP statistics (excluding first step)
    if not step_flops:
        return
    step_flops_excl_first = step_flops[1:] if len(step_flops) > 1 else step_flops
    total_flops = sum(step_flops_excl_first)
    gflops_per_step_stats = [f / 1e9 for f in step_flops_excl_first]
    print("\n=== FLOP Statistics (excluding first step) ===")
    print(f"Total FLOPs: {format_flops(total_flops)}")
    print(f"Average GFLOPs/step: {np.mean(gflops_per_step_stats):.2f}")
    print(f"Min GFLOPs/step: {min(gflops_per_step_stats):.2f}")
    print(f"Max GFLOPs/step: {max(gflops_per_step_stats):.2f}")
    print(f"Std dev GFLOPs/step: {np.std(gflops_per_step_stats):.2f}")

    if len(flop_breakdowns) == len(step_flops):
        breakdowns_excl_first = flop_breakdowns[1:] if len(flop_breakdowns) > 1 else flop_breakdowns
        component_names = ['embeddings', 'encoder', 'routing', 'chunk', 'main_network', 'dechunk', 'decoder', 'final_layer']
        total_by_component = {name: 0 for name in component_names}
        for bd in breakdowns_excl_first:
            for name in component_names:
                total_by_component[name] = total_by_component.get(name, 0) + bd.get(name, 0)

        print("\n=== FLOP Breakdown by Component ===")
        for name in component_names:
            flops = total_by_component[name]
            if flops > 0:
                pct = 100.0 * flops / total_flops if total_flops > 0 else 0
                print(f"  {name}: {format_flops(flops)} ({pct:.1f}%)")


def _make_lite_cfg_sample_fn(model, cond_tail_dropping_fraction, uncond_tail_dropping_fraction):
    """Build a CFG sampler with separate conditional and unconditional budgets."""
    def sample_fn(x, t, y, cfg_scale, return_chunking_info=False, flop_counter=None):
        assert x.shape[0] % 2 == 0, "Lite-CFG expects batched [cond; uncond] inputs"
        half_n = x.shape[0] // 2

        half = x[:half_n]
        combined = torch.cat([half, half], dim=0)

        frac = torch.tensor(
            [cond_tail_dropping_fraction] * half_n
            + [uncond_tail_dropping_fraction] * half_n,
            device=x.device, dtype=torch.float32,
        )

        model_out, boundary_predictions = model.forward(
            combined, t, y,
            return_chunking_info=return_chunking_info,
            flop_counter=flop_counter,
            tail_dropping_fraction=frac,
        )

        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        output = torch.cat([eps, rest], dim=1)
        return output, boundary_predictions

    return sample_fn


def main(
    config: Config,
    ckpt_path: str = None,
    cfg_scale: float = 4.0,
    num_sampling_steps: int = 250,
    seed: int = 1,
    vae_variant: str = "ema",
    visualize: bool = True,
    count_flops_enabled: bool = False,
    profile_enabled: bool = False,
    profile_trace_path: str = "sampling_trace.json",
    tail_dropping_fraction: float = 0.0,
    lite_cfg: bool = False,
    uncond_tail_dropping_fraction: float = 0.0,
):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

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

    if ckpt_path:
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = create_diffusion(
        timestep_respacing=str(num_sampling_steps),
        noise_schedule=config.diffusion.noise_schedule,
        diffusion_steps=config.diffusion.diffusion_steps,
        learn_sigma=config.training.learn_sigma,
    )
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_variant}").to(device)

    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    n = len(class_labels)
    z = torch.randn(n, latent_channels, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    using_cfg = cfg_scale > 1.0
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
    if tail_dropping_fraction > 0.0:
        print(f"Tail dropping fraction: {tail_dropping_fraction} "
              f"(dropping weakest {tail_dropping_fraction:.0%} of boundary tokens each step)")
    if using_cfg and lite_cfg:
        print(f"Lite-CFG enabled: cond tail_dropping_fraction={tail_dropping_fraction}, "
              f"uncond tail_dropping_fraction={uncond_tail_dropping_fraction}")

    autocast_context = torch.autocast(device_type=device, dtype=autocast_dtype)
    
    if profile_enabled:
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        profile_context = profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
    else:
        profile_context = nullcontext()

    with profile_context as prof, autocast_context:
        samples, all_chunking_info = diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
            return_chunking_info=True,
            count_flops=count_flops_enabled,
        )

    if profile_enabled:
        prof.export_chrome_trace(profile_trace_path)
        print(f"Profiler trace written to: {profile_trace_path}")

    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))

    if visualize:
        visualize_chunking(all_chunking_info, n, vae=vae, save_path="chunking_viz.png",
                           lite_cfg=using_cfg and lite_cfg)

    print_timing_and_flop_stats(all_chunking_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from DiT with YAML configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--ckpt", type=str, default=None, 
                        help="Path to checkpoint (overrides config.ckpt)")
    parser.add_argument("--cfg-scale", type=float, default=4.0,
                        help="CFG scale")
    parser.add_argument("--num-sampling-steps", type=int, default=250,
                        help="Number of DDPM sampling steps")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for random number generator")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema",
                        help="VAE variant to use")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable chunking visualization")
    parser.add_argument("--count-flops", action="store_true",
                        help="Enable per-step FLOP counting")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.profiler around sampling loop")
    parser.add_argument("--profile-trace-path", type=str, default="sampling_trace.json",
                        help="Path to Chrome/Perfetto trace JSON output")
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
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.num_sampling_steps,
        seed=args.seed,
        vae_variant=args.vae,
        visualize=not args.no_visualize,
        count_flops_enabled=args.count_flops,
        profile_enabled=args.profile,
        profile_trace_path=args.profile_trace_path,
        tail_dropping_fraction=args.tail_dropping_fraction,
        lite_cfg=args.lite_cfg,
        uncond_tail_dropping_fraction=args.uncond_tail_dropping_fraction,
    )
