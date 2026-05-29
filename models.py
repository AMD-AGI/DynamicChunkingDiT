# Copyright (c) Advanced Micro Devices, Inc.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed

from config import ModelConfig
from hnet.models.hnet import HNet
from hnet.modules import HNetDiTBlock
from contextlib import nullcontext

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                              DiT Model Components                             #
#################################################################################


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, cond_dim=None):
        super().__init__()
        cond_dim = cond_dim or hidden_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbedWithConv2d(PatchEmbed):
    def __init__(
            self,
            img_size = 224,
            patch_size = 16,
            in_chans = 3,
            embed_dim = 768,
            norm_layer = None,
            flatten = True,
            output_fmt = None,
            bias: bool = True,
            device=None,
            dtype=None,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=flatten,
            output_fmt=output_fmt,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(3,3), padding='same', stride=patch_size, bias=bias)


def _count_dit_depth(arch_layout) -> int:
    """Count total DiT blocks in the innermost stage of the architecture."""
    import re
    # Recurse to innermost stage
    while isinstance(arch_layout, list) and len(arch_layout) == 3:
        arch_layout = arch_layout[1]
    # Now at innermost: either a list with one element or a string
    if isinstance(arch_layout, list):
        arch_layout = arch_layout[0]
    # Parse and count DiT blocks
    matches = re.findall(r"[dD](\d+)", arch_layout)
    return sum(int(n) for n in matches)


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        in_channels=4,
        num_classes=1000,
        model_config: ModelConfig = None,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = True,
    ):
        super().__init__()
        
        hnet_config = model_config.hnet
        hidden_size = hnet_config.d_model[0]
        num_heads = hnet_config.num_heads[0]
        depth = _count_dit_depth(hnet_config.arch_layout)
        mlp_ratio = model_config.mlp_ratio
        
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = 1
        self.num_heads = num_heads

        # Compute reduced enc/dec dim from factor
        factor = hnet_config.enc_dec_dim_factor[0] if hnet_config.enc_dec_dim_factor else None
        intermediate_dim = hidden_size // factor if factor else hidden_size

        self.x_embedder = PatchEmbedWithConv2d(input_size, 1, in_channels, intermediate_dim, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Sin-cos pos_embed at innermost d_model; chunked through each stage,
        # added at the innermost stage just before the main network
        innermost_d_model = hnet_config.d_model[-1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, innermost_d_model), requires_grad=False)

        # Create H-Net model (dit_kwargs will be updated per-stage in HNet)
        dit_kwargs = dict(
            depth=depth,
            mlp_ratio=mlp_ratio,
        )
        self.hnet = HNet(
            hnet_config,
            stage_idx=0,
            dit_kwargs=dit_kwargs,
        )

        self.final_layer = FinalLayer(intermediate_dim, 1, self.out_channels, cond_dim=hidden_size)
        self.initialize_weights()
        
        # Cache for boundary predictions (used for ratio loss during training)
        self._last_boundary_predictions = None

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks: 
        def _zero_out_adaLN_modulation(module):
            if isinstance(module, HNetDiTBlock):
                nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(module.adaLN_modulation[-1].bias, 0)
        self.hnet.apply(_zero_out_adaLN_modulation)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x, t, y, return_chunking_info=False, flop_counter=None, tail_dropping_fraction=0.0):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        return_chunking_info: if True, also return boundary predictions for visualization
        flop_counter: optional ComponentFlopCounter for per-component FLOP tracking
        """
        if flop_counter is not None:
            flop_counter.set_model(self)
        
        # Embeddings
        ctx = flop_counter.track("embeddings") if flop_counter else nullcontext()
        with ctx:
            x = self.x_embedder(x)                   # (N, T, intermediate_dim)
            t = self.t_embedder(t)                   # (N, D)
            y = self.y_embedder(y, self.training)    # (N, D)
            c = t + y                                # (N, D)
        
        mask = torch.ones(x.size(0), x.size(1), dtype=torch.int32, device=x.device)
        num_rows = num_cols = int(x.shape[1] ** 0.5)
        x, boundary_predictions = self.hnet(x, cond=c, mask=mask, pos_embed=self.pos_embed, num_rows=num_rows, num_cols=num_cols, flop_counter=flop_counter, tail_dropping_fraction=tail_dropping_fraction)
        self._last_boundary_predictions = boundary_predictions  # Cache for ratio loss
        
        # Final layer
        ctx = flop_counter.track("final_layer") if flop_counter else nullcontext()
        with ctx:
            x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x, boundary_predictions if return_chunking_info else None

    def forward_with_cfg(self, x, t, y, cfg_scale, return_chunking_info=False, flop_counter=None, tail_dropping_fraction=0.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out, boundary_predictions = self.forward(combined, t, y, return_chunking_info=return_chunking_info, flop_counter=flop_counter, tail_dropping_fraction=tail_dropping_fraction)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        output = torch.cat([eps, rest], dim=1)
        return output, boundary_predictions


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def create_dit(
    model_config: ModelConfig,
    input_size: int,
    in_channels: int,
    num_classes: int,
    class_dropout_prob: float = 0.1,
    learn_sigma: bool = True,
) -> DiT:
    """Create a DiT model from a ModelConfig."""
    return DiT(
        input_size=input_size,
        in_channels=in_channels,
        num_classes=num_classes,
        model_config=model_config,
        class_dropout_prob=class_dropout_prob,
        learn_sigma=learn_sigma,
    )
