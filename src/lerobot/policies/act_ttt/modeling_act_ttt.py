#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ACT-TTT: Action Chunking Transformer with Test-Time Training.

Replaces standard softmax self-attention with TTT (Test-Time Training) blocks
and replaces decoder cross-attention with encoder-learned inner models.

Reference:
    ViT^3: Unlocking Test-Time Training in Vision (CVPR 2026)
    Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.act_ttt.configuration_act_ttt import ACTTTTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# ============================================================================
# TTT Blocks
# ============================================================================


class TTTBlock(nn.Module):
    """Full TTT block with SwiGLU + 3x3 depthwise convolution branches.

    Used for image token processing in the backbone where tokens have 2D spatial
    structure. Each branch trains an inner model on (K, V) with one gradient step,
    then applies the updated model to Q.
    """

    def __init__(self, dim, num_heads, input_resolution, mlp_ratio=4.0, qkv_bias=True, drop_path=0.0):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.input_resolution = input_resolution

        # Convolutional positional encoding (CPE)
        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)

        # TTT inner module parameters
        # SwiGLU branch: w1, w2 for global modeling
        self.w1 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        # 3x3 DWC branch: w3 for local spatial modeling
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)
        nn.init.trunc_normal_(self.w3, std=0.02)

        # QKV projection: q1,k1,v1 (dim) + q2,k2,v2 (head_dim)
        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim + head_dim, dim)

        equivalent_head_dim = 9
        self.scale = equivalent_head_dim ** -0.5
        self.inner_lr = 1.0

        # Drop path
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _MLP(dim, mlp_hidden_dim, drop=0.0)

    def forward(self, x):
        """
        Args:
            x: (B, N, C) where N = H * W
        Returns:
            (B, N, C)
        """
        H, W = self.input_resolution
        B, N, C = x.shape
        assert N == H * W

        # CPE
        x = x + self.cpe(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # Pre-norm
        residual = x
        x_norm = self.norm1(x)

        # TTT forward
        d = C // self.num_heads
        q1, k1, v1, q2, k2, v2 = torch.split(
            self.qkv(x_norm), [C, C, C, d, d, d], dim=-1
        )

        # SwiGLU branch
        q1 = q1.reshape(B, N, self.num_heads, d).transpose(1, 2)
        k1 = k1.reshape(B, N, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(B, N, self.num_heads, d).transpose(1, 2)
        w1, w2 = self._inner_train_swiglu(k1, v1, self.w1, self.w2)
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(B, N, C)

        # 3x3 DWC branch
        q2 = q2.reshape(B, H, W, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(B, H, W, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(B, H, W, d).permute(0, 3, 1, 2)
        w3 = self._inner_train_3x3dwc(k2, v2, self.w3)
        x2 = F.conv2d(q2.reshape(1, B * d, H, W), w3, padding=1, groups=B * d)
        x2 = x2.reshape(B, d, N).transpose(1, 2)

        x = self.proj(torch.cat([x1, x2], dim=-1))
        x = residual + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

    def _inner_train_swiglu(self, k, v, w1, w2):
        """One-step inner training for SwiGLU branch."""
        z1 = k @ w1
        z2 = k @ w2
        sig = torch.sigmoid(z2)
        a = z2 * sig

        e = -v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        w1_new = w1 - self.inner_lr * g1
        w2_new = w2 - self.inner_lr * g2
        return w1_new, w2_new

    def _inner_train_3x3dwc(self, k, v, w):
        """One-step inner training for 3x3 depthwise conv branch."""
        B, C, H, W = k.shape
        e = -v / float(v.shape[2] * v.shape[3]) * self.scale

        k_padded = F.pad(k, (1, 1, 1, 1))
        outs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ys = 1 + dy
                xs = 1 + dx
                dot = (k_padded[:, :, ys : ys + H, xs : xs + W] * e).sum(dim=(-2, -1))
                outs.append(dot)
        g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)

        w_new = w.repeat(B, 1, 1, 1) - self.inner_lr * g
        return w_new


class TTTBlockSwiGLU(nn.Module):
    """TTT block with only the SwiGLU branch (no 3x3 DWC).

    Used for 1D token sequences (latent, state, action) that lack 2D spatial
    structure. Follows the pre-norm + residual + FFN pattern of standard
    transformer blocks.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=0.0, inner_lr=1.0,
                 use_moe=False, n_experts=3, top_k=1):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.inner_lr = inner_lr
        self.use_moe = use_moe

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)
        self.proj = nn.Linear(dim, dim)
        self.scale = head_dim ** -0.5

        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_moe:
            self.mlp = MoEFFN(dim, mlp_hidden_dim, n_experts=n_experts, top_k=top_k)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(mlp_hidden_dim, dim),
                nn.Dropout(0.0),
            )

    def forward(self, x):
        """
        Args:
            x: (B, N, C) — arbitrary sequence length N
        Returns:
            out: (B, N, C)
            aux_loss: MoE load balancing loss (scalar, 0.0 if MoE disabled)
        """
        B, N, C = x.shape
        d = C // self.num_heads

        residual = x
        x_norm = self.norm1(x)

        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        q = q.reshape(B, N, self.num_heads, d).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, d).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, d).transpose(1, 2)

        # Inner training
        w1, w2 = self._inner_train_swiglu(k, v, self.w1, self.w2)

        # Inner inference
        x = (q @ w1) * F.silu(q @ w2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = residual + self.drop_path(x)

        # FFN (MoE or standard)
        if self.use_moe:
            mlp_out, aux_loss = self.mlp(self.norm2(x))
            x = x + self.drop_path(mlp_out)
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            aux_loss = 0.0
        return x, aux_loss

    def _inner_train_swiglu(self, k, v, w1, w2):
        """One-step inner training for SwiGLU branch."""
        z1 = k @ w1
        z2 = k @ w2
        sig = torch.sigmoid(z2)
        a = z2 * sig

        e = -v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        w1_new = w1 - self.inner_lr * g1
        w2_new = w2 - self.inner_lr * g2
        return w1_new, w2_new


# ============================================================================
# Utility modules
# ============================================================================


class DropPath(nn.Module):
    """Stochastic depth (DropPath) regularization."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class _MLP(nn.Module):
    """Standard MLP with GELU activation and depthwise conv."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dwc = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h=None, w=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        if h is not None and w is not None:
            B, N, C = x.shape
            x = x + self.dwc(x.reshape(B, h, w, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1).contiguous()
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MoEFFN(nn.Module):
    """Mixture-of-Experts FFN for modality-aware output modulation.

    Each expert is a 2-layer FFN that learns to modulate the shared TTT inner
    model output for a specific modality group. A learned router assigns tokens
    to top-k experts based on token features.
    """

    def __init__(self, dim, hidden_dim, n_experts=3, top_k=1, drop=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(n_experts)
        ])

        self.router = nn.Linear(dim, n_experts)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: (B, N, C)
        Returns:
            expert_out: (B, N, C)
            aux_loss: load balancing loss (scalar)
        """
        B, N, C = x.shape

        logits = self.router(x)                          # (B, N, n_experts)
        gates, indices = torch.topk(logits, self.top_k)  # (B, N, top_k)
        gates = F.softmax(gates, dim=-1)

        flat_x = x.reshape(B * N, C)
        flat_indices = indices.reshape(B * N, self.top_k)
        flat_gates = gates.reshape(B * N, self.top_k)

        expert_out = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            expert_ids = flat_indices[:, k]
            gate_vals = flat_gates[:, k]
            for e in range(self.n_experts):
                mask = (expert_ids == e)
                if mask.any():
                    out_e = self.experts[e](flat_x[mask])
                    expert_out[mask] += gate_vals[mask].unsqueeze(-1) * out_e

        expert_out = self.drop(expert_out).reshape(B, N, C)

        aux_loss = self._load_balancing_loss(logits, indices) if self.training else 0.0
        return expert_out, aux_loss

    def _load_balancing_loss(self, logits, indices):
        """Switch Transformer style load balancing loss."""
        B, N, E = logits.shape
        router_probs = F.softmax(logits, dim=-1)
        mask = torch.zeros(B, N, E, device=logits.device)
        mask.scatter_(2, indices, 1.0)
        tokens_per_expert = mask.mean(dim=[0, 1])
        router_per_expert = router_probs.mean(dim=[0, 1])
        return E * (tokens_per_expert * router_per_expert).sum()


# ============================================================================
# ViTTT-style Image Backbone
# ============================================================================


class _ConvLayer(nn.Module):
    """Conv + optional BN + optional activation."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True, norm=True, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_ch) if norm else None
        self.act = nn.GELU() if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ViTTTImageBackbone(nn.Module):
    """ViTTT-style image backbone: CNN stem + TTT blocks.

    Produces image tokens with 2D spatial structure, processed by TTT blocks
    with both SwiGLU (global) and 3x3 DWC (local) branches.

    Architecture:
        Stem: 5 conv layers with 3x stride-2 downsampling → 8x spatial reduction
        Body: N TTT blocks at fixed resolution
    """

    def __init__(self, image_size, embed_dim=32, depth=2, num_heads=2, patch_size=4, mlp_ratio=4.0,
                 use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        H, W = image_size
        # Stem: 5 conv layers, 3 stride-2 → 8x downsample
        self.stem = nn.Sequential(
            _ConvLayer(3, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            _ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            _ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            _ConvLayer(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            _ConvLayer(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False, act=False),
        )
        self.out_h = H // 8
        self.out_w = W // 8

        # TTT blocks
        self.blocks = nn.ModuleList([
            TTTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                input_resolution=(self.out_h, self.out_w),
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, N, embed_dim) where N = (H/8) * (W/8)
        """
        x = self.stem(x)  # (B, embed_dim, H/8, W/8)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, N, C)

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)
        return x

    @property
    def output_dim(self):
        return self.blocks[0].dim if self.blocks else 0


# ============================================================================
# ACT-TTT Encoder
# ============================================================================


class ACTTTTEncoder(nn.Module):
    """TTT-based encoder for multi-modal token fusion.

    Processes [latent, state, image_tokens] with SwiGLU-only TTT blocks.
    Saves inner model parameters for each layer to be used by the decoder
    as cross-attention replacement.
    """

    def __init__(self, config: ACTTTTConfig):
        super().__init__()
        self.use_checkpoint = getattr(config, 'use_gradient_checkpointing', False)
        self.layers = nn.ModuleList([
            TTTBlockSwiGLU(
                dim=config.dim_model,
                num_heads=config.n_heads,
                mlp_ratio=config.dim_feedforward / config.dim_model,
                inner_lr=config.ttt_inner_lr,
                drop_path=0.0,
                use_moe=config.use_moe,
                n_experts=config.moe_n_experts,
                top_k=config.moe_top_k,
            )
            for _ in range(config.n_encoder_layers)
        ])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def _compute_inner_model(self, layer, x):
        """Compute inner model parameters for one encoder layer."""
        B, N, C = x.shape
        d = C // layer.num_heads
        x_norm = layer.norm1(x)
        q, k, v = layer.qkv(x_norm).chunk(3, dim=-1)
        k = k.reshape(B, N, layer.num_heads, d).transpose(1, 2)
        v = v.reshape(B, N, layer.num_heads, d).transpose(1, 2)
        w1_new, w2_new = layer._inner_train_swiglu(k, v, layer.w1, layer.w2)
        return w1_new.detach(), w2_new.detach()

    def forward(self, x):
        """
        Args:
            x: (B, N, C) — [latent, state, *image_tokens]
        Returns:
            encoder_out: (B, N, C)
            inner_models: list of (w1, w2) per layer for decoder cross-attention
            aux_loss_total: sum of MoE load balancing losses across layers
        """
        inner_models = []
        aux_loss_total = 0.0
        for layer in self.layers:
            # Apply layer with optional gradient checkpointing
            if self.use_checkpoint and self.training:
                x, aux_loss = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x, aux_loss = layer(x)
            aux_loss_total = aux_loss_total + aux_loss

            # Compute inner model from MoE-modulated output (modality-aware)
            w1_new, w2_new = self._compute_inner_model(layer, x)
            inner_models.append((w1_new, w2_new))

        x = self.norm(x)
        return x, inner_models, aux_loss_total


# ============================================================================
# ACT-TTT Decoder
# ============================================================================


class ACTTTTDecoderLayer(nn.Module):
    """Decoder layer with TTT self-attention and inner model cross-attention.

    Self-attention: SwiGLU TTT on action tokens.
    Cross-attention: Apply inner model (w1, w2) to decoder queries.
    Inner model is initialized from encoder and updated via EMA.
    """

    def __init__(self, config: ACTTTTConfig):
        super().__init__()
        self.self_ttt = TTTBlockSwiGLU(
            dim=config.dim_model,
            num_heads=config.n_heads,
            mlp_ratio=config.dim_feedforward / config.dim_model,
            inner_lr=config.ttt_inner_lr,
        )

        # Cross-attention via inner model
        self.cross_norm = nn.LayerNorm(config.dim_model)
        self.cross_q_proj = nn.Linear(config.dim_model, config.dim_model)
        self.cross_proj = nn.Linear(config.dim_model, config.dim_model)
        self.cross_dropout = nn.Dropout(config.dropout)

        # Teacher branch: shared, EMA-updated, long-term statistical knowledge
        head_dim = config.dim_model // config.n_heads
        self.register_buffer("cross_w1_teacher", torch.zeros(1, config.n_heads, head_dim, head_dim))
        self.register_buffer("cross_w2_teacher", torch.zeros(1, config.n_heads, head_dim, head_dim))

        # FFN
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = _get_activation_fn(config.feedforward_activation)

        self.pre_norm = config.pre_norm
        self.num_heads = config.n_heads
        self.head_dim = head_dim

    def forward(self, x, sample_inner_model=None):
        """
        Args:
            x: (B, T, C) — decoder tokens (action embeddings)
            sample_inner_model: (w1, w2) from current encoder, each (B, heads, d, d)
        Returns:
            (B, T, C)
        """
        # Self-attention via TTT
        x, _ = self.self_ttt(x)

        # Dual-branch cross-attention
        residual = x
        x_norm = self.cross_norm(x)
        B, T, C = x_norm.shape
        d = self.head_dim

        q = self.cross_q_proj(x_norm).reshape(B, T, self.num_heads, d).transpose(1, 2)

        # Teacher branch: shared EMA weights, long-term knowledge
        teacher_out = (q @ self.cross_w1_teacher) * F.silu(q @ self.cross_w2_teacher)

        # Sample branch: per-sample encoder weights, current observation
        if sample_inner_model is not None:
            w1_s, w2_s = sample_inner_model
            sample_out = (q @ w1_s) * F.silu(q @ w2_s)
        else:
            sample_out = torch.zeros_like(teacher_out)

        cross_out = teacher_out + sample_out
        cross_out = cross_out.transpose(1, 2).reshape(B, T, C)
        cross_out = self.cross_proj(cross_out)
        x = residual + self.cross_dropout(cross_out)

        # FFN
        residual = x
        x_norm = self.norm3(x) if self.pre_norm else x
        x = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = residual + self.dropout3(x)

        return x


class ACTTTTDecoder(nn.Module):
    """Stack of TTT decoder layers with EMA-updated cross inner models."""

    def __init__(self, config: ACTTTTConfig):
        super().__init__()
        self.use_checkpoint = getattr(config, 'use_gradient_checkpointing', False)
        self.layers = nn.ModuleList([
            ACTTTTDecoderLayer(config) for _ in range(config.n_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.dim_model)

        # EMA update config
        self.ema_r = 0.8
        self.ema_update_interval = 100

    def init_cross_inner_models(self, encoder_inner_models):
        """Initialize teacher branch from encoder's inner models.

        Called once before training. Teacher is shared (batch-mean).
        """
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                enc_idx = min(i, len(encoder_inner_models) - 1)
                w1_enc, w2_enc = encoder_inner_models[enc_idx]
                layer.cross_w1_teacher.copy_(w1_enc.mean(dim=0, keepdim=True))
                layer.cross_w2_teacher.copy_(w2_enc.mean(dim=0, keepdim=True))

    def ema_update_teacher(self, encoder_inner_models, step_count):
        """EMA update of teacher branch (shared, batch-mean).

        teacher_w = r * teacher_w + (1-r) * mean(encoder_w)
        """
        if step_count == 0 or step_count % self.ema_update_interval != 0:
            return
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                enc_idx = min(i, len(encoder_inner_models) - 1)
                w1_enc, w2_enc = encoder_inner_models[enc_idx]
                w1_avg = w1_enc.mean(dim=0, keepdim=True)
                w2_avg = w2_enc.mean(dim=0, keepdim=True)
                layer.cross_w1_teacher.copy_(
                    self.ema_r * layer.cross_w1_teacher + (1 - self.ema_r) * w1_avg
                )
                layer.cross_w2_teacher.copy_(
                    self.ema_r * layer.cross_w2_teacher + (1 - self.ema_r) * w2_avg
                )

    def forward(self, x, encoder_inner_models=None):
        """
        Args:
            x: (B, T, C) — action tokens
            encoder_inner_models: list of (w1, w2) per encoder layer, each (B, heads, d, d)
        Returns:
            (B, T, C)
        """
        for i, layer in enumerate(self.layers):
            # Get corresponding encoder inner model for sample branch
            sample_im = None
            if encoder_inner_models is not None:
                enc_idx = min(i, len(encoder_inner_models) - 1)
                sample_im = encoder_inner_models[enc_idx]

            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, sample_im, use_reentrant=False)
            else:
                x = layer(x, sample_inner_model=sample_im)

        x = self.norm(x)
        return x


# ============================================================================
# VAE Encoder (standard attention, same as ACT)
# ============================================================================


class VAEEncoderLayer(nn.Module):
    """Standard transformer encoder layer for VAE."""

    def __init__(self, dim_model, n_heads, dim_feedforward, dropout, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, x, pos_embed=None, key_padding_mask=None):
        skip = x
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        x = self.norm1(x)
        skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        x = self.norm2(x)
        return x


class VAEEncoder(nn.Module):
    """Standard transformer encoder for VAE."""

    def __init__(self, config: ACTTTTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            VAEEncoderLayer(
                dim_model=config.dim_model,
                n_heads=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.feedforward_activation,
            )
            for _ in range(config.n_vae_encoder_layers)
        ])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, x, pos_embed=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


# ============================================================================
# Main ACT-TTT Model
# ============================================================================


class ACTTTT(nn.Module):
    """Action Chunking Transformer with Test-Time Training.

    Architecture:
        1. ViTTT backbone: image → image tokens (2D spatial)
        2. VAE encoder: [cls, state, action] → latent (training only)
        3. TTT encoder: [latent, state, image_tokens] → fused tokens
            - Saves inner model params for cross-attention
        4. TTT decoder: action tokens + encoder inner models → predicted actions

    Compared to original ACT:
        - ResNet → ViTTT backbone (TTT-based image feature extraction)
        - Encoder self-attention → TTT (SwiGLU branch)
        - Decoder self-attention → TTT (SwiGLU branch)
        - Decoder cross-attention → encoder's learned inner model
    """

    def __init__(self, config: ACTTTTConfig):
        super().__init__()
        self.config = config

        # ===== VAE encoder (training only) =====
        if config.use_vae:
            self.vae_encoder = VAEEncoder(config)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                config.action_feature.shape[0], config.dim_model
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_tokens = 1 + config.chunk_size
            if config.robot_state_feature:
                num_input_tokens += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                _create_sinusoidal_pos_embedding(num_input_tokens, config.dim_model).unsqueeze(0),
            )

        # ===== ViTTT image backbone =====
        if config.image_features:
            self.backbone = ViTTTImageBackbone(
                image_size=config.image_size,
                embed_dim=config.backbone_embed_dim,
                depth=config.backbone_depth,
                num_heads=config.backbone_num_heads,
                patch_size=config.backbone_patch_size,
                mlp_ratio=config.backbone_mlp_ratio,
                use_checkpoint=getattr(config, 'use_gradient_checkpointing', False),
            )
            self.encoder_img_feat_input_proj = nn.Linear(config.backbone_embed_dim, config.dim_model)

        # ===== Encoder input projections =====
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

        # Encoder positional embeddings
        n_1d_tokens = 1  # latent
        if config.robot_state_feature:
            n_1d_tokens += 1
        if config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if config.image_features:
            self.encoder_cam_feat_pos_embed = _SinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Modality type embeddings: guide MoE router to distinguish token sources
        # 0=latent, 1=state, 2=env_state, 3=image
        self.modality_embed = nn.Embedding(4, config.dim_model)

        # ===== TTT Encoder =====
        self.encoder = ACTTTTEncoder(config)

        # ===== Decoder =====
        self.decoder = ACTTTTDecoder(config)
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Cross inner model initialization flag (set after first forward)
        self._cross_w_initialized = False
        self._ema_step = 0

        # ===== Output =====
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        ttt_inner_names = {"w1", "w2", "w3"}
        for name, p in chain(self.encoder.named_parameters(), self.decoder.named_parameters()):
            if p.dim() > 1 and name.split(".")[-1] not in ttt_inner_names:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """
        Args:
            batch: dict with keys:
                - observation.images: list of (B, C, H, W) images
                - observation.state: (B, state_dim)
                - action: (B, chunk_size, action_dim) [training only]
                - action_is_pad: (B, chunk_size)

        Returns:
            actions: (B, chunk_size, action_dim)
            (mu, log_sigma): VAE latent distribution parameters (None if not using VAE)
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions must be provided when using VAE in training mode."

        # Organize image keys into OBS_IMAGES list
        if self.config.image_features and OBS_IMAGES not in batch:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        batch_size = (
            batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        )

        # ===== VAE: encode [cls, state, action] → latent =====
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
                vae_input = [cls_embed, robot_state_embed, self.vae_encoder_action_input_proj(batch[ACTION])]
            else:
                vae_input = [cls_embed, self.vae_encoder_action_input_proj(batch[ACTION])]
            vae_input = torch.cat(vae_input, dim=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], dim=1)

            cls_token_out = self.vae_encoder(
                vae_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], dtype=torch.float32, device=next(self.parameters()).device
            )

        # ===== Prepare encoder inputs =====
        # Add modality embedding so MoE router can distinguish token sources
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample) + self.modality_embed.weight[0]]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        modality_idx = 1
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]) + self.modality_embed.weight[modality_idx])
            modality_idx += 1
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]) + self.modality_embed.weight[modality_idx])
            modality_idx += 1

        if self.config.image_features:
            img_modality_embed = self.modality_embed.weight[3]  # all image tokens share index 3
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)  # (B, N_hw, backbone_dim)
                cam_features = self.encoder_img_feat_input_proj(cam_features) + img_modality_embed  # (B, N_hw, dim_model)

                # 2D positional embedding using backbone's spatial resolution
                B, N, D = cam_features.shape
                H, W = self.backbone.out_h, self.backbone.out_w
                assert N == H * W, f"Backbone output {N} != {H}*{W}"
                cam_features_2d = cam_features.reshape(B, H, W, D).permute(0, 3, 1, 2)
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features_2d).to(dtype=cam_features.dtype)
                # Keep pos_embed at batch=1, will broadcast in the addition
                cam_pos_embed = einops.rearrange(cam_pos_embed, "1 c h w -> (h w) 1 c")

                cam_features = einops.rearrange(cam_features, "b n c -> n b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        # ===== TTT Encoder =====
        encoder_in = encoder_in_tokens + encoder_in_pos_embed
        encoder_in = encoder_in.permute(1, 0, 2)  # (B, N, C)
        encoder_out, inner_models, moe_aux_loss = self.encoder(encoder_in)

        # Initialize decoder teacher branch from encoder on first forward
        if not self._cross_w_initialized:
            self.decoder.init_cross_inner_models(inner_models)
            # Sync teacher weights across DDP ranks
            if dist.is_initialized():
                for layer in self.decoder.layers:
                    dist.broadcast(layer.cross_w1_teacher, src=0)
                    dist.broadcast(layer.cross_w2_teacher, src=0)
            self._cross_w_initialized = True

        # EMA update decoder teacher branch during training
        if self.training:
            self._ema_step += 1
            self.decoder.ema_update_teacher(inner_models, self._ema_step)

        # ===== TTT Decoder =====
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in.dtype,
            device=encoder_in.device,
        )
        decoder_in = decoder_in + self.decoder_pos_embed.weight.unsqueeze(1)
        decoder_in = decoder_in.permute(1, 0, 2)  # (B, T, C)

        # Pass encoder inner_models for sample branch (real-time)
        decoder_out = self.decoder(decoder_in, encoder_inner_models=inner_models)

        # ===== Output =====
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2), moe_aux_loss


# ============================================================================
# Policy wrapper
# ============================================================================


class ACTTTTPolicy(PreTrainedPolicy):
    """ACT-TTT Policy: Action Chunking Transformer with Test-Time Training.

    Wraps ACTTTT model with training/inference logic, loss computation,
    and temporal ensembling support.
    """

    config_class = ACTTTTConfig
    name = "act_ttt"

    def __init__(self, config: ACTTTTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACTTTT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = _TemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat), moe_aux_loss = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        if self.config.use_moe:
            loss_dict["moe_aux_loss"] = moe_aux_loss.item()
            loss = loss + moe_aux_loss * self.config.moe_balance_weight

        return loss, loss_dict


# ============================================================================
# Helper classes and functions
# ============================================================================


class _TemporalEnsembler:
    """Temporal ensembling as described in Algorithm 2 of the ACT paper."""

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class _SinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embedding."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed


def _create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embedding."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


def _get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}")