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
"""ACT-JEPA v3: Same as v2 but with RoPE (Rotary Position Embedding) instead of sinusoidal PE."""

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act_jepa_tarimg_same_v3.configuration_act_jepa_tarimg_same_v3 import (
    ACTJEPATarimgSameV3Config,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ============================================================
# RoPE utilities
# ============================================================

def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary positional embedding to query and key tensors.

    Args:
        q, k: (B, n_heads, S, head_dim)
        cos, sin: (S, head_dim)
    """
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def masked_smooth_l1_loss(input: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """Compute Smooth L1 over valid sequence positions only."""
    if input.shape != target.shape:
        raise ValueError(f"Input and target shapes must match, got {input.shape} and {target.shape}.")
    if valid_mask.shape != input.shape[:-1]:
        raise ValueError(
            f"Mask shape must match all non-feature dimensions, got {valid_mask.shape} "
            f"for input shape {input.shape}."
        )

    element_loss = F.smooth_l1_loss(input, target, reduction="none")
    expanded_mask = valid_mask.unsqueeze(-1).expand_as(element_loss)
    return (element_loss * expanded_mask).sum() / expanded_mask.sum().clamp(min=1)


def future_prediction_valid_mask(
    batch: dict[str, Tensor], target_image_key: str, prediction_horizon: int
) -> Tensor:
    """Return validity of each (action_t, target_image_t+1) prediction pair."""
    image_pad_key = f"{target_image_key}_is_pad"
    if image_pad_key not in batch:
        raise KeyError(
            f"Missing '{image_pad_key}', which is required to mask padded future-image targets."
        )

    action_is_pad = batch["action_is_pad"]
    image_is_pad = batch[image_pad_key]
    if action_is_pad.ndim != 2 or action_is_pad.shape[1] < prediction_horizon:
        raise ValueError(
            f"Expected action_is_pad to have shape (B, >= {prediction_horizon}), "
            f"got {action_is_pad.shape}."
        )
    if image_is_pad.ndim != 2 or image_is_pad.shape[1] < prediction_horizon + 1:
        raise ValueError(
            f"Expected {image_pad_key} to have shape (B, >= {prediction_horizon + 1}), "
            f"got {image_is_pad.shape}."
        )

    action_valid = ~action_is_pad[:, :prediction_horizon]
    target_image_valid = ~image_is_pad[:, 1 : prediction_horizon + 1]
    return action_valid & target_image_valid


class RotaryEmbedding(nn.Module):
    """1D Rotary Position Embedding."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> tuple[Tensor, Tensor]:
        """Returns (cos, sin) each of shape (seqlen, dim)."""
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)  # (seqlen, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seqlen, dim)
        return emb.cos(), emb.sin()


class RotaryEmbedding2D(nn.Module):
    """2D Rotary Position Embedding for spatial image tokens.

    Assigns independent rotary coordinate pairs to the height and width axes.
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D RoPE"
        half_dim = dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, 2, dtype=torch.float) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, h: int, w: int) -> tuple[Tensor, Tensor]:
        """Returns (cos, sin) each of shape (h*w, dim)."""
        device = self.inv_freq.device
        dtype = self.inv_freq.dtype

        # Height positions
        h_pos = torch.arange(h, device=device, dtype=dtype)
        h_freqs = torch.outer(h_pos, self.inv_freq)  # (h, dim//4)

        # Width positions
        w_pos = torch.arange(w, device=device, dtype=dtype)
        w_freqs = torch.outer(w_pos, self.inv_freq)  # (w, dim//4)

        # Lay out the first half as [height frequencies, width frequencies]
        # and repeat it in the second half. This matches rotate_half(), which
        # pairs the first half of a head with the second half, while keeping
        # height and width in independent rotary subspaces.
        h_freqs = h_freqs.unsqueeze(1).expand(-1, w, -1)  # (h, w, dim//4)
        w_freqs = w_freqs.unsqueeze(0).expand(h, -1, -1)  # (h, w, dim//4)
        freqs = torch.cat([h_freqs, w_freqs], dim=-1)  # (h, w, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (h, w, dim)
        emb = emb.reshape(h * w, -1)  # (h*w, dim)
        return emb.cos(), emb.sin()


# ============================================================
# Policy
# ============================================================

class ACTJEPATarimgSameV3Policy(PreTrainedPolicy):
    """ACT-JEPA v3 Policy (RoPE)."""

    config_class = ACTJEPATarimgSameV3Config
    name = "act_jepa_tarimg_same_v3"

    def __init__(self, config: ACTJEPATarimgSameV3Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACTJEPATarimgSameV3(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

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
    def update(self):
        """Update the EMA target path after the optimizer step."""
        self.model.update_target_encoder()

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
        actions = self.model(batch)
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, tf_loss, rollout_loss, sigreg_loss = self.model(batch, training=True)

        assert batch["action_is_pad"].ndim == 2
        action_mask = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(batch[ACTION])
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * action_mask
        ).sum() / action_mask.sum().clamp(min=1)

        loss_dict = {"l1_loss": l1_loss.item()}

        loss = l1_loss + self.config.observation_loss_weight * tf_loss
        loss_dict["tf_loss"] = tf_loss.item()

        if rollout_loss is not None:
            loss = loss + self.config.observation_loss_weight * rollout_loss
            loss_dict["rollout_loss"] = rollout_loss.item()

        if sigreg_loss is not None:
            loss = loss + self.config.sigreg_weight * sigreg_loss
            loss_dict["sigreg_loss"] = sigreg_loss.item()

        return loss, loss_dict


# ============================================================
# SIGReg
# ============================================================

class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: Tensor) -> Tensor:
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ============================================================
# Action Encoder
# ============================================================

class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, emb_dim: int, mlp_scale: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv1d(action_dim, action_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(action_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


# ============================================================
# Core Model
# ============================================================

class ACTJEPATarimgSameV3(nn.Module):
    """ACT-JEPA v3 core model with RoPE."""

    def __init__(self, config: ACTJEPATarimgSameV3Config):
        super().__init__()
        self.config = config

        # Backbone (shared)
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Context encoder
        self.context_encoder = ACTEncoder(config)

        if self.config.robot_state_feature:
            self.ctx_robot_state_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.ctx_env_state_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        if self.config.image_features:
            self.ctx_img_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        # 2D RoPE for image tokens (context & target)
        head_dim = config.dim_model // config.n_heads
        self.rope_2d = RotaryEmbedding2D(head_dim, theta=config.rope_theta)

        # Target encoder (EMA, transformer only)
        self.target_encoder = ACTEncoder(config, n_layers_override=config.n_target_encoder_layers)
        if self.config.image_features:
            self.target_img_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        # Action encoder
        action_dim = self.config.action_feature.shape[0]
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            emb_dim=config.dim_model,
            mlp_scale=config.action_encoder_mlp_scale,
        )

        # Project concat(s_x_pooled, action_emb) from 2D -> D
        self.s_x_proj = nn.Linear(config.dim_model, config.dim_model)
        self.action_cond_proj = nn.Linear(config.dim_model * 2, config.dim_model)

        # 1D RoPE for predictor
        self.rope_predictor = RotaryEmbedding(head_dim, theta=config.rope_theta)

        # Predictor: causal transformer over concat(s_x, action) tokens
        self.predictor = CausalTransformerEncoder(config, n_layers=config.n_predictor_layers)

        # Action decoder (cross-attention to context)
        self.action_mask_tokens = nn.Parameter(
            torch.randn(1, config.chunk_size, config.dim_model) * 0.02
        )
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, config.chunk_size, config.dim_model) * 0.02
        )
        self.action_decoder = JEPACrossAttentionTransformer(config, n_layers=config.n_decoder_layers)
        self.action_head = nn.Linear(config.dim_model, action_dim)

        # SIGReg
        self.sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_num_proj)

        # Initialize
        nn.init.normal_(self.action_mask_tokens, std=0.02)
        nn.init.normal_(self.action_pos_embed, std=0.02)

        # Initialize target as copy of context
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self.target_img_proj.load_state_dict(self.ctx_img_proj.state_dict())

        # Freeze target path
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_img_proj.parameters():
            p.requires_grad = False

        self.target_encoder.eval()
        self.target_img_proj.eval()

    def train(self, mode: bool = True):
        """Keep the frozen EMA target path deterministic during training."""
        super().train(mode)
        self.target_encoder.eval()
        self.target_img_proj.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self):
        for param_c, param_t in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data.mul_(self.config.ema_decay).add_(param_c.data, alpha=1 - self.config.ema_decay)
        for param_c, param_t in zip(
            self.ctx_img_proj.parameters(), self.target_img_proj.parameters()
        ):
            param_t.data.mul_(self.config.ema_decay).add_(param_c.data, alpha=1 - self.config.ema_decay)

    def _compute_image_rope(self, h: int, w: int) -> tuple[Tensor, Tensor]:
        """Compute 2D RoPE cos/sin for image tokens."""
        return self.rope_2d(h, w)

    def _encode_context(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode current observation into context representation."""
        encoder_in_tokens = []
        pos_idx = 0
        image_rope_cos = []
        image_rope_sin = []

        if self.config.robot_state_feature:
            obs_state = batch[OBS_STATE]
            if obs_state.ndim == 3:
                obs_state = obs_state[:, 0]
            state_token = self.ctx_robot_state_proj(obs_state).unsqueeze(0)
            encoder_in_tokens.append(state_token)
            pos_idx += 1

        if self.config.env_state_feature:
            env_token = self.ctx_env_state_proj(batch[OBS_ENV_STATE]).unsqueeze(0)
            encoder_in_tokens.append(env_token)
            pos_idx += 1

        n_1d_tokens = pos_idx

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                if img.ndim == 5:
                    img = img[:, 0]
                cam_features = self.backbone(img)["feature_map"]
                cam_features = self.ctx_img_proj(cam_features)
                feat_h, feat_w = cam_features.shape[-2], cam_features.shape[-1]
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                encoder_in_tokens.append(cam_features)

                # Compute a matching 2D RoPE block for every camera. Keeping
                # one block per camera also supports different feature-map sizes.
                cam_rope_cos, cam_rope_sin = self._compute_image_rope(feat_h, feat_w)
                image_rope_cos.append(cam_rope_cos)
                image_rope_sin.append(cam_rope_sin)

        encoder_in_tokens = torch.cat(encoder_in_tokens, dim=0)

        # Build full RoPE: state/env tokens use the identity rotation
        # (cos=1, sin=0), while image tokens use 2D RoPE.
        if image_rope_cos:
            rope_cos = torch.cat(image_rope_cos, dim=0)
            rope_sin = torch.cat(image_rope_sin, dim=0)
            S = encoder_in_tokens.shape[0]
            D = rope_cos.shape[-1]
            full_cos = torch.ones(S, D, device=encoder_in_tokens.device, dtype=encoder_in_tokens.dtype)
            full_sin = torch.zeros(S, D, device=encoder_in_tokens.device, dtype=encoder_in_tokens.dtype)
            full_cos[n_1d_tokens:] = rope_cos
            full_sin[n_1d_tokens:] = rope_sin
        else:
            full_cos, full_sin = None, None

        s_x = self.context_encoder(encoder_in_tokens, rope_cos=full_cos, rope_sin=full_sin)
        return s_x

    def _encode_target_images(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode all frames (0..T) through target encoder -> (B, T+1, D)."""
        img_list = batch[OBS_IMAGES]
        img_seq = img_list[0]
        if img_seq.ndim != 5:
            raise ValueError(f"Expected 5D image tensor. Got shape {img_seq.shape}.")

        B, T_total, C, H, W = img_seq.shape

        imgs_flat = img_seq.reshape(B * T_total, C, H, W)
        with torch.no_grad():
            features = self.backbone(imgs_flat)["feature_map"]
        features = self.target_img_proj(features)
        feat_h, feat_w = features.shape[-2], features.shape[-1]

        features = einops.rearrange(features, "bt c h w -> (h w) bt c")

        # 2D RoPE for target image tokens
        rope_cos, rope_sin = self._compute_image_rope(feat_h, feat_w)

        with torch.no_grad():
            encoded = self.target_encoder(features, rope_cos=rope_cos, rope_sin=rope_sin)

        encoded = encoded.mean(dim=0)
        z = encoded.reshape(B, T_total, -1)
        return z

    def _predictor_teacher_forcing(
        self,
        s_x_pooled: Tensor,
        action_emb: Tensor,
        z: Tensor,
        T: int,
    ) -> Tensor:
        """Teacher-forcing: predictor sees concat(s_x, a_k) at each step."""
        s_x_exp = s_x_pooled.unsqueeze(1).expand(-1, T, -1)
        cond = torch.cat([s_x_exp, action_emb], dim=-1)
        cond = self.action_cond_proj(cond)
        cond = cond.permute(1, 0, 2)  # (T, B, D)

        # 1D RoPE for predictor
        cos, sin = self.rope_predictor(T)
        out = self.predictor(cond, rope_cos=cos, rope_sin=sin)
        z_pred = out.permute(1, 0, 2)
        return z_pred

    def _predictor_rollout(
        self,
        s_x_pooled: Tensor,
        action_emb: Tensor,
        rollout_steps: int,
    ) -> Tensor:
        """Rollout: autoregressively feed predictions back."""
        B = s_x_pooled.shape[0]
        z_pred = None

        for step in range(rollout_steps):
            a_k = action_emb[:, step]
            cond = torch.cat([s_x_pooled, a_k], dim=-1)
            cond = self.action_cond_proj(cond)

            if step == 0:
                seq = cond.unsqueeze(0)
            else:
                z_prev = z_pred.detach().unsqueeze(0) if step < rollout_steps - 1 else z_pred.unsqueeze(0)
                seq = torch.cat([seq, z_prev, cond.unsqueeze(0)], dim=0)

            seq_len = seq.shape[0]
            cos, sin = self.rope_predictor(seq_len)
            out = self.predictor(seq, rope_cos=cos, rope_sin=sin)
            z_pred = out[-1]

        return z_pred.unsqueeze(1)

    def forward(
        self, batch: dict[str, Tensor], training: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        # 1. Encode context
        s_x = self._encode_context(batch)
        batch_size = s_x.shape[1]

        # 2. Action decoder
        action_mask = self.action_mask_tokens.expand(batch_size, -1, -1) + self.action_pos_embed
        action_decoder_out = self.action_decoder(action_mask, s_x)
        actions_hat = self.action_head(action_decoder_out)

        if not training:
            return actions_hat

        # 3. SIGReg
        sigreg_loss = None
        if self.config.image_features:
            n_non_image_tokens = int(self.config.robot_state_feature is not None) + int(
                self.config.env_state_feature is not None
            )
            image_tokens = s_x[n_non_image_tokens:]
            sigreg_loss = self.sigreg(image_tokens)

        # 4. Encode target images
        z = self._encode_target_images(batch)

        # 5. Pool s_x for predictor
        s_x_pooled = self.s_x_proj(s_x.mean(dim=0))

        # 6. Encode actions
        action_seq = batch[ACTION][:, :self.config.prediction_horizon]
        action_emb = self.action_encoder(action_seq)

        T = self.config.prediction_horizon

        target_image_key = next(iter(self.config.image_features))
        prediction_valid = future_prediction_valid_mask(batch, target_image_key, T)

        # 7. Teacher-forcing loss
        z_pred_tf = self._predictor_teacher_forcing(s_x_pooled, action_emb, z, T)
        tf_loss = masked_smooth_l1_loss(z_pred_tf, z[:, 1:T+1], prediction_valid)

        # 8. Rollout loss
        rollout_loss = None
        if self.config.rollout_steps > 0:
            R = self.config.rollout_steps
            z_pred_rollout = self._predictor_rollout(s_x_pooled, action_emb, R)
            rollout_valid = prediction_valid[:, :R].all(dim=1)
            rollout_loss = masked_smooth_l1_loss(
                z_pred_rollout.squeeze(1), z[:, R], rollout_valid
            )

        return actions_hat, tf_loss, rollout_loss, sigreg_loss


# ============================================================
# Transformer components (with RoPE)
# ============================================================

class CausalTransformerEncoder(nn.Module):
    """Causal transformer encoder with RoPE."""

    def __init__(self, config: ACTJEPATarimgSameV3Config, n_layers: int | None = None):
        super().__init__()
        num_layers = n_layers if n_layers is not None else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, x: Tensor, rope_cos: Tensor | None = None, rope_sin: Tensor | None = None) -> Tensor:
        S = x.size(0)
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, rope_cos=rope_cos, rope_sin=rope_sin, attn_mask=causal_mask)
        x = self.norm(x)
        return x


class JEPACrossAttentionTransformer(nn.Module):
    """Cross-attention transformer (no RoPE on cross-attention, RoPE optional on self-attn)."""

    def __init__(self, config: ACTJEPATarimgSameV3Config, n_layers: int | None = None):
        super().__init__()
        n_layers = n_layers or config.n_decoder_layers
        self.cross_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.query_norm = nn.LayerNorm(config.dim_model)
        self.ctx_norm = nn.LayerNorm(config.dim_model)
        self.cross_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, query_tokens: Tensor, context: Tensor) -> Tensor:
        q = query_tokens.permute(1, 0, 2)
        skip = q
        q_norm = self.query_norm(q)
        c_norm = self.ctx_norm(context)
        cross_out = self.cross_attn(q_norm, c_norm, value=context)[0]
        q = skip + self.cross_dropout(cross_out)
        x = q
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.permute(1, 0, 2)


class ACTEncoder(nn.Module):
    """Transformer encoder with RoPE support."""

    def __init__(self, config: ACTJEPATarimgSameV3Config, n_layers_override: int | None = None):
        super().__init__()
        num_layers = n_layers_override if n_layers_override is not None else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, rope_cos: Tensor | None = None, rope_sin: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    """Transformer encoder layer with RoPE support."""

    def __init__(self, config: ACTJEPATarimgSameV3Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim_model = config.dim_model
        self.head_dim = config.dim_model // config.n_heads
        self.dropout_p = config.dropout

        # Custom Q/K/V projections (instead of nn.MultiheadAttention)
        self.q_proj = nn.Linear(config.dim_model, config.dim_model)
        self.k_proj = nn.Linear(config.dim_model, config.dim_model)
        self.v_proj = nn.Linear(config.dim_model, config.dim_model)
        self.out_proj = nn.Linear(config.dim_model, config.dim_model)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, rope_cos=None, rope_sin=None, attn_mask=None):
        """
        x: (S, B, D)
        rope_cos, rope_sin: (S, head_dim) or None
        attn_mask: (S, S) bool, True = masked
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)

        S, B, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x).reshape(S, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)  # (B, H, S, head_dim)
        k = self.k_proj(x).reshape(S, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(x).reshape(S, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)

        # Apply RoPE to Q, K
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, S)

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, S, head_dim)

        # Reshape and project
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(S, B, D)  # (S, B, D)
        attn_output = self.out_proj(attn_output)

        x = skip + self.dropout1(attn_output)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


# ============================================================
# Utility classes
# ============================================================

class ACTTemporalEnsembler:
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


def get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
