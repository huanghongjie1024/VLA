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
"""ACT-JEPA + Target Image (same backbone, EMA) + SIGReg.

Target encoder processes future image frames through the shared ResNet backbone
and an EMA-updated transformer encoder + image projection.
"""

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

from lerobot.policies.act_jepa_tarimg_same.configuration_act_jepa_tarimg_same import (
    ACTJEPATarimgSameConfig,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTJEPATarimgSamePolicy(PreTrainedPolicy):
    """ACT-JEPA + Target Image (same backbone, EMA) + SIGReg Policy."""

    config_class = ACTJEPATarimgSameConfig
    name = "act_jepa_tarimg_same"

    def __init__(self, config: ACTJEPATarimgSameConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACTJEPATarimgSame(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
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
        actions = self.model(batch)
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, observations_hat, target_observations, sigreg_loss = self.model(batch, training=True)

        # Action loss: L1
        assert batch["action_is_pad"].ndim == 2
        action_mask = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(batch[ACTION])
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * action_mask
        ).sum() / action_mask.sum().clamp(min=1)

        loss_dict = {"l1_loss": l1_loss.item()}

        # Observation loss: smooth L1
        obs_loss = F.smooth_l1_loss(observations_hat, target_observations, reduction="mean")
        loss_dict["obs_loss"] = obs_loss.item()

        loss = l1_loss + self.config.observation_loss_weight * obs_loss

        # SIGReg loss
        if sigreg_loss is not None:
            loss = loss + self.config.sigreg_weight * sigreg_loss
            loss_dict["sigreg_loss"] = sigreg_loss.item()

        # Update target encoder via EMA
        self.model.update_target_encoder()

        return loss, loss_dict


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
        """proj: (T, B, D)"""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ACTJEPATarimgSame(nn.Module):
    """ACT-JEPA with image-based target encoder (shared backbone, EMA) and SIGReg."""

    def __init__(self, config: ACTJEPATarimgSameConfig):
        super().__init__()
        self.config = config

        # Backbone for image feature extraction (shared between context and target)
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Context encoder
        self.context_encoder = ACTEncoder(config)

        # Context encoder input projections
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

        # Context encoder positional embeddings
        n_1d_tokens = 0
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.ctx_1d_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.ctx_cam_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Target encoder (transformer only, backbone is shared)
        self.target_encoder = ACTEncoder(config, n_layers_override=config.n_target_encoder_layers)
        if self.config.image_features:
            self.target_img_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
            self.target_cam_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Predictor
        self.obs_mask_tokens = nn.Parameter(
            torch.randn(1, config.n_obs_mask_tokens, config.dim_model) * 0.02
        )
        self.register_buffer("predictor_pe", self._build_sinusoidal_pe(config.n_obs_mask_tokens, config.dim_model))
        self.predictor = JEPACrossAttentionTransformer(config, n_layers=config.n_predictor_layers)

        # Action decoder
        self.action_mask_tokens = nn.Parameter(
            torch.randn(1, config.chunk_size, config.dim_model) * 0.02
        )
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, config.chunk_size, config.dim_model) * 0.02
        )
        self.action_decoder = JEPACrossAttentionTransformer(config, n_layers=config.n_decoder_layers)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        # SIGReg
        self.sigreg = SIGReg(
            knots=config.sigreg_knots,
            num_proj=config.sigreg_num_proj,
        )
        self._image_token_size = None

        # Validate architecture match between context and target encoders
        assert config.n_obs_mask_tokens == config.prediction_horizon
        for (n_c, p_c), (n_t, p_t) in zip(
            self.context_encoder.named_parameters(), self.target_encoder.named_parameters()
        ):
            assert p_c.shape == p_t.shape

        # Initialize parameters
        nn.init.normal_(self.obs_mask_tokens, std=0.02)
        nn.init.normal_(self.action_mask_tokens, std=0.02)
        nn.init.normal_(self.action_pos_embed, std=0.02)

        # Initialize target as exact copy of context (BYOL/DINO standard)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self.target_img_proj.load_state_dict(self.ctx_img_proj.state_dict())

        # Freeze target path
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_img_proj.parameters():
            p.requires_grad = False

    @staticmethod
    def _build_sinusoidal_pe(length: int, dim: int) -> Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

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

    def _encode_context(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Encode current observation into context representation s_x."""
        encoder_in_tokens = []
        encoder_in_pos_embed = []
        pos_idx = 0

        # Robot state token
        if self.config.robot_state_feature:
            obs_state = batch[OBS_STATE]
            if obs_state.ndim == 3:
                obs_state = obs_state[:, 0]
            state_token = self.ctx_robot_state_proj(obs_state).unsqueeze(0)
            encoder_in_tokens.append(state_token)
            encoder_in_pos_embed.append(self.ctx_1d_pos_embed.weight[pos_idx : pos_idx + 1].unsqueeze(1))
            pos_idx += 1

        # Environment state token
        if self.config.env_state_feature:
            env_token = self.ctx_env_state_proj(batch[OBS_ENV_STATE]).unsqueeze(0)
            encoder_in_tokens.append(env_token)
            encoder_in_pos_embed.append(self.ctx_1d_pos_embed.weight[pos_idx : pos_idx + 1].unsqueeze(1))
            pos_idx += 1

        # Image feature tokens
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                if img.ndim == 5:
                    img = img[:, 0]
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.ctx_cam_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.ctx_img_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.append(cam_features)
                encoder_in_pos_embed.append(cam_pos_embed)

        encoder_in_tokens = torch.cat(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.cat(encoder_in_pos_embed, dim=0)

        s_x = self.context_encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        return s_x, encoder_in_pos_embed

    def _encode_target_images(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode future image frames into target representations.

        Returns:
            target_observations: (B, T, D) where T = prediction_horizon.
        """
        img_list = batch[OBS_IMAGES]
        img_seq = img_list[0]  # (B, T, C, H, W)
        if img_seq.ndim != 5:
            raise ValueError(
                f"Target image encoding requires 5D image tensor (B, T, C, H, W). Got shape {img_seq.shape}."
            )

        B, T_total, C, H, W = img_seq.shape
        T_target = T_total - 1  # skip frame 0

        # Process future frames through shared backbone
        future_imgs = img_seq[:, 1:]  # (B, T_target, C, H, W)
        future_imgs_flat = future_imgs.reshape(B * T_target, C, H, W)
        with torch.no_grad():
            target_features = self.backbone(future_imgs_flat)["feature_map"]

        # Project through target img proj
        target_features = self.target_img_proj(target_features)
        target_cam_pos_embed = self.target_cam_pos_embed(target_features).to(dtype=target_features.dtype)

        # Reshape to sequence: (HW, B*T_target, D)
        target_features = einops.rearrange(target_features, "bt c h w -> (h w) bt c")
        target_cam_pos_embed = einops.rearrange(target_cam_pos_embed, "bt c h w -> (h w) bt c")

        # Encode through target transformer
        with torch.no_grad():
            target_encoded = self.target_encoder(target_features, pos_embed=target_cam_pos_embed)

        # Pool over spatial tokens: mean pooling
        target_encoded = target_encoded.mean(dim=0)  # (B*T_target, D)
        target_observations = target_encoded.reshape(B, T_target, -1)  # (B, T_target, D)

        return target_observations

    def forward(
        self, batch: dict[str, Tensor], training: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor | None]:
        batch_size = batch[OBS_STATE].shape[0]

        # 1. Encode context: current observation -> s_x
        s_x, _ = self._encode_context(batch)

        # 2. Action decoder: s_x + action mask tokens -> predicted actions
        action_mask = self.action_mask_tokens.expand(batch_size, -1, -1) + self.action_pos_embed
        action_decoder_out = self.action_decoder(action_mask, s_x)
        actions_hat = self.action_head(action_decoder_out)

        if not training:
            return actions_hat

        # 3. SIGReg on image tokens
        sigreg_loss = None
        if self.config.image_features:
            if self._image_token_size is None:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 96, 96, device=s_x.device)
                    feat = self.backbone(dummy)["feature_map"]
                    self._image_token_size = feat.shape[-2] * feat.shape[-1]
            image_tokens = s_x[-self._image_token_size:]  # (HW, B, D)
            sigreg_loss = self.sigreg(image_tokens)

        # 4. Target encoder: future images -> target observations
        target_observations = self._encode_target_images(batch)  # (B, T, D)

        # 5. Predictor: s_x + obs mask tokens -> predicted s^y
        obs_mask = self.obs_mask_tokens.expand(batch_size, -1, -1) + self.predictor_pe.unsqueeze(0)
        observations_hat = self.predictor(obs_mask, s_x)

        return actions_hat, observations_hat, target_observations, sigreg_loss


class JEPACrossAttentionTransformer(nn.Module):
    """Cross-attention transformer: cross-attend to context, then self-attention."""

    def __init__(self, config: ACTJEPATarimgSameConfig, n_layers: int | None = None):
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
    """Transformer encoder with multiple layers."""

    def __init__(self, config: ACTJEPATarimgSameConfig, n_layers_override: int | None = None):
        super().__init__()
        num_layers = n_layers_override if n_layers_override is not None else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTJEPATarimgSameConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]
        x = skip + self.dropout1(x)
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


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings."""

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


def get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
