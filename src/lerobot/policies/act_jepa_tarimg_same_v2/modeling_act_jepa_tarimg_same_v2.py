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
"""ACT-JEPA v2: Action-conditioned predictor with teacher-forcing + rollout loss.

Predictor receives concat(s_x, action) at each timestep with causal masking.
s_x = context encoder output (pooled to single vector), action = action embedding.
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

from lerobot.policies.act_jepa_tarimg_same_v2.configuration_act_jepa_tarimg_same_v2 import (
    ACTJEPATarimgSameV2Config,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTJEPATarimgSameV2Policy(PreTrainedPolicy):
    """ACT-JEPA v2 Policy."""

    config_class = ACTJEPATarimgSameV2Config
    name = "act_jepa_tarimg_same_v2"

    def __init__(self, config: ACTJEPATarimgSameV2Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACTJEPATarimgSameV2(config)

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

        actions_hat, tf_loss, rollout_loss, sigreg_loss = self.model(batch, training=True)

        # Action loss: L1 (optionally per-dimension weighted)
        assert batch["action_is_pad"].ndim == 2
        action_mask = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(batch[ACTION])
        per_element_loss = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        if self.config.loss_weight_per_dim is not None:
            weight = torch.tensor(
                self.config.loss_weight_per_dim,
                device=per_element_loss.device,
                dtype=per_element_loss.dtype,
            )
            per_element_loss = per_element_loss * weight
        l1_loss = (per_element_loss * action_mask).sum() / action_mask.sum().clamp(min=1)

        loss_dict = {"l1_loss": l1_loss.item()}

        loss = l1_loss + self.config.observation_loss_weight * tf_loss
        loss_dict["tf_loss"] = tf_loss.item()

        if rollout_loss is not None:
            loss = loss + self.config.observation_loss_weight * rollout_loss
            loss_dict["rollout_loss"] = rollout_loss.item()

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


class ActionEncoder(nn.Module):
    """Encode action sequence into embeddings."""

    def __init__(self, action_dim: int, emb_dim: int, mlp_scale: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv1d(action_dim, action_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(action_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, action_dim) -> (B, T, emb_dim)"""
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class ACTJEPATarimgSameV2(nn.Module):
    """ACT-JEPA v2 core model.

    Action-conditioned predictor with causal masking.
    Predictor receives concat(s_x, action) at each timestep.
    """

    def __init__(self, config: ACTJEPATarimgSameV2Config):
        super().__init__()
        self.config = config

        # Backbone (shared between context and target)
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Context encoder (for action decoder)
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

        n_1d_tokens = 0
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.ctx_1d_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.ctx_cam_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Target encoder (EMA, transformer only)
        self.target_encoder = ACTEncoder(config, n_layers_override=config.n_target_encoder_layers)
        if self.config.image_features:
            self.target_img_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
            self.target_cam_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Action encoder (for predictor conditioning)
        action_dim = self.config.action_feature.shape[0]
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            emb_dim=config.dim_model,
            mlp_scale=config.action_encoder_mlp_scale,
        )

        # Project concat(s_x_pooled, action_emb) from 2D -> D
        self.s_x_proj = nn.Linear(config.dim_model, config.dim_model)
        self.action_cond_proj = nn.Linear(config.dim_model * 2, config.dim_model)

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
        self.sigreg = SIGReg(
            knots=config.sigreg_knots,
            num_proj=config.sigreg_num_proj,
        )
        self._image_token_size = None

        # Validate architecture match
        for (n_c, p_c), (n_t, p_t) in zip(
            self.context_encoder.named_parameters(), self.target_encoder.named_parameters()
        ):
            assert p_c.shape == p_t.shape

        # Initialize
        nn.init.normal_(self.action_mask_tokens, std=0.02)
        nn.init.normal_(self.action_pos_embed, std=0.02)

        # Initialize target as copy of context
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        # Freeze target path
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        if self.config.image_features:
            self.target_img_proj.load_state_dict(self.ctx_img_proj.state_dict())
            for p in self.target_img_proj.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        for param_c, param_t in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data.mul_(self.config.ema_decay).add_(param_c.data, alpha=1 - self.config.ema_decay)
        if self.config.image_features:
            for param_c, param_t in zip(
                self.ctx_img_proj.parameters(), self.target_img_proj.parameters()
            ):
                param_t.data.mul_(self.config.ema_decay).add_(param_c.data, alpha=1 - self.config.ema_decay)

    def _encode_context(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Encode current observation into context representation."""
        encoder_in_tokens = []
        encoder_in_pos_embed = []
        pos_idx = 0

        if self.config.robot_state_feature:
            obs_state = batch[OBS_STATE]
            if obs_state.ndim == 3:
                obs_state = obs_state[:, 0]
            state_token = self.ctx_robot_state_proj(obs_state).unsqueeze(0)
            encoder_in_tokens.append(state_token)
            encoder_in_pos_embed.append(self.ctx_1d_pos_embed.weight[pos_idx : pos_idx + 1].unsqueeze(1))
            pos_idx += 1

        if self.config.env_state_feature:
            env_token = self.ctx_env_state_proj(batch[OBS_ENV_STATE]).unsqueeze(0)
            encoder_in_tokens.append(env_token)
            encoder_in_pos_embed.append(self.ctx_1d_pos_embed.weight[pos_idx : pos_idx + 1].unsqueeze(1))
            pos_idx += 1

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
        """Encode all frames (0..T) through target encoder -> (B, T+1, D)."""
        img_list = batch[OBS_IMAGES]
        img_seq = img_list[0]  # (B, T+1, C, H, W)
        if img_seq.ndim != 5:
            raise ValueError(f"Expected 5D image tensor. Got shape {img_seq.shape}.")

        B, T_total, C, H, W = img_seq.shape

        imgs_flat = img_seq.reshape(B * T_total, C, H, W)
        with torch.no_grad():
            features = self.backbone(imgs_flat)["feature_map"]
        features = self.target_img_proj(features)
        pos_embed = self.target_cam_pos_embed(features).to(dtype=features.dtype)

        features = einops.rearrange(features, "bt c h w -> (h w) bt c")
        pos_embed = einops.rearrange(pos_embed, "bt c h w -> (h w) bt c")

        with torch.no_grad():
            encoded = self.target_encoder(features, pos_embed=pos_embed)

        encoded = encoded.mean(dim=0)  # (B*T_total, D)
        z = encoded.reshape(B, T_total, -1)  # (B, T_total, D)
        return z

    def _build_predictor_input(
        self,
        s_x_pooled: Tensor,
        action_emb: Tensor,
        T: int,
    ) -> Tensor:
        """Build predictor input: concat(s_x, action_k) at each timestep.

        Args:
            s_x_pooled: (B, D) — pooled context encoder output
            action_emb: (B, T, D) — action embeddings
            T: number of timesteps
        Returns: (T, B, D)
        """
        # s_x expanded to each timestep
        s_x_exp = s_x_pooled.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        # Concatenate and project
        cond = torch.cat([s_x_exp, action_emb], dim=-1)  # (B, T, 2D)
        cond = self.action_cond_proj(cond)  # (B, T, D)
        cond = cond.permute(1, 0, 2)  # (T, B, D)
        return cond

    def _predictor_teacher_forcing(
        self,
        s_x_pooled: Tensor,
        action_emb: Tensor,
        z: Tensor,
        T: int,
    ) -> Tensor:
        """Teacher-forcing: predictor sees concat(s_x, a_k) at each step.

        Args:
            s_x_pooled: (B, D)
            action_emb: (B, T, D)
            z: (B, T+1, D) — target encoder outputs
            T: prediction horizon
        Returns: z_pred (B, T, D)
        """
        seq = self._build_predictor_input(s_x_pooled, action_emb, T)  # (T, B, D)
        out = self.predictor(seq)  # (T, B, D)
        z_pred = out.permute(1, 0, 2)  # (B, T, D)
        return z_pred

    def _predictor_rollout(
        self,
        s_x_pooled: Tensor,
        action_emb: Tensor,
        rollout_steps: int,
    ) -> Tensor:
        """Rollout: autoregressively feed predictions back (equation 3).

        At each step k, predictor sees concat(s_x, a_k) and predicted z_{k-1}.
        Only backprop through the last recurrent step (T=2 in practice).

        Args:
            s_x_pooled: (B, D)
            action_emb: (B, T, D)
            rollout_steps: number of autoregressive steps (T in equation 3)
        Returns: z_pred at final step (B, D)
        """
        B = s_x_pooled.shape[0]
        D = s_x_pooled.shape[1]
        z_pred = None

        for step in range(rollout_steps):
            # Build single-step input: concat(s_x, a_step)
            a_k = action_emb[:, step]  # (B, D)
            cond = torch.cat([s_x_pooled, a_k], dim=-1)  # (B, 2D)
            cond = self.action_cond_proj(cond)  # (B, D)

            # Append to sequence (autoregressive: predictor sees all previous)
            if step == 0:
                seq = cond.unsqueeze(0)  # (1, B, D)
            else:
                # Detach previous predictions except the last step
                if step < rollout_steps - 1:
                    z_prev = z_pred.detach().unsqueeze(0)  # (1, B, D), no grad
                else:
                    z_prev = z_pred.unsqueeze(0)  # (1, B, D), with grad
                seq = torch.cat([seq, z_prev, cond.unsqueeze(0)], dim=0)  # grows each step

            out = self.predictor(seq)  # (seq_len, B, D)
            z_pred = out[-1]  # (B, D) — last position's output

        return z_pred.unsqueeze(1)  # (B, 1, D)

    def forward(
        self, batch: dict[str, Tensor], training: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        batch_size = batch[OBS_STATE].shape[0]

        # 1. Encode context (for action decoder)
        s_x, _ = self._encode_context(batch)

        # 2. Action decoder
        action_mask = self.action_mask_tokens.expand(batch_size, -1, -1) + self.action_pos_embed
        action_decoder_out = self.action_decoder(action_mask, s_x)
        actions_hat = self.action_head(action_decoder_out)

        if not training:
            return actions_hat

        # 3. SIGReg on context image tokens (all cameras)
        sigreg_loss = None
        if self.config.image_features:
            if self._image_token_size is None:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 96, 96, device=s_x.device)
                    feat = self.backbone(dummy)["feature_map"]
                    self._image_token_size = feat.shape[-2] * feat.shape[-1]
            n_image_tokens = self._image_token_size * len(self.config.image_features)
            image_tokens = s_x[-n_image_tokens:]
            sigreg_loss = self.sigreg(image_tokens)

        # 4. Encode target images (all frames 0..T)
        z = self._encode_target_images(batch)  # (B, T+1, D)

        # 5. Pool s_x for predictor: mean pool over all tokens -> (B, D)
        s_x_pooled = s_x.mean(dim=0)  # (B, D)
        s_x_pooled = self.s_x_proj(s_x_pooled)  # (B, D)

        # 6. Encode actions
        action_seq = batch[ACTION][:, :self.config.prediction_horizon]  # (B, T, action_dim)
        action_emb = self.action_encoder(action_seq)  # (B, T, D)

        T = self.config.prediction_horizon

        # 7. Teacher-forcing loss: predictor sees concat(s_x, a_k) at each step
        z_pred_tf = self._predictor_teacher_forcing(s_x_pooled, action_emb, z, T)  # (B, T, D)
        tf_loss = F.smooth_l1_loss(z_pred_tf, z[:, 1:T+1], reduction="mean")

        # 8. Rollout loss: autoregressive for R steps, compare final prediction with z_{R+1}
        rollout_loss = None
        if self.config.rollout_steps > 0:
            R = self.config.rollout_steps
            z_pred_rollout = self._predictor_rollout(s_x_pooled, action_emb, R)  # (B, 1, D)
            rollout_loss = F.smooth_l1_loss(z_pred_rollout.squeeze(1), z[:, R], reduction="mean")

        return actions_hat, tf_loss, rollout_loss, sigreg_loss


class CausalTransformerEncoder(nn.Module):
    """Transformer encoder with causal (autoregressive) masking."""

    def __init__(self, config: ACTJEPATarimgSameV2Config, n_layers: int | None = None):
        super().__init__()
        num_layers = n_layers if n_layers is not None else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (S, B, D) — sequence-first
        Causal: position k attends to positions 0..k only.
        """
        S = x.size(0)
        # True means masked (not attended to)
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, key_padding_mask=None, attn_mask=causal_mask)
        x = self.norm(x)
        return x


class JEPACrossAttentionTransformer(nn.Module):
    """Cross-attention transformer: cross-attend to context, then self-attention."""

    def __init__(self, config: ACTJEPATarimgSameV2Config, n_layers: int | None = None):
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

    def __init__(self, config: ACTJEPATarimgSameV2Config, n_layers_override: int | None = None):
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
    def __init__(self, config: ACTJEPATarimgSameV2Config):
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

    def forward(self, x, pos_embed=None, key_padding_mask=None, attn_mask=None):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
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
