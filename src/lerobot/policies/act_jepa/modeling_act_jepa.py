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
"""ACT-JEPA: Action Chunking Transformer with Joint Embedding Predictive Architecture.

An end-to-end architecture that jointly learns action prediction (supervised) and
latent observation prediction (self-supervised). During training, context encoder,
target encoder, predictor, and action decoder are all used. During inference, only
the context encoder and action decoder are used.
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

from lerobot.policies.act_jepa.configuration_act_jepa import ACTJEPAConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTJEPAPolicy(PreTrainedPolicy):
    """ACT-JEPA Policy: Action Chunking Transformer with Joint Embedding Predictive Architecture.

    Jointly optimizes action prediction and latent observation prediction end-to-end.
    During inference, only the context encoder and action decoder are used.
    """

    config_class = ACTJEPAConfig
    name = "act_jepa"

    def __init__(
        self,
        config: ACTJEPAConfig,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTJEPA(config)

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
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
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
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Returns:
            loss: Combined action + observation loss.
            loss_dict: Dictionary of individual loss components for logging.
        """
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, observations_hat, target_observations, mu, log_sigma_x2 = self.model(batch, training=True)

        # Action loss: L1 between predicted and ground truth actions.
        # Only non-padded tokens contribute to the loss (correct denominator).
        assert batch["action_is_pad"].ndim == 2, (
            f"action_is_pad must be (B, chunk_size), got ndim={batch['action_is_pad'].ndim}"
        )
        action_mask = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(batch[ACTION])
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * action_mask
        ).sum() / action_mask.sum().clamp(min=1)

        loss_dict = {"l1_loss": l1_loss.item()}

        # Observation loss: L1 between predicted and target latent observations.
        obs_loss = F.smooth_l1_loss(observations_hat, target_observations, reduction="mean")
        loss_dict["obs_loss"] = obs_loss.item()

        loss = l1_loss + self.config.observation_loss_weight * obs_loss

        # VAE KL loss.
        if self.config.use_vae and mu is not None:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = kld.item()
            loss = loss + self.config.kl_weight * kld

        # Update target encoder via EMA.
        self.model.update_target_encoder()

        return loss, loss_dict


class ACTJEPA(nn.Module):
    """ACT-JEPA core model.

    Components:
        - Context encoder: ResNet-18 backbone + state projection + transformer encoder → s_x
        - Target encoder: state sequence projection + positional encoding + transformer → s^y (EMA-updated)
        - Predictor: cross-attention(s_x, obs_mask_tokens) + transformer → predicted s^y
        - Action decoder: cross-attention(s_x, action_mask_tokens) + transformer → predicted actions
    """

    def __init__(self, config: ACTJEPAConfig):
        super().__init__()
        self.config = config

        # ---- Backbone for image feature extraction ----
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # ---- Context encoder: encodes current observation into s_x ----
        self.context_encoder = ACTEncoder(config)

        # Context encoder input projections.
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

        # Context encoder positional embeddings.
        n_1d_tokens = 0
        if config.use_vae:
            n_1d_tokens += 1  # for latent z token
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.ctx_1d_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.ctx_cam_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # ---- Target encoder: encodes future state sequence into s^y (EMA-updated) ----
        self.target_encoder = ACTEncoder(config, n_layers_override=config.n_target_encoder_layers)
        state_dim = self.config.robot_state_feature.shape[0] if self.config.robot_state_feature else 0
        if state_dim == 0:
            raise ValueError("ACT-JEPA requires a robot state feature (proprioceptive state) for the target encoder.")
        self.target_state_proj = nn.Linear(state_dim, config.dim_model)
        self.register_buffer("target_pe", self._build_sinusoidal_pe(config.prediction_horizon, config.dim_model))
        self.register_buffer("predictor_pe", self._build_sinusoidal_pe(config.n_obs_mask_tokens, config.dim_model))

        # ---- Predictor: predicts latent observations from s_x ----
        self.obs_mask_tokens = nn.Parameter(
            torch.randn(1, config.n_obs_mask_tokens, config.dim_model) * 0.02
        )
        self.predictor = JEPACrossAttentionTransformer(config, n_layers=config.n_predictor_layers)

        # ---- Action decoder: predicts actions from s_x ----
        self.action_mask_tokens = nn.Parameter(
            torch.randn(1, config.chunk_size, config.dim_model) * 0.02
        )
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, config.chunk_size, config.dim_model) * 0.02
        )
        self.action_decoder = JEPACrossAttentionTransformer(config, n_layers=config.n_decoder_layers)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        # ---- VAE encoder: encodes [cls, robot_state, action_sequence] → latent z ----
        if config.use_vae:
            self.vae_encoder = ACTEncoder(config, n_layers_override=config.n_vae_encoder_layers)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0], config.dim_model
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_vae_tokens = 1 + config.chunk_size  # cls + action tokens
            if self.config.robot_state_feature:
                num_vae_tokens += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                self._build_sinusoidal_pe(num_vae_tokens, config.dim_model).unsqueeze(0),
            )
            # Project latent z into context encoder token space.
            self.ctx_latent_proj = nn.Linear(config.latent_dim, config.dim_model)
            # Learnable prior for inference (instead of hardcoded zeros).
            self.prior_z = nn.Parameter(torch.zeros(1, config.latent_dim))

        assert config.n_obs_mask_tokens == config.prediction_horizon, (
            f"n_obs_mask_tokens ({config.n_obs_mask_tokens}) must equal "
            f"prediction_horizon ({config.prediction_horizon})."
        )
        for (n_c, p_c), (n_t, p_t) in zip(
            self.context_encoder.named_parameters(), self.target_encoder.named_parameters()
        ):
            assert p_c.shape == p_t.shape, (
                f"Context encoder param '{n_c}' shape {p_c.shape} != "
                f"target encoder param '{n_t}' shape {p_t.shape}. "
                f"EMA requires identical architectures."
            )

        # Initialize custom parameters (PyTorch handles MHA/Linear defaults).
        nn.init.normal_(self.obs_mask_tokens, std=0.02)
        nn.init.normal_(self.action_mask_tokens, std=0.02)
        nn.init.normal_(self.action_pos_embed, std=0.02)

        # Initialize target as exact copy of context (BYOL/DINO standard).
        # Without this, first EMA step blends two random networks → unstable.
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self.target_state_proj.load_state_dict(self.ctx_robot_state_proj.state_dict())

        # Freeze entire target path (EMA-updated, not gradient-trained).
        # .data writes from update_target_encoder() bypass requires_grad.
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_state_proj.parameters():
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
        """Update target encoder parameters via exponential moving average of context encoder."""
        for param_c, param_t in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data.mul_(self.config.ema_decay).add_(param_c.data, alpha=1 - self.config.ema_decay)

        # EMA update target_state_proj from ctx_robot_state_proj (same architecture).
        if self.config.robot_state_feature:
            for param_c, param_t in zip(
                self.ctx_robot_state_proj.parameters(), self.target_state_proj.parameters()
            ):
                param_t.data.mul_(self.config.ema_decay).add_(param_c.data, alpha=1 - self.config.ema_decay)

    def _encode_context(self, batch: dict[str, Tensor], latent_sample: Tensor | None = None) -> Tensor:
        """Encode current observation into context representation s_x.

        Returns:
            s_x: (S_ctx, B, D) context tokens from the transformer encoder.
        """
        encoder_in_tokens = []
        encoder_in_pos_embed = []

        pos_idx = 0

        # Latent z token from VAE (first token).
        if self.config.use_vae and latent_sample is not None:
            latent_token = self.ctx_latent_proj(latent_sample).unsqueeze(0)  # (1, B, D)
            encoder_in_tokens.append(latent_token)
            encoder_in_pos_embed.append(self.ctx_1d_pos_embed.weight[pos_idx : pos_idx + 1].unsqueeze(1))
            pos_idx += 1

        # Robot state token.
        if self.config.robot_state_feature:
            obs_state = batch[OBS_STATE]
            # When observation_delta_indices loads multiple frames, obs_state is (B, T, state_dim).
            # Extract current state (t=0) for the context encoder.
            if obs_state.ndim == 3:
                obs_state = obs_state[:, 0]
            state_token = self.ctx_robot_state_proj(obs_state).unsqueeze(0)  # (1, B, D)
            encoder_in_tokens.append(state_token)
            encoder_in_pos_embed.append(self.ctx_1d_pos_embed.weight[pos_idx : pos_idx + 1].unsqueeze(1))
            pos_idx += 1

        # Environment state token.
        if self.config.env_state_feature:
            env_token = self.ctx_env_state_proj(batch[OBS_ENV_STATE]).unsqueeze(0)  # (1, B, D)
            encoder_in_tokens.append(env_token)
            encoder_in_pos_embed.append(self.ctx_1d_pos_embed.weight[pos_idx : pos_idx + 1].unsqueeze(1))
            pos_idx += 1

        # Image feature tokens.
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                # When observation_delta_indices loads multiple frames, img is (B, T, C, H, W).
                # Extract current frame (t=0) for the context encoder.
                if img.ndim == 5:
                    img = img[:, 0]
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.ctx_cam_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.ctx_img_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.append(cam_features)  # (HW, B, D)
                encoder_in_pos_embed.append(cam_pos_embed)

        # Concatenate along sequence dimension: (S_ctx, B, D).
        encoder_in_tokens = torch.cat(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.cat(encoder_in_pos_embed, dim=0)

        s_x = self.context_encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        return s_x, encoder_in_pos_embed

    def forward(
        self, batch: dict[str, Tensor], training: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Forward pass.

        During inference: returns predicted actions (B, chunk_size, action_dim).
        During training: returns (actions_hat, observations_hat, target_observations, mu, log_sigma_x2).
        """
        batch_size = batch[OBS_STATE].shape[0]

        # 0. VAE encoder: encode [cls, robot_state, action_sequence] → latent z
        mu = log_sigma_x2 = None
        if self.config.use_vae and training:
            obs_state = batch[OBS_STATE]
            if obs_state.ndim == 3:
                obs_state = obs_state[:, 0]
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, chunk_size, D)
            vae_tokens = [cls_embed]
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(obs_state).unsqueeze(1)
                vae_tokens.append(robot_state_embed)
            vae_tokens.append(action_embed)
            vae_input = torch.cat(vae_tokens, dim=1)  # (B, 1+S(+1), D)

            num_vae_tokens = vae_input.shape[1]
            pos_embed = self.vae_encoder_pos_enc[:, :num_vae_tokens, :]  # (1, N, D)

            cls_joint_is_pad = torch.full(
                (batch_size, 1 + (1 if self.config.robot_state_feature else 0)),
                False,
                device=obs_state.device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], dim=1)

            cls_token_out = self.vae_encoder(
                vae_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # (B, D) — cls token output
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            if self.config.vae_deterministic:
                latent_sample = mu
            else:
                latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        elif self.config.use_vae:
            latent_sample = self.prior_z.expand(batch_size, -1)
        else:
            latent_sample = None

        # 1. Encode context: current observation → s_x
        s_x, _ = self._encode_context(batch, latent_sample=latent_sample)

        # 2. Action decoder: s_x + action mask tokens → predicted actions
        action_mask = self.action_mask_tokens.expand(batch_size, -1, -1) + self.action_pos_embed  # (B, chunk_size, D)
        action_decoder_out = self.action_decoder(action_mask, s_x)  # (B, chunk_size, D)
        actions_hat = self.action_head(action_decoder_out)  # (B, chunk_size, action_dim)

        if not training:
            return actions_hat

        # 3. Target encoder: future state sequence → s^y (target for observation loss)
        # Entire target path is EMA-updated (BYOL style), no gradients needed.
        # observation_delta_indices=[0,1,...,prediction_horizon-1] loads T frames into OBS_STATE.
        # Index 0 = current state (used by context encoder); indices 1..T-1 = future states.
        obs_state_full = batch[OBS_STATE]
        if obs_state_full.ndim == 3:
            state_seq = obs_state_full[:, 1:]  # (B, T-1, state_dim)
        else:
            raise ValueError(
                f"ACT-JEPA training requires observation_delta_indices to load multiple frames. "
                f"Got obs_state shape {obs_state_full.shape}."
            )
        _, T, state_dim = state_seq.shape
        assert T == self.config.prediction_horizon, (
            f"Future state sequence length ({T}) must equal prediction_horizon "
            f"({self.config.prediction_horizon})."
        )
        assert state_dim == self.config.robot_state_feature.shape[0], (
            f"observation_state_sequence last dim ({state_dim}) must match "
            f"robot_state_feature dim ({self.config.robot_state_feature.shape[0]})."
        )
        state_tokens = self.target_state_proj(state_seq)  # (B, T, D)
        state_tokens = state_tokens.permute(1, 0, 2)  # (T, B, D)
        # Fixed sinusoidal PE registered as buffer, passed as pos_embed (not added to tokens).
        with torch.no_grad():
            target_observations = self.target_encoder(state_tokens, pos_embed=self.target_pe.unsqueeze(1))  # (T, B, D)
        target_observations = target_observations.permute(1, 0, 2)  # (B, T, D)

        # 4. Predictor: s_x + obs mask tokens → predicted s^y
        # Add temporal positional encoding so predictor knows which token = which future step.
        obs_mask = self.obs_mask_tokens.expand(batch_size, -1, -1) + self.predictor_pe.unsqueeze(0)  # (B, n_pred, D)
        observations_hat = self.predictor(obs_mask, s_x)  # (B, n_pred, D)

        return actions_hat, observations_hat, target_observations, mu, log_sigma_x2


class JEPACrossAttentionTransformer(nn.Module):
    """Cross-attention transformer block: cross-attend to context, then self-attention.

    Used by both the predictor and the action decoder.
    Mask tokens cross-attend to context encoder output, then pass through self-attention layers.
    """

    def __init__(self, config: ACTJEPAConfig, n_layers: int | None = None):
        super().__init__()
        n_layers = n_layers or config.n_decoder_layers

        # Cross-attention: mask tokens attend to context
        self.cross_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.query_norm = nn.LayerNorm(config.dim_model)
        self.ctx_norm = nn.LayerNorm(config.dim_model)
        self.cross_dropout = nn.Dropout(config.dropout)

        # Self-attention transformer layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        query_tokens: Tensor,
        context: Tensor,
    ) -> Tensor:
        """
        Args:
            query_tokens: (B, S_query, D) — mask tokens (action or observation).
            context: (S_ctx, B, D) — context encoder output tokens.

        Returns:
            (B, S_query, D) — output tokens.
        """
        # Rearrange query to (S_query, B, D) for nn.MHA
        q = query_tokens.permute(1, 0, 2)

        # Cross-attention: query attends to context
        skip = q
        q_norm = self.query_norm(q)
        c_norm = self.ctx_norm(context)
        cross_out = self.cross_attn(q_norm, c_norm, value=context)[0]
        q = skip + self.cross_dropout(cross_out)

        # Self-attention transformer layers in (S, B, D) format (batch_first=False)
        x = q
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x.permute(1, 0, 2)  # (S, B, D) → (B, S, D)


class ACTEncoder(nn.Module):
    """Transformer encoder with multiple layers and optional normalization."""

    def __init__(self, config: ACTJEPAConfig, n_layers_override: int | None = None):
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
    def __init__(self, config: ACTJEPAConfig):
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
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
