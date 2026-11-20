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
"""Configuration for the ACT-JEPA policy (Action Chunking Transformer with Joint Embedding Predictive Architecture)."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_jepa")
@dataclass
class ACTJEPAConfig(PreTrainedConfig):
    """Configuration for the ACT-JEPA policy.

    ACT-JEPA combines Action Chunking Transformer with Joint Embedding Predictive Architecture.
    It jointly optimizes action prediction (supervised) and latent observation prediction (self-supervised)
    in a single end-to-end training stage.

    The architecture consists of:
    - Context encoder: encodes current image + proprioceptive state into a shared latent representation.
    - Target encoder: encodes a sequence of future proprioceptive states into latent targets (EMA-updated).
    - Predictor: predicts future latent observations from the context representation.
    - Action decoder: predicts action sequences from the context representation.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation.
        normalization_mapping: Feature type to normalization mode mapping.
        vision_backbone: Name of the torchvision resnet backbone for encoding images.
        pretrained_backbone_weights: Pretrained weights for the backbone. None means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace ResNet's final stride with dilated convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads in the transformer blocks' multi-head attention.
        dim_feedforward: The FFN hidden dimension in the transformer blocks.
        feedforward_activation: The activation function in the transformer block's FFN layers.
        n_encoder_layers: The number of transformer layers for the context encoder.
        n_target_encoder_layers: The number of transformer layers for the target encoder.
        n_predictor_layers: The number of transformer layers for the predictor.
        n_decoder_layers: The number of transformer layers for the action decoder.
        prediction_horizon: Number of future steps to predict for the observation loss (n in paper).
        ema_decay: EMA decay rate for updating the target encoder from the context encoder.
        observation_loss_weight: Weight for the observation loss relative to the action loss.
        n_action_mask_tokens: Number of learnable mask tokens for the action decoder.
        n_obs_mask_tokens: Number of learnable mask tokens for the predictor.
        dropout: Dropout rate in transformer layers.
        temporal_ensemble_coeff: Coefficient for temporal ensembling. None disables it.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_target_encoder_layers: int = 4
    n_predictor_layers: int = 6
    n_decoder_layers: int = 1

    # JEPA-specific.
    prediction_horizon: int = 5
    ema_decay: float = 0.996
    observation_loss_weight: float = 1.0
    n_action_mask_tokens: int = 100
    n_obs_mask_tokens: int = 5

    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0
    vae_deterministic: bool = False

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> list:
        # Load current state (t=0) plus prediction_horizon future states (t=1..T).
        # Indices 0..prediction_horizon → prediction_horizon+1 frames total.
        return list(range(self.prediction_horizon + 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
