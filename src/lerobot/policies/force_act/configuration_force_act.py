#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("force_act")
@dataclass
class ForceACTConfig(PreTrainedConfig):
    """Configuration class for the Force-ACT policy.

    Designed for dual-arm robotic arms with heterogeneous joint scales.
    Action space (18D, dual arm):
    Left arm:  [Δx, Δy, Δz, sin(Δroll), cos(Δroll), sin(Δpitch), cos(Δpitch), Δyaw, Δgripper]
    Right arm: [Δx, Δy, Δz, sin(Δroll), cos(Δroll), sin(Δpitch), cos(Δpitch), Δyaw, Δgripper]

    Key differences from standard ACT:
    - Delta action representation for all dimensions
    - Sin-Cos encoding for rotation axes (roll, pitch) with unit circle projection
    - Range normalization (MIN_MAX) for linear axes
    - Weighted loss for balancing dimension contributions

    Args:
        rotation_indices: Dual-arm sin/cos pairs.
        yaw_indices: Dual-arm yaw dims for extra loss weight.
        All other parameters match standard ACT configuration.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Rotation axis configuration for unit circle projection.
    # Each tuple is (sin_dim_index, cos_dim_index) in the action vector.
    # 18D layout: [左臂9D, 右臂9D]
    # Per arm: [Δx, Δy, Δz, sin(Δroll), cos(Δroll), sin(Δpitch), cos(Δpitch), Δyaw, Δgripper]
    #   index:   0    1    2      3          4          5           6           7      8
    # Right arm offset = 9
    rotation_indices: list[tuple[int, int]] = field(
        default_factory=lambda: [(3, 4), (5, 6), (12, 13), (14, 15)]
    )
    unit_circle_eps: float = 1e-6

    # Yaw dimension indices (to apply extra loss weight).
    yaw_indices: list[int] = field(
        default_factory=lambda: [7, 16]
    )

    # Weighted loss: extra multiplier for yaw dimensions.
    # Set to 1.0 to disable weighting.
    loss_weight_yaw: float = 4.0

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
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

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
                "`n_action_steps` must be 1 when using temporal ensembling."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= `chunk_size` ({self.chunk_size})."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
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
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
