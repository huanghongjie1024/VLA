#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Configuration for the Force-π₀ policy.

Force-π₀ extends the π₀ VLA model with a dedicated force/torque encoder
for contact-rich manipulation tasks, following the FAVLA approach for
force modality integration.
"""
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("force_pi0")
@dataclass
class ForcePI0Config(PreTrainedConfig):
    """Configuration class for the Force-π₀ policy.

    Force-π₀ extends π₀ with a force/torque encoder for contact-rich manipulation.
    It combines the VLM backbone + flow matching from π₀ with a dedicated force
    encoder inspired by FAVLA, action chunking from ACT, and temporal ensembling.

    Args:
        # VLM backbone
        vlm_model_id: HuggingFace model ID for the PaliGemma VLM backbone.
        max_seq_len: Maximum sequence length for the VLM.

        # LoRA
        lora_rank: Rank of the LoRA adaptation matrices.
        lora_alpha: LoRA scaling factor.

        # Action head
        chunk_size: Number of future action steps predicted at once (action chunking).
        n_action_steps: Number of action steps to execute per model invocation.
        action_dim: Dimensionality of the action space.

        # Flow matching
        flow_steps: Number of ODE integration steps for flow matching at inference.
        flow_sigma_min: Minimum noise sigma for flow matching.

        # Force encoder
        force_encoder_layers: Number of transformer layers in the force encoder.
        force_encoder_dim: Hidden dimension of the force encoder.
        force_encoder_heads: Number of attention heads in the force encoder.
        force_dim: Dimensionality of the raw force/torque data (default: 6 for F/T).
        force_window_size: Number of force samples per tokenization window.
        force_window_stride: Stride for sliding window tokenization.
        force_sample_rate: Expected force sensor sample rate in Hz.

        # Force fusion
        force_gate_init: Initial value for the force gate sigmoid bias.
        force_cross_attn_heads: Number of heads for cross-modal force attention.

        # Contact classifier
        use_contact_classifier: Whether to include the contact state classifier.

        # Temporal ensembling
        temporal_ensemble_coeff: Exponential decay coefficient for temporal ensembling.
            None disables temporal ensembling (requires n_action_steps > 1).

        # Training
        dropout: Dropout rate for transformer layers.
        optimizer_lr: Learning rate for non-backbone parameters.
        optimizer_lr_force: Learning rate for the force encoder.
        optimizer_lr_backbone: Learning rate for the LoRA parameters.
        optimizer_weight_decay: Weight decay for AdamW.

        # Loss weights
        flow_loss_weight: Weight for the flow matching loss.
        force_recon_loss_weight: Weight for the force feature reconstruction loss.
        contact_loss_weight: Weight for the contact state classification loss.
    """

    # Input / output structure.
    n_obs_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # VLM backbone.
    vlm_model_id: str = "google/paligemma2-3b-pt-224"
    max_seq_len: int = 256

    # LoRA.
    lora_rank: int = 16
    lora_alpha: float = 32.0

    # Action head.
    chunk_size: int = 50
    n_action_steps: int = 50
    action_dim: int = 7

    # Flow matching.
    flow_steps: int = 10
    flow_sigma_min: float = 0.001

    # Force encoder.
    force_encoder_layers: int = 6
    force_encoder_dim: int = 512
    force_encoder_heads: int = 8
    force_dim: int = 6
    force_window_size: int = 64
    force_window_stride: int = 32
    force_sample_rate: int = 100

    # Force fusion.
    force_gate_init: float = 0.0
    force_cross_attn_heads: int = 8

    # Contact classifier.
    use_contact_classifier: bool = True

    # Temporal ensembling (from ACT).
    temporal_ensemble_coeff: float | None = 0.01

    # Training.
    dropout: float = 0.1

    # Optimizer presets.
    optimizer_lr: float = 2.5e-5
    optimizer_lr_force: float = 1e-4
    optimizer_lr_backbone: float = 5e-5
    optimizer_weight_decay: float = 1e-4

    # Loss weights.
    flow_loss_weight: float = 1.0
    force_recon_loss_weight: float = 0.1
    contact_loss_weight: float = 0.05

    def __post_init__(self):
        super().__post_init__()

        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. "
                "The policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= "
                f"`chunk_size` ({self.chunk_size})."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )
        if self.force_window_size <= 0 or self.force_window_stride <= 0:
            raise ValueError(
                "Force window size and stride must be positive."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            warmup_steps=500,
            decay_steps=300_000,
            decay_lr=1e-6,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
