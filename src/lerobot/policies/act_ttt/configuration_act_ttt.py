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
"""Configuration for the ACT-TTT policy (Action Chunking Transformer with Test-Time Training).

Replaces standard softmax self-attention with TTT (Test-Time Training) blocks
and replaces cross-attention with encoder-learned inner models.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("act_ttt")
@dataclass
class ACTTTTConfig(PreTrainedConfig):
    """Configuration for the ACT-TTT policy.

    ACT-TTT replaces:
    - Image backbone (ResNet) with ViTTT-style CNN stem + TTT blocks
    - Encoder/decoder self-attention with TTT blocks
    - Decoder cross-attention with encoder-learned inner models

    Args:
        n_obs_steps: Number of observation steps.
        chunk_size: Number of action steps to predict at once.
        n_action_steps: Number of action steps to execute per policy invocation.
        normalization_mapping: Feature type to normalization mode mapping.
        vision_backbone: Reserved for compatibility (not used, ViTTT backbone is used instead).
        image_size: Input image size (H, W).
        # ViTTT backbone
        backbone_embed_dim: Embedding dimension of the ViTTT backbone.
        backbone_depth: Number of TTT blocks in the backbone.
        backbone_num_heads: Number of attention heads in backbone TTT blocks.
        backbone_patch_size: Patch size for the CNN stem.
        backbone_mlp_ratio: MLP expansion ratio in backbone TTT blocks.
        # Transformer encoder (TTT)
        pre_norm: Whether to use pre-norm in transformer blocks.
        dim_model: Hidden dimension for encoder/decoder.
        n_heads: Number of heads in encoder/decoder TTT blocks.
        dim_feedforward: FFN hidden dimension in encoder/decoder.
        feedforward_activation: Activation function for FFN.
        n_encoder_layers: Number of TTT encoder layers.
        n_decoder_layers: Number of TTT decoder layers.
        ttt_inner_lr: Learning rate for TTT inner training loop.
        # VAE
        use_vae: Whether to use VAE.
        latent_dim: VAE latent dimension.
        n_vae_encoder_layers: Number of layers in VAE encoder.
        # Inference
        temporal_ensemble_coeff: Coefficient for temporal ensembling.
        # Training
        dropout: Dropout rate.
        kl_weight: Weight for KL divergence loss.
        optimizer_lr: Learning rate.
        optimizer_weight_decay: Weight decay.
        optimizer_lr_backbone: Learning rate for backbone.
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
    vision_backbone: str = "vittt"  # placeholder, actual backbone is ViTTT-style

    # ViTTT backbone settings
    image_size: tuple[int, int] = (480, 640)
    backbone_embed_dim: int = 256
    backbone_depth: int = 6
    backbone_num_heads: int = 8
    backbone_patch_size: int = 4
    backbone_mlp_ratio: float = 4.0

    # Transformer encoder/decoder (TTT) settings.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 2048
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 6
    n_decoder_layers: int = 2
    ttt_inner_lr: float = 0.3

    # MoE (Mixture-of-Experts) for modality-aware output modulation
    use_moe: bool = False
    moe_n_experts: int = 3
    moe_top_k: int = 1
    moe_balance_weight: float = 0.01

    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Gradient checkpointing to reduce memory usage
    use_gradient_checkpointing: bool = False

    # Training preset
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

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
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=1000,
            num_decay_steps=80000,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
        )

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
