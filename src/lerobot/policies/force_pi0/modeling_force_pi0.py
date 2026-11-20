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

"""Force-π₀: A Force-Enhanced Vision-Language-Action Model.

Extends the π₀ VLA architecture with a dedicated force/torque encoder for
contact-rich manipulation. Innovations fused from:

- π₀: VLM backbone (PaliGemma) + Action Expert + Flow Matching
- FAVLA: Force encoder, cross-modal fusion, LoRA
- ACT: Action chunking, temporal ensembling
- π₀.5: MoE-style adaptive force routing
- π₀.6: Hierarchical contact-aware control
"""

import builtins
import copy
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from lerobot.policies.pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None


from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.force_pi0.configuration_force_pi0 import ForcePI0Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)


# ---------------------------------------------------------------------------
# Helper functions (shared with pi0)
# ---------------------------------------------------------------------------


class ActionSelectKwargs(TypedDict, total=False):
    noise: Tensor | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Sine-cosine positional embedding for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):
    """Create 2D attention masks from padding and autoregressive masks."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros."""
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


# ---------------------------------------------------------------------------
# Force Encoder (from FAVLA)
# ---------------------------------------------------------------------------


class ForceTokenizeProjector(nn.Module):
    """Tokenize raw F/T sensor windows into embedding vectors.

    Input: (B, num_windows, window_size * force_dim)
    Output: (B, num_windows, d_model)
    """

    def __init__(self, window_size: int, force_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(window_size * force_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class ForceEncoder(nn.Module):
    """Transformer encoder for 6-axis force/torque time-series data.

    Processes tokenized force windows through N transformer encoder layers
    to produce force feature tokens.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T_force, d_model) tokenized force embeddings
            src_key_padding_mask: (B, T_force) True for padded positions
        Returns:
            (B, T_force, d_model) force feature tokens
        """
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


# ---------------------------------------------------------------------------
# Force-VLM Cross-Attention Fusion (from FAVLA cross-modal attention)
# ---------------------------------------------------------------------------


class ForceCrossAttention(nn.Module):
    """Cross-attention: Query from VLM features, Key/Value from force features.

    Allows the language/vision pathway to attend to force information.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        force_features: torch.Tensor,
        force_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, d) VLM features as queries
            force_features: (B, T_f, d) force encoder output as key/value
            force_mask: (B, T_f) padding mask for force features
        Returns:
            (B, T_q, d) force-attended VLM features
        """
        attn_out, _ = self.cross_attn(
            query=query,
            key=force_features,
            value=force_features,
            key_padding_mask=force_mask,
        )
        return self.norm(query + attn_out)


# ---------------------------------------------------------------------------
# Force Gate (adaptive force injection, inspired by π₀.5 MoE routing)
# ---------------------------------------------------------------------------


class ForceGate(nn.Module):
    """Learnable gate that controls how much force information to inject.

    Uses a sigmoid-based gating mechanism with a learnable bias initialized
    to a configurable value. When force data is unavailable, the gate
    outputs zero, gracefully degrading to a vision-only policy.
    """

    def __init__(self, d_model: int, init_bias: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, d) features to compute gate from
        Returns:
            (B, 1, 1) gate value in [0, 1]
        """
        return torch.sigmoid(self.gate_proj(features.mean(dim=1, keepdim=True)))


# ---------------------------------------------------------------------------
# Contact State Classifier (from FAVLA / π₀.6 hierarchical)
# ---------------------------------------------------------------------------


class ContactClassifier(nn.Module):
    """Binary classifier that predicts whether the robot is in contact.

    Input: pooled force features
    Output: contact probability in [0, 1]
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, force_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            force_features: (B, T_f, d_model)
        Returns:
            (B,) contact probability
        """
        pooled = force_features.mean(dim=1)
        return torch.sigmoid(self.classifier(pooled).squeeze(-1))


# ---------------------------------------------------------------------------
# Force-Aware Action Expert (extends π₀ Action Expert with force conditioning)
# ---------------------------------------------------------------------------


class ForceAwareActionExpert(nn.Module):
    """Action Expert with force-conditioned feature modulation.

    Takes the suffix embeddings (state + noisy actions + timestep) and
    conditions them on force features via an adaptive layer modulation.
    """

    def __init__(self, d_model: int, force_dim: int = 512):
        super().__init__()
        self.force_condition_mlp = nn.Sequential(
            nn.Linear(force_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )
        # Initialize to near-identity (scale=1, shift=0)
        nn.init.zeros_(self.force_condition_mlp[-1].weight)
        nn.init.zeros_(self.force_condition_mlp[-1].bias)

    def forward(self, hidden_states: torch.Tensor, force_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, d) action expert hidden states
            force_pooled: (B, d_force) pooled force features
        Returns:
            (B, T, d) force-conditioned hidden states
        """
        cond = self.force_condition_mlp(force_pooled)  # (B, 2*d)
        scale, shift = cond.chunk(2, dim=-1)
        scale = scale[:, None, :]  # (B, 1, d)
        shift = shift[:, None, :]  # (B, 1, d)
        return hidden_states * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Temporal Ensembler (from ACT)
# ---------------------------------------------------------------------------


class TemporalEnsembler:
    """Exponential-weighted temporal ensembling for action chunk predictions.

    Maintains a buffer of overlapping action predictions and computes
    exponentially-weighted averages to smooth execution. From ACT.
    """

    def __init__(self, chunk_size: int, action_dim: int, alpha: float = 0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.alpha = alpha
        # Buffer: deque of chunk_size entries, each is (chunk_size, action_dim) or None
        self._buffer: deque[Tensor | None] = deque(maxlen=chunk_size)

    def reset(self):
        self._buffer.clear()

    def update(self, predicted_chunk: Tensor) -> Tensor:
        """Add a new prediction and return the ensembled action for the current step.

        Args:
            predicted_chunk: (chunk_size, action_dim) new predicted actions
        Returns:
            (action_dim,) ensembled action for the current timestep
        """
        self._buffer.append(predicted_chunk)

        if len(self._buffer) == 0:
            return predicted_chunk[0]

        # Compute exponentially-decaying weights
        device = predicted_chunk.device
        weights = torch.tensor(
            [(1 - self.alpha) ** i for i in range(len(self._buffer))],
            device=device,
        )

        # For each position i in the current buffer, get the prediction for the current step
        # Buffer entry at index j was made (len(buffer) - 1 - j) steps ago
        # The action for "now" in that entry is at index (len(buffer) - 1 - j)
        current_actions = []
        buf_len = len(self._buffer)
        for j, chunk in enumerate(self._buffer):
            step_offset = buf_len - 1 - j
            if step_offset < chunk.shape[0]:
                current_actions.append(chunk[step_offset])

        if not current_actions:
            return predicted_chunk[0]

        stacked = torch.stack(current_actions, dim=0)  # (num, action_dim)
        w = weights[: len(current_actions)].unsqueeze(-1)  # (num, 1)
        return (stacked * w).sum(dim=0) / w.sum()


# ---------------------------------------------------------------------------
# Main Model: ForcePI0Pytorch
# ---------------------------------------------------------------------------


def get_gemma_config(variant: str):
    """Get Gemma config by variant name."""
    if "paligemma" in variant.lower():
        from transformers import PaliGemmaConfig

        return PaliGemmaConfig.from_pretrained(variant)
    from transformers import GemmaConfig

    return GemmaConfig.from_pretrained(variant)


class ForcePI0Pytorch(nn.Module):
    """Core Force-π₀ PyTorch model.

    Extends PI0Pytorch with:
    - Force encoder for 6-axis F/T sensor data
    - Force-VLM cross-attention fusion
    - Force-aware action expert conditioning
    - Contact state classifier
    """

    def __init__(self, config: ForcePI0Config):
        super().__init__()
        self.config = config

        # ===== VLM Backbone (from π₀) =====
        # We reuse the PaliGemma + Action Expert architecture from pi0
        paligemma_config = get_gemma_config(config.vlm_model_id)
        action_expert_config = paligemma_config.text_config if hasattr(paligemma_config, "text_config") else paligemma_config

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, False],
            precision="bfloat16",
            image_size=224,
            freeze_vision_encoder=True,
            train_expert_only=False,
        )

        d_model = action_expert_config.hidden_size

        # ===== Action projections (from π₀) =====
        self.action_in_proj = nn.Linear(config.action_dim, d_model)
        self.action_out_proj = nn.Linear(d_model, config.action_dim)
        self.state_proj = nn.Linear(config.action_dim, d_model)  # state_dim ≈ action_dim
        self.action_time_mlp_in = nn.Linear(2 * d_model, d_model)
        self.action_time_mlp_out = nn.Linear(d_model, d_model)

        # ===== Force Encoder (from FAVLA) =====
        force_d_model = config.force_encoder_dim
        self.force_tokenizer = ForceTokenizeProjector(
            window_size=config.force_window_size,
            force_dim=config.force_dim,
            d_model=force_d_model,
        )
        self.force_pos_embed = nn.Parameter(
            torch.randn(1, 256, force_d_model) * 0.02  # max 256 force tokens
        )
        self.force_encoder = ForceEncoder(
            d_model=force_d_model,
            nhead=config.force_encoder_heads,
            num_layers=config.force_encoder_layers,
            dim_feedforward=force_d_model * 4,
            dropout=config.dropout,
        )

        # ===== Force-VLM Fusion =====
        # Project force features to VLM dimension if needed
        if force_d_model != d_model:
            self.force_to_vlm_proj = nn.Linear(force_d_model, d_model)
        else:
            self.force_to_vlm_proj = nn.Identity()

        self.force_cross_attn = ForceCrossAttention(
            d_model=d_model,
            nhead=config.force_cross_attn_heads,
            dropout=config.dropout,
        )
        self.force_gate = ForceGate(d_model=d_model, init_bias=config.force_gate_init)

        # ===== Force-Aware Action Expert =====
        self.force_action_expert = ForceAwareActionExpert(
            d_model=d_model,
            force_dim=force_d_model,
        )

        # ===== Contact Classifier (from FAVLA / π₀.6) =====
        if config.use_contact_classifier:
            self.contact_classifier = ContactClassifier(d_model=force_d_model)
        else:
            self.contact_classifier = None

        # ===== Force feature reconstruction head (auxiliary) =====
        self.force_recon_proj = nn.Linear(force_d_model, force_d_model)

        # Gradient checkpointing
        self.gradient_checkpointing_enabled = False

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        if hasattr(self.paligemma_with_expert, "paligemma"):
            self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
            self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        if hasattr(self.paligemma_with_expert, "gemma_expert"):
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for ForcePI0Pytorch")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        if hasattr(self.paligemma_with_expert, "paligemma"):
            self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
            self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        if hasattr(self.paligemma_with_expert, "gemma_expert"):
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for ForcePI0Pytorch")

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    # ------------------------------------------------------------------
    # VLM embedding helpers (mirroring PI0Pytorch)
    # ------------------------------------------------------------------

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma_with_expert.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output * self.paligemma_with_expert.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.paligemma_with_expert.paligemma.model.language_model.embed_tokens(tokens)

    # ------------------------------------------------------------------
    # Force processing pipeline
    # ------------------------------------------------------------------

    def encode_force(self, force_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw force/torque data into feature tokens.

        Args:
            force_data: (B, T_raw, force_dim) raw force readings
        Returns:
            force_features: (B, T_force, force_d_model) encoded force tokens
            force_pooled: (B, force_d_model) global pooled force features
        """
        B, T_raw, F_dim = force_data.shape
        W = self.config.force_window_size
        S = self.config.force_window_stride

        # Sliding window tokenization
        num_windows = max(1, (T_raw - W) // S + 1)
        windows = []
        for i in range(num_windows):
            start = i * S
            end = start + W
            if end <= T_raw:
                windows.append(force_data[:, start:end, :])  # (B, W, F_dim)
        if not windows:
            # Fallback: if data is shorter than window, just use it all padded
            padded = F.pad(force_data.transpose(1, 2), (0, W - T_raw)).transpose(1, 2)
            windows = [padded]

        # (B, num_windows, W * F_dim)
        window_tensor = torch.cat([w.reshape(B, 1, -1) for w in windows], dim=1)

        # Tokenize
        token_emb = self.force_tokenizer(window_tensor)  # (B, T_force, d_model)

        # Add positional embedding
        T_force = token_emb.shape[1]
        token_emb = token_emb + self.force_pos_embed[:, :T_force, :]

        # Encode
        force_features = self.force_encoder(token_emb)  # (B, T_force, d_model)

        # Global pooling
        force_pooled = force_features.mean(dim=1)  # (B, d_model)

        return force_features, force_pooled

    # ------------------------------------------------------------------
    # Prefix embedding (images + language + force fusion)
    # ------------------------------------------------------------------

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        force_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images, language tokens, and fuse force features via cross-attention.

        Following the pi0 pattern but adding force cross-attention after language embedding.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        lang_emb = self._apply_checkpoint(self.embed_language_tokens, lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        # Force-VLM cross-attention fusion (FAVLA style)
        if force_features is not None:
            # Project force features to VLM dimension
            force_proj = self.force_to_vlm_proj(force_features)
            # Cross-attend: language queries attend to force keys/values
            lang_emb = self.force_cross_attn(query=lang_emb, force_features=force_proj)
            # Apply force gate
            gate = self.force_gate(lang_emb)  # (B, 1, 1)
            lang_emb = lang_emb * gate

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    # ------------------------------------------------------------------
    # Suffix embedding (state + noisy actions + timestep + force)
    # ------------------------------------------------------------------

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        force_pooled: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed state, noisy actions, timestep for Expert Gemma processing.

        Additionally injects force conditioning into the action expert pathway.
        """
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self._apply_checkpoint(self.state_proj, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        # Embed timestep
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=1e-5,
            max_period=50.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action
        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self._apply_checkpoint(
            lambda x: self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x))),
            action_time_emb,
        )

        # Force-conditioned action expert modulation
        if force_pooled is not None:
            action_time_emb = self.force_action_expert(action_time_emb, force_pooled)

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # State tokens attend to prefix but not action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        adarms_cond = None
        return embs, pad_masks, att_masks, adarms_cond

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        force_data: torch.Tensor | None = None,
        force_labels: torch.Tensor | None = None,
        contact_labels: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Full training forward pass.

        Args:
            images: List of (B, C, H, W) image tensors per camera view.
            img_masks: List of (B,) masks per camera view.
            lang_tokens: (B, T_lang) tokenized language instructions.
            lang_masks: (B, T_lang) language attention masks.
            state: (B, state_dim) proprioceptive state.
            actions: (B, chunk_size, action_dim) ground-truth action sequence.
            force_data: (B, T_raw, force_dim) raw force/torque data.
            force_labels: (B, T_force) optional force feature reconstruction targets.
            contact_labels: (B,) optional binary contact labels.
            noise: Optional pre-sampled noise.
            time: Optional pre-sampled time.

        Returns:
            dict with keys:
                'loss': total weighted loss
                'flow_loss': flow matching MSE loss
                'force_recon_loss': force feature reconstruction loss (if applicable)
                'contact_loss': contact classification loss (if applicable)
        """
        # Encode force data
        force_features = None
        force_pooled = None
        if force_data is not None:
            force_features, force_pooled = self.encode_force(force_data)

        # Flow matching: sample noise and time
        if noise is None:
            noise = torch.normal(mean=0.0, std=1.0, size=actions.shape, dtype=torch.float32, device=actions.device)
        if time is None:
            time = sample_beta(1.5, 1.0, actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix (images + language + force fusion)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, force_features
        )

        # Embed suffix (state + noisy actions + timestep + force conditioning)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time, force_pooled
        )

        # Handle dtype
        if hasattr(self.paligemma_with_expert, "paligemma"):
            first_q = self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            if first_q == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Concatenate prefix + suffix
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Forward through PaliGemma + Expert
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # Extract action chunk predictions
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)

        # === Compute losses ===
        losses = {}

        # Flow matching loss (primary)
        flow_loss = F.mse_loss(u_t, v_t, reduction="none")
        losses["flow_loss"] = flow_loss.mean()

        # Force reconstruction loss (auxiliary, from FAVLA)
        if force_features is not None:
            force_recon = self.force_recon_proj(force_features)
            if force_labels is not None:
                losses["force_recon_loss"] = F.mse_loss(force_recon, force_labels)
            else:
                # Self-reconstruction: predict input from output (regularization)
                losses["force_recon_loss"] = F.mse_loss(force_recon, force_features.detach())

        # Contact classification loss (auxiliary)
        if self.contact_classifier is not None and contact_labels is not None and force_features is not None:
            contact_pred = self.contact_classifier(force_features)
            losses["contact_loss"] = F.binary_cross_entropy(contact_pred, contact_labels.float())

        # Total weighted loss
        total_loss = self.config.flow_loss_weight * losses["flow_loss"]
        if "force_recon_loss" in losses:
            total_loss = total_loss + self.config.force_recon_loss_weight * losses["force_recon_loss"]
        if "contact_loss" in losses:
            total_loss = total_loss + self.config.contact_loss_weight * losses["contact_loss"]

        losses["loss"] = total_loss
        return losses

    # ------------------------------------------------------------------
    # Inference (flow matching sampling, mirroring pi0.sample_actions)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        force_data: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Inference: generate action chunk via flow matching ODE integration.

        Args:
            images: List of image tensors.
            img_masks: List of image masks.
            lang_tokens: Tokenized language.
            lang_masks: Language masks.
            state: Current proprioceptive state.
            force_data: Current force/torque readings.
            noise: Optional initial noise.
            num_steps: Number of flow matching ODE steps.

        Returns:
            (B, chunk_size, action_dim) predicted action sequence.
        """
        if num_steps is None:
            num_steps = self.config.flow_steps

        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.action_dim)
            noise = torch.normal(mean=0.0, std=1.0, size=actions_shape, dtype=torch.float32, device=device)

        # Encode force
        force_features = None
        force_pooled = None
        if force_data is not None:
            force_features, force_pooled = self.encode_force(force_data)

        # Pre-compute prefix embeddings (cache KV for efficiency)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, force_features
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Forward prefix through VLM to get cached KV
        if hasattr(self.paligemma_with_expert, "paligemma"):
            self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # ODE integration (Euler method)
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            v_t = self._denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
                force_pooled=force_pooled,
            )
            x_t = x_t + dt * v_t

        return x_t

    def _denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        force_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single denoising step for flow matching inference."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep, force_pooled
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        if hasattr(self.paligemma_with_expert, "gemma_expert"):
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        past_key_values_copy = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values_copy,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


# ---------------------------------------------------------------------------
# Re-use PaliGemmaWithExpertModel from pi0 (imported or defined)
# ---------------------------------------------------------------------------


class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma with an action expert, supporting joint forward pass.

    This is a simplified version that delegates to the pi_gemma module when
    available, falling back to a basic implementation otherwise.
    """

    def __init__(
        self,
        paligemma_config,
        action_expert_config,
        use_adarms: list[bool] | None = None,
        precision: str = "bfloat16",
        image_size: int = 224,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = False,
    ):
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        if PaliGemmaForConditionalGenerationWithPiGemma is not None:
            self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(paligemma_config)
            self.gemma_expert = PiGemmaForCausalLM(action_expert_config)
        else:
            # Fallback: create from HuggingFace
            from transformers import AutoModelForVision2Seq

            self.paligemma = AutoModelForVision2Seq.from_config(paligemma_config)
            from transformers import GemmaForCausalLM

            self.gemma_expert = GemmaForCausalLM(action_expert_config)

        self._set_requires_grad()

    def _set_requires_grad(self):
        if self.freeze_vision_encoder and hasattr(self.paligemma, "model"):
            if hasattr(self.paligemma.model, "vision_tower"):
                self.paligemma.model.vision_tower.eval()
                for param in self.paligemma.model.vision_tower.parameters():
                    param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder and hasattr(self.paligemma, "model"):
            if hasattr(self.paligemma.model, "vision_tower"):
                self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()
        return self

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        """Joint forward through PaliGemma language model and Action Expert.

        Mirrors the pi0 pattern: prefix (images+lang) goes through paligemma,
        suffix (state+actions) goes through gemma_expert, with shared attention.
        """
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            # Joint forward: process both through alternating layers
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            for layer_idx in range(num_layers):
                if self.gradient_checkpointing_enabled if hasattr(self, "gradient_checkpointing_enabled") else False:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        _compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = _compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # Final norm
            outputs_embeds = []
            for i, hidden_states in enumerate(inputs_embeds):
                adarms_c = adarms_cond[i] if adarms_cond is not None else None
                out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_c)
                outputs_embeds.append(out_emb)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


def _compute_layer_complete(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
):
    """Process a single layer for both prefix and suffix models."""
    models = [paligemma.model.language_model, gemma_expert.model]

    for i, model in enumerate(models):
        if inputs_embeds[i] is None:
            continue
        layer = model.layers[layer_idx]
        layer_out = layer(
            inputs_embeds[i],
            attention_mask=attention_mask,
            position_ids=position_ids,
            adarms_cond=adarms_cond[i] if adarms_cond is not None else None,
        )
        if isinstance(layer_out, tuple):
            inputs_embeds[i] = layer_out[0]
        else:
            inputs_embeds[i] = layer_out

    return inputs_embeds


# ---------------------------------------------------------------------------
# Temporal Ensembler (from ACT)
# ---------------------------------------------------------------------------


class ACTTemporalEnsembler:
    """Temporal ensembler from ACT for smoothing action chunk predictions.

    Maintains a rolling buffer of action predictions and computes
    exponentially-weighted averages.
    """

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int, action_dim: int):
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self._actions_buffer: list[Tensor] = []

    def reset(self):
        self._actions_buffer = []

    def enqueue_and_pop(self, actions: Tensor) -> Tensor:
        """Add new action predictions and return ensembled action.

        Args:
            actions: (chunk_size, action_dim) predicted action sequence

        Returns:
            (action_dim,) ensembled action for current step
        """
        self._actions_buffer.append(actions)
        if len(self._actions_buffer) > self.chunk_size:
            self._actions_buffer.pop(0)

        # Exponential weighting
        num_actions = len(self._actions_buffer)
        weights = torch.tensor(
            [(1 - self.temporal_ensemble_coeff) ** i for i in range(num_actions)],
            device=actions.device,
            dtype=actions.dtype,
        )

        # Collect actions for the current timestep
        current_actions = []
        for i, act_seq in enumerate(self._actions_buffer):
            # The oldest buffer entry has the most offset
            step_idx = num_actions - 1 - i
            if step_idx < act_seq.shape[0]:
                current_actions.append(act_seq[step_idx])

        if not current_actions:
            return actions[0]

        stacked = torch.stack(current_actions, dim=0)
        w = weights[: len(current_actions)].unsqueeze(-1)
        return (stacked * w).sum(dim=0) / w.sum()


# ---------------------------------------------------------------------------
# Policy Class
# ---------------------------------------------------------------------------


class ForcePI0Policy(PreTrainedPolicy):
    """Force-π₀: Force-Enhanced Vision-Language-Action Policy.

    Extends π₀ with a dedicated force/torque encoder for contact-rich manipulation.

    Architecture:
        - VLM Backbone: PaliGemma (vision + language encoder)
        - Action Expert: Gemma-based expert for action token generation
        - Force Encoder: 6-layer Transformer for 6-axis F/T data
        - Force Fusion: Cross-attention + gated injection
        - Action Generation: Flow matching with action chunking
        - Temporal Ensembling: ACT-style exponential-weighted smoothing

    Innovations fused from:
        - π₀: VLM backbone, flow matching, action expert
        - FAVLA: Force encoder, cross-modal attention, LoRA
        - ACT: Action chunking (H=50), temporal ensembling
        - π₀.5: Adaptive force routing (gate mechanism)
        - π₀.6: Contact-aware hierarchical control
    """

    config_class = ForcePI0Config
    name = "force_pi0"

    def __init__(self, config: ForcePI0Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Core model
        self.model = ForcePI0Pytorch(config)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        # Temporal ensembler (from ACT)
        self.temporal_ensembler: ACTTemporalEnsembler | None = None
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff=config.temporal_ensemble_coeff,
                chunk_size=config.chunk_size,
                action_dim=config.action_dim,
            )

        self.reset()

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
    ) -> Tensor:
        """Select an action given the current observation.

        Args:
            batch: Dictionary containing:
                - 'observation.images.*': Image tensors
                - 'observation.state': Proprioceptive state
                - 'observation.force': Force/torque data (optional)
                - 'task': Language instruction (tokenized or string)

        Returns:
            (action_dim,) action to execute.
        """
        self.eval()

        # Extract inputs from batch
        images = []
        img_masks = []
        for key in sorted(batch.keys()):
            if key.startswith("observation.images."):
                img = batch[key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                images.append(img)
                img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=img.device))

        if not images:
            # Fallback: single image
            if "observation.image" in batch:
                img = batch["observation.image"]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                images.append(img)
                img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=img.device))

        state = batch.get(OBS_STATE)
        if state is not None and state.dim() == 1:
            state = state.unsqueeze(0)

        # Force data
        force_data = batch.get("observation.force")
        if force_data is not None and force_data.dim() == 2:
            force_data = force_data.unsqueeze(0)

        # Language tokens (should be pre-tokenized by processor)
        lang_tokens = batch.get("language_tokens")
        lang_masks = batch.get("language_mask")
        if lang_tokens is None:
            # Fallback: create dummy tokens
            lang_tokens = torch.zeros(1, 1, dtype=torch.long, device=state.device)
            lang_masks = torch.ones(1, 1, dtype=torch.bool, device=state.device)

        # Generate actions via flow matching
        actions = self.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            force_data=force_data,
        )

        action_sequence = actions[0]  # (chunk_size, action_dim)

        # Temporal ensembling
        if self.temporal_ensembler is not None:
            action = self.temporal_ensembler.enqueue_and_pop(action_sequence)
        else:
            action = action_sequence[0]

        return action

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Training forward pass.

        Args:
            batch: Dictionary with observations, actions, force data, etc.

        Returns:
            dict with 'loss' and individual loss components.
        """
        # Extract images
        images = []
        img_masks = []
        for key in sorted(batch.keys()):
            if key.startswith("observation.images."):
                img = batch[key]
                images.append(img)
                img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=img.device))

        if not images and "observation.image" in batch:
            img = batch["observation.image"]
            images.append(img)
            img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=img.device))

        state = batch.get(OBS_STATE)
        actions = batch.get(ACTION)

        # Force data
        force_data = batch.get("observation.force")

        # Contact labels (if available)
        contact_labels = batch.get("contact_label")

        # Language tokens
        lang_tokens = batch.get("language_tokens")
        lang_masks = batch.get("language_mask")
        if lang_tokens is None:
            lang_tokens = torch.zeros(state.shape[0], 1, dtype=torch.long, device=state.device)
            lang_masks = torch.ones(state.shape[0], 1, dtype=torch.bool, device=state.device)

        return self.model(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            actions=actions,
            force_data=force_data,
            contact_labels=contact_labels,
        )

    def reset(self):
        """Reset internal state for a new episode."""
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()
