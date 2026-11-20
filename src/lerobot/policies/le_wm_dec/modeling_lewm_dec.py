"""LE-WM with cross-attention action decoder.

Replaces the MLP action head with a cross-attention decoder that attends to
the predictor's output embeddings, similar to ACT-JEPA's action decoder.
"""

from __future__ import annotations

import math
from collections import deque

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.le_wm_dec.configuration_lewm_dec import LEWMDecConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# World model components (same as le_wm)
# ---------------------------------------------------------------------------


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


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (einops.rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = einops.rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        block_class: type = Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        for _ in range(depth):
            self.layers.append(block_class(hidden_dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class Embedder(nn.Module):
    """Embeds actions via Conv1d + MLP."""

    def __init__(self, input_dim: int = 10, smoothed_dim: int = 10, emb_dim: int = 10, mlp_scale: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D) -> (B, T, emb_dim)"""
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim, hidden_dim, output_dim or input_dim,
            depth, heads, dim_head, mlp_dim, dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """x: (B, T, d), c: (B, T, action_dim)"""
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        return self.transformer(x, c)


class ViTEncoder(nn.Module):
    """ViT encoder using HuggingFace transformers ViTModel."""

    def __init__(self, model_name: str = "google/vit-base-patch16-224", freeze: bool = False,
                 target_size: int = 224):
        super().__init__()
        from transformers import ViTModel

        self.vit = ViTModel.from_pretrained(model_name)
        self.embed_dim = self.vit.config.hidden_size
        self.target_size = target_size

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) normalized image tensor.
        Returns:
            cls_emb: (B, hidden_size) CLS token embedding.
        """
        if pixel_values.shape[-1] != self.target_size or pixel_values.shape[-2] != self.target_size:
            pixel_values = F.interpolate(
                pixel_values, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False
            )
        output = self.vit(pixel_values=pixel_values)
        return output.last_hidden_state[:, 0]


# ---------------------------------------------------------------------------
# Cross-Attention Action Decoder (replaces MLP action head)
# ---------------------------------------------------------------------------


class CrossAttentionDecoder(nn.Module):
    """Cross-attention action decoder: query tokens attend to context embeddings.

    Similar to ACT-JEPA's action decoder.
    """

    def __init__(self, embed_dim: int, action_dim: int, n_layers: int, n_heads: int,
                 dim_feedforward: int, dropout: float, chunk_size: int):
        super().__init__()
        self.action_mask_tokens = nn.Parameter(torch.randn(1, chunk_size, embed_dim) * 0.02)
        self.action_pos_embed = nn.Parameter(torch.randn(1, chunk_size, embed_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.ctx_norm = nn.LayerNorm(embed_dim)
        self.cross_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.action_head = nn.Linear(embed_dim, action_dim)

        nn.init.normal_(self.action_mask_tokens, std=0.02)
        nn.init.normal_(self.action_pos_embed, std=0.02)

    def forward(self, context: Tensor) -> Tensor:
        """
        Args:
            context: (B, T_ctx, D) — predictor output embeddings.
        Returns:
            actions: (B, chunk_size, action_dim)
        """
        B = context.shape[0]
        query = self.action_mask_tokens.expand(B, -1, -1) + self.action_pos_embed  # (B, chunk_size, D)

        # Cross-attention: query attends to context
        q = query.permute(1, 0, 2)  # (chunk_size, B, D)
        c = context.permute(1, 0, 2)  # (T_ctx, B, D)
        skip = q
        q_norm = self.query_norm(q)
        c_norm = self.ctx_norm(c)
        cross_out = self.cross_attn(q_norm, c_norm, value=c)[0]
        q = skip + self.cross_dropout(cross_out)

        # Self-attention layers
        x = q
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.permute(1, 0, 2)  # (B, chunk_size, D)
        return self.action_head(x)


class DecoderLayer(nn.Module):
    """Standard transformer decoder layer (self-attention + FFN)."""

    def __init__(self, dim: int, heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x_norm = self.norm1(x)
        x = self.self_attn(x_norm, x_norm, value=x_norm)[0]
        x = skip + self.dropout1(x)
        skip = x
        x_norm = self.norm2(x)
        x = self.linear2(self.dropout(F.gelu(self.linear1(x_norm))))
        x = skip + self.dropout2(x)
        return x


# ---------------------------------------------------------------------------
# LEWM Model with cross-attention decoder
# ---------------------------------------------------------------------------


class LEWMDecModel(nn.Module):
    """LE-WM world model with cross-attention action decoder."""

    def __init__(self, config: LEWMDecConfig, action_dim: int):
        super().__init__()
        self.config = config

        # Image encoder (ViT)
        self.encoder = ViTEncoder(
            model_name=config.vit_model_name,
            freeze=config.freeze_encoder,
        )
        encoder_out_dim = self.encoder.embed_dim

        # Project encoder output to embed_dim
        self.projector = MLP(
            input_dim=encoder_out_dim,
            hidden_dim=config.projector_hidden_dim,
            output_dim=config.embed_dim,
        )

        # Action encoder
        self.action_encoder = Embedder(
            input_dim=action_dim,
            smoothed_dim=action_dim,
            emb_dim=config.embed_dim,
        )

        # AR Predictor
        self.predictor = ARPredictor(
            num_frames=max(config.chunk_size, config.prediction_horizon),
            input_dim=config.embed_dim,
            hidden_dim=config.embed_dim,
            output_dim=config.embed_dim,
            depth=config.ar_depth,
            heads=config.ar_heads,
            mlp_dim=config.ar_mlp_dim,
            dim_head=config.ar_dim_head,
            dropout=config.ar_dropout,
        )

        # Prediction projector
        self.pred_proj = MLP(
            input_dim=config.embed_dim,
            hidden_dim=config.pred_proj_hidden_dim,
            output_dim=config.embed_dim,
        )

        # Cross-attention action decoder
        self.action_decoder = CrossAttentionDecoder(
            embed_dim=config.embed_dim,
            action_dim=action_dim,
            n_layers=config.n_decoder_layers,
            n_heads=config.n_decoder_heads,
            dim_feedforward=config.decoder_dim_feedforward,
            dropout=config.decoder_dropout,
            chunk_size=config.chunk_size,
        )

        # SIGReg regularizer
        self.sigreg = SIGReg(
            knots=config.sigreg_knots,
            num_proj=config.sigreg_num_proj,
        )

    def encode(self, images: Tensor, actions: Tensor | None = None) -> dict[str, Tensor]:
        """Encode images and optionally actions into embeddings."""
        single_frame = images.ndim == 4
        if single_frame:
            images = images.unsqueeze(1)

        B, T = images.shape[:2]
        flat_imgs = einops.rearrange(images, "b t c h w -> (b t) c h w")
        cls_emb = self.encoder(flat_imgs)
        emb = self.projector(cls_emb)
        emb = einops.rearrange(emb, "(b t) d -> b t d", b=B)

        result: dict[str, Tensor] = {"emb": emb}

        if actions is not None:
            result["act_emb"] = self.action_encoder(actions)

        return result

    def predict(self, ctx_emb: Tensor, ctx_act_emb: Tensor) -> Tensor:
        """Predict next-step embedding from context."""
        preds = self.predictor(ctx_emb, ctx_act_emb)
        return self.pred_proj(preds)

    def compute_world_model_loss(
        self, pred_emb: Tensor, target_emb: Tensor, emb_for_sigreg: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute world model loss (MSE + SIGReg)."""
        pred_loss = (pred_emb - target_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb_for_sigreg.transpose(0, 1))
        total = pred_loss + self.config.sigreg_weight * sigreg_loss
        return total, pred_loss, sigreg_loss


# ---------------------------------------------------------------------------
# LEWM Dec Policy
# ---------------------------------------------------------------------------


class LEWMDecPolicy(PreTrainedPolicy):
    """LE-WM Policy with cross-attention action decoder."""

    config_class = LEWMDecConfig
    name = "le_wm_dec"

    def __init__(self, config: LEWMDecConfig, **kwargs):
        super().__init__(config)
        self.config = config

        action_feature = config.action_feature
        if action_feature is None:
            raise ValueError("LE-WM Dec requires an action output feature.")
        action_dim = action_feature.shape[0]

        self.model = LEWMDecModel(config, action_dim=action_dim)
        self.reset()

    def get_optim_params(self) -> list[dict]:
        return [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("model.encoder") and p.requires_grad
                ],
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if n.startswith("model.encoder") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _prepare_images(self, batch: dict[str, Tensor], single_frame: bool = False) -> Tensor:
        if self.config.image_features:
            images = [batch[key] for key in self.config.image_features]
            img = images[0]  # (B, C, H, W) or (B, T, C, H, W)
            if single_frame and img.ndim == 5:
                img = img[:, -1]  # (B, C, H, W)
            # Resize to match ViT expected input size
            h, w = img.shape[-2], img.shape[-1]
            target = self.config.img_size
            if h != target or w != target:
                img = F.interpolate(
                    img.reshape(-1, *img.shape[-3:]),
                    size=(target, target), mode="bilinear", align_corners=False
                ).reshape(*img.shape[:-2], target, target)
            return img
        elif self.config.env_state_feature:
            raise NotImplementedError("LE-WM Dec currently requires image inputs.")
        raise ValueError("No image features found in config.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        images = self._prepare_images(batch)  # (B, T, C, H, W)
        actions = batch[ACTION]

        if images.ndim != 5:
            raise ValueError(
                f"LE-WM Dec training requires multi-frame images (B, T, C, H, W). Got shape {images.shape}."
            )

        B, T_total = images.shape[:2]
        T_target = T_total - 1

        # Encode current frame (t=0) as context
        current_img = images[:, 0]  # (B, C, H, W)
        ctx_output = self.model.encode(current_img)
        ctx_emb = ctx_output["emb"]  # (B, 1, embed_dim)

        # Encode future frames (t=1..T) as targets
        future_imgs = images[:, 1:]  # (B, T_target, C, H, W)
        tgt_output = self.model.encode(future_imgs)
        target_emb = tgt_output["emb"]  # (B, T_target, embed_dim)

        # Encode actions
        act_emb = self.model.action_encoder(actions)  # (B, chunk_size, embed_dim)

        # Predict: context + actions -> predicted future embeddings
        ctx_rep = ctx_emb.expand(-1, T_target, -1)  # (B, T_target, embed_dim)
        ctx_act = act_emb[:, :T_target]  # (B, T_target, embed_dim)
        pred_emb = self.model.predict(ctx_rep, ctx_act)  # (B, T_target, embed_dim)

        # World model loss: predicted future vs actual future
        wm_loss, pred_loss, sigreg_loss = self.model.compute_world_model_loss(
            pred_emb, target_emb, pred_emb
        )

        # Action loss: use cross-attention decoder
        pred_actions = self.model.action_decoder(pred_emb)  # (B, T_target, action_dim)
        # Truncate to match the shorter dimension
        min_len = min(pred_actions.shape[1], actions.shape[1])
        pred_actions = pred_actions[:, :min_len]  # (B, min_len, action_dim)
        actions = actions[:, :min_len]  # (B, min_len, action_dim)

        action_mask = ~batch.get("action_is_pad", torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device))[:, :min_len]
        action_mask = action_mask.unsqueeze(-1).expand_as(actions)
        action_loss = F.l1_loss(pred_actions, actions, reduction="none")
        action_loss = (action_loss * action_mask).sum() / action_mask.sum().clamp(min=1)

        total_loss = action_loss + self.config.wm_loss_weight * wm_loss

        loss_dict = {
            "action_loss": action_loss.item(),
            "wm_loss": wm_loss.item(),
            "pred_loss": pred_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        images = self._prepare_images(batch, single_frame=True)  # (B, C, H, W)
        output = self.model.encode(images)
        emb = output["emb"]

        B = emb.shape[0]
        dummy_act_emb = torch.zeros(
            B, self.config.history_size, self.config.embed_dim,
            device=emb.device, dtype=emb.dtype,
        )

        ctx_emb = emb.expand(-1, self.config.history_size, -1)
        pred_emb = self.model.predict(ctx_emb, dummy_act_emb)

        actions = self.model.action_decoder(pred_emb)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
