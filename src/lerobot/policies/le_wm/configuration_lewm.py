"""Configuration for the LE-WM policy (Latent Embedding World Model)."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("le_wm")
@dataclass
class LEWMConfig(PreTrainedConfig):
    """Configuration for the LE-WM (Latent Embedding World Model) policy.

    LE-WM is a JEPA-based world model that learns to predict future state embeddings
    from observations and actions. An action head is added on top to map predicted
    embeddings to actions for use in lerobot's training/evaluation pipeline.

    Args:
        n_obs_steps: Number of observation steps (current frame only).
        chunk_size: Size of the action prediction chunk.
        n_action_steps: Number of action steps to execute per policy invocation.
        normalization_mapping: Feature type to normalization mode mapping.

        # ViT encoder
        vit_model_name: HuggingFace model name for the ViT encoder.
        img_size: Input image resolution.
        freeze_encoder: Whether to freeze the ViT encoder weights.

        # World model dimensions
        embed_dim: Embedding dimension for the world model.
        history_size: Number of context frames for the AR predictor.
        num_preds: Number of prediction steps (offset for target embedding).

        # AR Predictor
        ar_depth: Number of transformer layers in the AR predictor.
        ar_heads: Number of attention heads.
        ar_mlp_dim: MLP hidden dimension in the predictor.
        ar_dim_head: Dimension per attention head.
        ar_dropout: Dropout rate in the predictor.

        # Projector / PredProj
        projector_hidden_dim: Hidden dimension for the projector MLP.
        pred_proj_hidden_dim: Hidden dimension for the prediction projector MLP.

        # SIGReg (world model regularization)
        sigreg_weight: Weight for the SIGReg loss.
        sigreg_knots: Number of knots for the SIGReg statistic.
        sigreg_num_proj: Number of random projections for SIGReg.

        # Action head
        action_head_type: Type of action head ("mlp" for now, extensible).
        action_head_hidden_dim: Hidden dimension for the MLP action head.

        # Loss weighting
        wm_loss_weight: Weight for the world model loss relative to action loss.

        # Training preset
        optimizer_lr: Learning rate.
        optimizer_weight_decay: Weight decay.
        optimizer_lr_backbone: Learning rate for the ViT backbone.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 1
    n_action_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ViT encoder.
    vit_model_name: str = "google/vit-base-patch16-224"
    img_size: int = 224
    freeze_encoder: bool = False

    # World model dimensions.
    embed_dim: int = 192
    history_size: int = 3
    num_preds: int = 1
    prediction_horizon: int = 5

    # AR Predictor.
    ar_depth: int = 6
    ar_heads: int = 16
    ar_mlp_dim: int = 2048
    ar_dim_head: int = 64
    ar_dropout: float = 0.1

    # Projector / PredProj.
    projector_hidden_dim: int = 2048
    pred_proj_hidden_dim: int = 2048

    # SIGReg.
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    # Action head.
    action_head_type: str = "mlp"
    action_head_hidden_dim: int = 512

    # Loss weighting.
    wm_loss_weight: float = 0.1

    # Training preset.
    optimizer_lr: float = 5e-5
    optimizer_weight_decay: float = 1e-3
    optimizer_lr_backbone: float = 5e-5

    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(self.prediction_horizon + 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
