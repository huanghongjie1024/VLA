"""Configuration for LE-WM with cross-attention action decoder."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("le_wm_dec")
@dataclass
class LEWMDecConfig(PreTrainedConfig):
    """LE-WM with cross-attention action decoder instead of MLP action head."""

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

    # Action decoder (cross-attention).
    n_decoder_layers: int = 1
    n_decoder_heads: int = 8
    decoder_dim_feedforward: int = 512
    decoder_dropout: float = 0.1

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
