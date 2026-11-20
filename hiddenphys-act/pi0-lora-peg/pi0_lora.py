"""OpenPI pi0 LoRA adapter for the HiddenPhys peg-in-hole task."""
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import sys

import numpy as np

harness = Path(os.environ["FORCEHARNESS"])
sys.path[:0] = [str(harness), str(harness / "scripts/baselines/pi05")]

import pi05_fh as fh
from openpi.models.pi0_config import Pi0Config
from openpi.training.weight_loaders import CheckpointWeightLoader


@dataclass(frozen=True)
class PegDataConfig(fh.FhDataConfig):
    def create(self, assets_dirs, model_config):
        data = super().create(assets_dirs, model_config)
        return replace(data, use_quantile_norm=True)


original_make_config = fh.make_config


def make_config(*args, **kwargs):
    cfg = original_make_config(*args, **kwargs)
    model = Pi0Config(
        pi05=False,
        action_horizon=cfg.model.action_horizon,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    data = PegDataConfig(
        repo_id=cfg.data.repo_id,
        assets=cfg.data.assets,
        base_config=cfg.data.base_config,
        use_force=cfg.data.use_force,
    )
    return replace(
        cfg,
        model=model,
        data=data,
        weight_loader=CheckpointWeightLoader(os.environ["PI0_PARAMS"]),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
    )


def verify_cpu():
    """Check one real OpenPI training batch per condition without loading model weights."""
    import jax
    from openpi.training.data_loader import create_data_loader

    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("verify-data requires JAX_PLATFORMS=cpu")

    checks = {}
    prompt = None
    for name, force in fh.VARIANTS.items():
        cfg = make_config(name, batch_size=2, num_workers=0)
        fh.register(cfg)
        loader = create_data_loader(cfg, shuffle=False, num_batches=1)
        observation, actions = next(iter(loader))
        state = np.asarray(observation.state)
        action = np.asarray(actions)
        tokens = np.asarray(observation.tokenized_prompt)
        assert cfg.data.use_force == force
        assert cfg.model.model_type.value == "pi0"
        assert state.shape == (2, 32) and action.shape == (2, 10, 32)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        assert np.all(action[:, :, 13:] == 0)
        assert np.any(state[:, 26:] != 0) == force
        if prompt is None:
            prompt = tokens
        else:
            assert np.array_equal(tokens, prompt)
        checks[name] = dict(
            model=cfg.model.model_type.value,
            current_wrench=force,
            state_shape=list(state.shape),
            actions_shape=list(action.shape),
            language_tokens_shape=list(tokens.shape),
            images={k: list(v.shape) for k, v in observation.images.items()},
        )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    fh.make_config = make_config
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "serve":
        del sys.argv[1]
        import pi05_server

        pi05_server.main()
    elif command == "verify-data":
        verify_cpu()
    else:
        fh.main()
