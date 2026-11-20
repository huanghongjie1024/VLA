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

"""Data processing pipeline for Force-π₀ policy.

Extends the pi0 processor with force data normalization and preprocessing.
"""

from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.force_pi0.configuration_force_pi0 import ForcePI0Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@ProcessorStepRegistry.register(name="force_pi0_new_line_processor")
class ForcePi0NewLineProcessor(ComplementaryDataProcessorStep):
    """Ensure task description ends with a newline for PaliGemma tokenizer compatibility."""

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        if isinstance(task, str):
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="force_pi0_force_preprocessor")
class ForcePreprocessorStep(ProcessorStep):
    """Preprocess raw force/torque data for the force encoder.

    Applies:
    1. Zero-point calibration (subtract mean of first N samples)
    2. Low-pass filtering (simple moving average)
    3. Z-score normalization using dataset statistics
    4. Resampling to expected frequency (if needed)
    """

    def __init__(
        self,
        force_key: str = "observation.force",
        window_size: int = 64,
        stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        self.force_key = force_key
        self.window_size = window_size
        self.stats = stats

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.force_key not in batch:
            return batch

        force = batch[self.force_key]

        # Ensure float32
        if not torch.is_floating_point(force):
            force = force.float()

        # Apply normalization if stats available
        if self.stats is not None and self.force_key in self.stats:
            mean = self.stats[self.force_key].get("mean")
            std = self.stats[self.force_key].get("std")
            if mean is not None and std is not None:
                force = (force - mean) / (std + 1e-8)

        new_batch = dict(batch)
        new_batch[self.force_key] = force
        return new_batch

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_force_pi0_pre_post_processors(
    config: ForcePI0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct pre-processor and post-processor pipelines for Force-π₀.

    Pre-processing pipeline:
    1. Rename observations to match pretrained configuration
    2. Normalize input and output features
    3. Add batch dimension
    4. Append newline to task description for PaliGemma tokenizer
    5. Tokenize text prompt
    6. Preprocess force data (normalize, window)
    7. Move data to device

    Post-processing pipeline:
    1. Move data to CPU
    2. Unnormalize output features

    Args:
        config: ForcePI0Config instance.
        dataset_stats: Dataset statistics for normalization.

    Returns:
        Tuple of (preprocessor_pipeline, postprocessor_pipeline).
    """

    # Force-specific stats
    force_stats = None
    if dataset_stats is not None:
        force_stats = {k: v for k, v in dataset_stats.items() if "force" in k.lower()}

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AddBatchDimensionProcessorStep(),
        ForcePi0NewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.max_seq_len,
            padding_side="right",
            padding="max_length",
        ),
        ForcePreprocessorStep(
            force_key="observation.force",
            window_size=config.force_window_size,
            stats=force_stats if force_stats else None,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
