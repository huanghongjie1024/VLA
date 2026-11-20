# Modified from the LeRobot upstream version in this VLA snapshot.
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

from .act.configuration_act import ACTConfig as ACTConfig
from .act_jepa.configuration_act_jepa import ACTJEPAConfig as ACTJEPAConfig
from .act_jepa_tarimg_same.configuration_act_jepa_tarimg_same import ACTJEPATarimgSameConfig as ACTJEPATarimgSameConfig
from .act_jepa_tarimg_same_v2.configuration_act_jepa_tarimg_same_v2 import ACTJEPATarimgSameV2Config as ACTJEPATarimgSameV2Config
from .act_jepa_tarimg_same_v3.configuration_act_jepa_tarimg_same_v3 import ACTJEPATarimgSameV3Config as ACTJEPATarimgSameV3Config
from .act_ttt.configuration_act_ttt import ACTTTTConfig as ACTTTTConfig
from .le_wm.configuration_lewm import LEWMConfig as LEWMConfig
from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig
# from .groot.configuration_groot import GrootConfig as GrootConfig
from .force_pi0.configuration_force_pi0 import ForcePI0Config as ForcePI0Config
from .pi0.configuration_pi0 import PI0Config as PI0Config
from .pi0_fast.configuration_pi0_fast import PI0FastConfig as PI0FastConfig
from .pi05.configuration_pi05 import PI05Config as PI05Config
from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
from .smolvla.processor_smolvla import SmolVLANewLineProcessor
from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig
from .vqbet.configuration_vqbet import VQBeTConfig as VQBeTConfig
from .wall_x.configuration_wall_x import WallXConfig as WallXConfig
from .xvla.configuration_xvla import XVLAConfig as XVLAConfig

__all__ = [
    "ACTConfig",
    "ACTTTTConfig",
    "DiffusionConfig",
    "ForcePI0Config",
    "LEWMConfig",
    "PI0Config",
    "PI05Config",
    "PI0FastConfig",
    "SmolVLAConfig",
    "SARMConfig",
    "TDMPCConfig",
    "VQBeTConfig",
    "GrootConfig",
    "XVLAConfig",
    "WallXConfig",
]
