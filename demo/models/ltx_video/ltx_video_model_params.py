# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from demo.utils.base_params import BaseModelParams


@dataclass(frozen=True)
class LTXVideo2BModelParams(BaseModelParams):
    # Model constants
    VAE_SPATIAL_COMPRESSION_RATIO: int = 32
    VAE_TEMPORAL_COMPRESSION_RATIO: int = 8
    TRANSFORMER_SPATIAL_PATCH_SIZE: int = 1
    TRANSFORMER_TEMPORAL_PATCH_SIZE: int = 1

    # Batch Size
    MIN_BATCH_SIZE: int = 1
    MAX_BATCH_SIZE: int = 4

    # Video-specific parameter dimensions
    MIN_VIDEO_DIM: int = 480
    MAX_VIDEO_DIM: int = 768
    MIN_VIDEO_NUM_FRAMES: int = 25
    MAX_VIDEO_NUM_FRAMES: int = 161

    # Sequence lengths
    MAX_SEQUENCE_LENGTH: int = 128

    # Diffusers or Transformers Pipeline Source
    PIPELINE_SOURCE: str = "Lightricks/LTX-Video"

    # Defaults for LTX Video 2B Pipeline
    NUM_INFERENCE_STEPS: int = 50
    GUIDANCE_SCALE: float = 3.0
    NUM_VIDEOS_PER_PROMPT: int = 1
