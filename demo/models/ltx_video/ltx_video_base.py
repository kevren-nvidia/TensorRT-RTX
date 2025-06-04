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


from demo.models.ltx_video.ltx_video_model_params import LTXVideo2BModelParams
from demo.utils.base_model import BaseModel


class LTXVideoBase(BaseModel):
    """
    LTX Video Base model.

    This model is the base class for all unique LTX Video models.
    It contains common methods for all LTX Video models.
    """

    def __init__(
        self,
        name: str,
        device: str,
        verbose: bool,
        model_params: LTXVideo2BModelParams,
    ):
        super().__init__(name, device, verbose, model_params)

    def _compute_latent_dims(self, height, width, num_frames=None):
        """Compute latent dimensions."""
        latent_height = height // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        latent_width = width // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO

        if num_frames is not None:
            latent_num_frames = (
                num_frames - 1
            ) // self.model_params.VAE_TEMPORAL_COMPRESSION_RATIO + 1
            return latent_height, latent_width, latent_num_frames
        else:
            return latent_height, latent_width

    def _compute_packed_latent_shape(self, latent_height, latent_width, num_frames):
        """Compute packed latent shape."""
        latent_num_frames = (num_frames - 1) // self.model_params.VAE_TEMPORAL_COMPRESSION_RATIO + 1
        return (
            (latent_num_frames // self.model_params.TRANSFORMER_TEMPORAL_PATCH_SIZE)
            * (latent_height // self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE)
            * (latent_width // self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE)
        )
