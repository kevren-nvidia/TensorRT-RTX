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

"""
LTX Video VAE Model for ONNX → TRT pipeline
"""

from typing import Dict, Optional, Tuple

import torch
from diffusers import AutoencoderKLLTXVideo

from demo.models.ltx_video.ltx_video_base import LTXVideoBase
from demo.models.ltx_video.ltx_video_model_params import LTXVideo2BModelParams


class LTXVideoVAEDecoder(LTXVideoBase):
    """
    LTX Video VAE Decoder model.

    This model decodes latents to videos.
    """

    def __init__(
        self,
        name: str = "ltx_video_vae_decoder",
        device: str = "cuda",
        verbose: bool = True,
        model_params: LTXVideo2BModelParams = LTXVideo2BModelParams,
    ):
        super().__init__(name, device, verbose, model_params)

        # Model configuration
        full_model = AutoencoderKLLTXVideo.from_pretrained(
            self.model_params.PIPELINE_SOURCE,
            subfolder="vae",
        )
        self.config = full_model.config

        # Latent statistics
        self.latents_mean = full_model.latents_mean.clone().to(self.device)
        self.latents_std = full_model.latents_std.clone().to(self.device)

    def get_input_names(self):
        """Return list of input tensor names."""
        return ["latent"]

    def get_output_names(self):
        """Return list of output tensor names."""
        return ["video"]

    def get_input_profile(self, batch_size, height, width, num_frames, static_shape=False):
        """
        Return TensorRT input profile for dynamic shapes.
        """
        latent_height, latent_width, latent_num_frames = self._compute_latent_dims(
            height, width, num_frames
        )

        if static_shape:
            return {
                "latent": (
                    batch_size,
                    self.config["latent_channels"],
                    latent_num_frames,
                    latent_height,
                    latent_width,
                ),
            }

        min_latent_height, min_latent_width, min_latent_num_frames = self._compute_latent_dims(
            self.model_params.MIN_VIDEO_DIM,
            self.model_params.MIN_VIDEO_DIM,
            self.model_params.MIN_VIDEO_NUM_FRAMES,
        )
        max_latent_height, max_latent_width, max_latent_num_frames = self._compute_latent_dims(
            self.model_params.MAX_VIDEO_DIM,
            self.model_params.MAX_VIDEO_DIM,
            self.model_params.MAX_VIDEO_NUM_FRAMES,
        )

        return {
            "latent": [
                (
                    self.model_params.MIN_BATCH_SIZE,
                    self.config["latent_channels"],
                    min_latent_num_frames,
                    min_latent_height,
                    min_latent_width,
                ),
                (
                    batch_size,
                    self.config["latent_channels"],
                    latent_num_frames,
                    latent_height,
                    latent_width,
                ),
                (
                    self.model_params.MAX_BATCH_SIZE,
                    self.config["latent_channels"],
                    max_latent_num_frames,
                    max_latent_height,
                    max_latent_width,
                ),
            ]
        }

    def get_shape_dict(self, batch_size, height, width, num_frames):
        """Return shape dictionary for tensor allocation."""
        input_shapes = self.get_input_profile(
            batch_size, height, width, num_frames, static_shape=True
        )

        output_shapes = {
            "video": (batch_size, self.config["out_channels"], num_frames, height, width),
        }

        return {**input_shapes, **output_shapes}


class LTXVideoVAEEncoder(LTXVideoBase):
    """
    LTX Video VAE Encoder model.

    This model encodes images.
    """

    def __init__(
        self,
        name: str = "ltx_video_vae_decoder",
        device: str = "cuda",
        verbose: bool = True,
        model_params: LTXVideo2BModelParams = LTXVideo2BModelParams,
    ):
        super().__init__(name, device, verbose, model_params)

        # Model configuration
        full_model = AutoencoderKLLTXVideo.from_pretrained(
            self.model_params.PIPELINE_SOURCE,
            subfolder="vae",
        )
        self.config = full_model.config

        # Latent statistics
        self.latents_mean = full_model.latents_mean.clone().to(self.device)
        self.latents_std = full_model.latents_std.clone().to(self.device)

    def get_input_names(self):
        """Return list of input tensor names."""
        return ["image"]

    def get_output_names(self):
        """Return list of output tensor names."""
        return ["latent_image"]

    def get_input_profile(self, batch_size, height, width, num_frames, static_shape=False):
        """
        Return TensorRT input profile for dynamic shapes.
        """
        if static_shape:
            return {
                "image": [
                    (1, self.config["in_channels"], 1, height, width),
                ]
            }

        min_height, min_width = self.model_params.MIN_VIDEO_DIM, self.model_params.MIN_VIDEO_DIM
        max_height, max_width = self.model_params.MAX_VIDEO_DIM, self.model_params.MAX_VIDEO_DIM

        return {
            "image": [
                (1, self.config["in_channels"], 1, min_height, min_width),
                (1, self.config["in_channels"], 1, height, width),
                (1, self.config["in_channels"], 1, max_height, max_width),
            ]
        }

    def get_shape_dict(self, batch_size, height, width, num_frames):
        """Return shape dictionary for tensor allocation."""
        input_shapes = self.get_input_profile(
            batch_size, height, width, num_frames, static_shape=True
        )
        latent_height, latent_width = self._compute_latent_dims(height, width)

        output_shapes = {
            "latent_image": (
                1,
                self.config["block_out_channels"][1],
                1,
                latent_height,
                latent_width,
            ),
        }

        return {**input_shapes, **output_shapes}
