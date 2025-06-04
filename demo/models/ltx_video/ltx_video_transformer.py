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
LTX Video Transformer Model for ONNX → TRT pipeline
"""

from diffusers import LTXVideoTransformer3DModel

from demo.models.ltx_video.ltx_video_base import LTXVideoBase
from demo.models.ltx_video.ltx_video_model_params import LTXVideo2BModelParams


class LTXVideoTransformer(LTXVideoBase):
    """
    LTX Video Transformer model for denoising.

    This model takes noisy latents and denoises them using text conditioning.
    Supports both text-to-video and text+image-to-video generation.
    """

    def __init__(
        self,
        name: str = "ltx_video_transformer_2b_txt2vid",
        device: str = "cuda",
        verbose: bool = False,
        do_classifier_free_guidance: bool = False,
        is_image_pipeline: bool = False,
        model_params: LTXVideo2BModelParams = LTXVideo2BModelParams,
    ):
        if is_image_pipeline:
            assert name.endswith(
                "_img2vid"
            ), "Image pipeline requires model name to end with '_img2vid'"
        else:
            assert name.endswith(
                "_txt2vid"
            ), "Text pipeline requires model name to end with '_txt2vid'"

        super().__init__(name, device, verbose, model_params)

        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.is_image_pipeline = is_image_pipeline

        # Model configuration
        self.config = LTXVideoTransformer3DModel.load_config(
            self.model_params.PIPELINE_SOURCE,
            subfolder="transformer",
        )

        # Batch multiplier for classifier-free guidance
        self.xB = 2 if do_classifier_free_guidance else 1

        # Inner dimension for rotary embeddings
        self.inner_dim = self.config["num_attention_heads"] * self.config["attention_head_dim"]
        self.out_channels = self.config.get("out_channels")

    def get_input_names(self):
        """Return list of input tensor names."""
        return [
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "encoder_attention_mask",
            "image_rotary_emb_cos",
            "image_rotary_emb_sin",
        ]

    def get_output_names(self):
        """Return list of output tensor names."""
        return ["latent"]

    def _compute_packed_latent_shape(self, latent_height, latent_width, num_frames):
        """Compute packed latent shape."""
        latent_num_frames = (num_frames - 1) // self.model_params.VAE_TEMPORAL_COMPRESSION_RATIO + 1
        return (
            (latent_num_frames // self.model_params.TRANSFORMER_TEMPORAL_PATCH_SIZE)
            * (latent_height // self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE)
            * (latent_width // self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE)
        )

    def get_input_profile(self, batch_size, height, width, num_frames, static_shape=False):
        """
        Return TensorRT input profile for dynamic shapes.
        """
        latent_height, latent_width = self._compute_latent_dims(height, width)

        # Calculate packed latent shape
        opt_packed_latent_shape = self._compute_packed_latent_shape(
            latent_height, latent_width, num_frames
        )

        if static_shape:
            return {
                "hidden_states": (
                    self.xB * batch_size,
                    opt_packed_latent_shape,
                    self.config["in_channels"],
                ),
                "encoder_hidden_states": (
                    self.xB * batch_size,
                    self.config["in_channels"],
                    self.config["caption_channels"],
                ),
                "encoder_attention_mask": (self.xB * batch_size, self.config["in_channels"]),
                "image_rotary_emb_cos": (
                    self.xB * batch_size,
                    opt_packed_latent_shape,
                    self.config["cross_attention_dim"],
                ),
                "image_rotary_emb_sin": (
                    self.xB * batch_size,
                    opt_packed_latent_shape,
                    self.config["cross_attention_dim"],
                ),
                "timestep": (self.xB * batch_size, opt_packed_latent_shape)
                if self.is_image_pipeline
                else (self.xB * batch_size,),
            }

        min_latent_height, min_latent_width = self._compute_latent_dims(
            self.model_params.MIN_VIDEO_DIM, self.model_params.MIN_VIDEO_DIM
        )
        max_latent_height, max_latent_width = self._compute_latent_dims(
            self.model_params.MAX_VIDEO_DIM, self.model_params.MAX_VIDEO_DIM
        )

        min_packed_latent_shape = self._compute_packed_latent_shape(
            min_latent_height, min_latent_width, self.model_params.MIN_VIDEO_NUM_FRAMES
        )
        max_packed_latent_shape = self._compute_packed_latent_shape(
            max_latent_height, max_latent_width, self.model_params.MAX_VIDEO_NUM_FRAMES
        )

        input_profile = {
            "hidden_states": [
                (
                    self.xB * self.model_params.MIN_BATCH_SIZE,
                    min_packed_latent_shape,
                    self.config["in_channels"],
                ),
                (
                    self.xB * batch_size,
                    opt_packed_latent_shape,
                    self.config["in_channels"],
                ),
                (
                    self.xB * self.model_params.MAX_BATCH_SIZE,
                    max_packed_latent_shape,
                    self.config["in_channels"],
                ),
            ],
            "encoder_hidden_states": [
                (
                    self.xB * self.model_params.MIN_BATCH_SIZE,
                    self.config["in_channels"],
                    self.config["caption_channels"],
                ),
                (
                    self.xB * batch_size,
                    self.config["in_channels"],
                    self.config["caption_channels"],
                ),
                (
                    self.xB * self.model_params.MAX_BATCH_SIZE,
                    self.config["in_channels"],
                    self.config["caption_channels"],
                ),
            ],
            "encoder_attention_mask": [
                (self.xB * self.model_params.MIN_BATCH_SIZE, self.config["in_channels"]),
                (self.xB * batch_size, self.config["in_channels"]),
                (self.xB * self.model_params.MAX_BATCH_SIZE, self.config["in_channels"]),
            ],
            "image_rotary_emb_cos": [
                (
                    self.xB * self.model_params.MIN_BATCH_SIZE,
                    min_packed_latent_shape,
                    self.config["cross_attention_dim"],
                ),
                (
                    self.xB * batch_size,
                    opt_packed_latent_shape,
                    self.config["cross_attention_dim"],
                ),
                (
                    self.xB * self.model_params.MAX_BATCH_SIZE,
                    max_packed_latent_shape,
                    self.config["cross_attention_dim"],
                ),
            ],
            "image_rotary_emb_sin": [
                (
                    self.xB * self.model_params.MIN_BATCH_SIZE,
                    min_packed_latent_shape,
                    self.config["cross_attention_dim"],
                ),
                (
                    self.xB * batch_size,
                    opt_packed_latent_shape,
                    self.config["cross_attention_dim"],
                ),
                (
                    self.xB * self.model_params.MAX_BATCH_SIZE,
                    max_packed_latent_shape,
                    self.config["cross_attention_dim"],
                ),
            ],
            "timestep": [
                (self.xB * self.model_params.MIN_BATCH_SIZE, min_packed_latent_shape)
                if self.is_image_pipeline
                else (self.xB * self.model_params.MIN_BATCH_SIZE,),
                (self.xB * batch_size, opt_packed_latent_shape)
                if self.is_image_pipeline
                else (self.xB * batch_size,),
                (self.xB * self.model_params.MAX_BATCH_SIZE, max_packed_latent_shape)
                if self.is_image_pipeline
                else (self.xB * self.model_params.MAX_BATCH_SIZE,),
            ],
        }

        return input_profile

    def get_shape_dict(self, batch_size, height, width, num_frames):
        """Return shape dictionary for tensor allocation."""
        input_shapes = self.get_input_profile(
            batch_size, height, width, num_frames, static_shape=True
        )

        latent_height, latent_width = self._compute_latent_dims(height, width)
        packed_latent_shape = self._compute_packed_latent_shape(
            latent_height, latent_width, num_frames
        )

        output_shapes = {
            "latent": (self.xB * batch_size, packed_latent_shape, self.config["in_channels"]),
        }

        return {**input_shapes, **output_shapes}
