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
T5 Text Encoder Model for ONNX → TRT pipeline
"""

from typing import Optional

from transformers import AutoConfig

from demo.utils.base_model import BaseModel
from demo.utils.base_params import BaseModelParams


class T5TextEncoder(BaseModel):
    """
    T5 Text Encoder model.

    This model encodes text prompts into embeddings for conditioning.
    """

    def __init__(
        self,
        name: str = "t5_text_encoder",
        device: str = "cuda",
        verbose: bool = True,
        max_sequence_length: int = 128,
        model_params: BaseModelParams = BaseModelParams,
        hf_token: Optional[str] = None,
    ):
        super().__init__(name, device, verbose, model_params, hf_token)

        self.max_sequence_length = max_sequence_length

        # Model configuration
        self.config = AutoConfig.from_pretrained(
            self.model_params.PIPELINE_SOURCE,
            subfolder="text_encoder",
            token=self.hf_token,
        )

    def get_input_names(self):
        """Return list of input tensor names."""
        return ["input_ids"]

    def get_output_names(self):
        """Return list of output tensor names."""
        return ["text_embeddings"]

    def get_input_profile(self, batch_size, static_shape=False):
        """
        Return TensorRT input profile for dynamic shapes.
        """
        if static_shape:
            return {
                "input_ids": (batch_size, self.max_sequence_length),
            }
        else:
            return {
                "input_ids": [
                    (self.model_params.MIN_BATCH_SIZE, self.max_sequence_length),
                    (batch_size, self.max_sequence_length),
                    (self.model_params.MAX_BATCH_SIZE, self.max_sequence_length),
                ]
            }

    def get_shape_dict(self, batch_size):
        """Return shape dictionary for tensor allocation."""
        input_shapes = self.get_input_profile(batch_size, static_shape=True)

        output_shapes = {
            "text_embeddings": (batch_size, self.max_sequence_length, self.config.d_model),
        }

        return {**input_shapes, **output_shapes}
