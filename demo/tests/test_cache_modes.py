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
Test cache modes (lean vs full) using pytest.
"""

from pathlib import Path

import pytest

from demo.utils.path_manager import PathManager


@pytest.mark.cache
@pytest.mark.unit
class TestCacheModes:
    """Test lean vs full cache modes."""

    def test_full_cache_mode(self, temp_cache_dir: Path):
        """Test full cache mode keeps all models."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), verbose=True, cache_mode="full")

        # Simulate pipeline 1 models
        pipeline1_models = {
            "text_encoder": ("t5_text_encoder", "fp16"),
            "transformer": ("ltx_video_transformer", "fp8"),
            "vae": ("ltx_video_vae", "fp16"),
        }

        path_manager.set_pipeline_models("ltx_video_2b", pipeline1_models)

        # Create dummy files
        for role, (model_id, precision) in pipeline1_models.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            engine_path = path_manager.get_engine_path(model_id, precision)
            metadata_path = path_manager.get_metadata_path(model_id, precision)

            onnx_path.touch()
            engine_path.touch()
            metadata_path.touch()

        # Switch to different models for the same pipeline
        pipeline2_models = {
            "text_encoder": ("t5_text_encoder", "fp16"),  # Same
            "transformer": ("flux_transformer", "fp16"),  # Different
            "vae": ("sdxl_vae", "fp16"),  # Different
        }

        # Create dummy files for new models
        for role, (model_id, precision) in pipeline2_models.items():
            if model_id != "t5_text_encoder":  # Don't recreate shared model
                onnx_path = path_manager.get_onnx_path(model_id, precision)
                engine_path = path_manager.get_engine_path(model_id, precision)
                metadata_path = path_manager.get_metadata_path(model_id, precision)

                onnx_path.touch()
                engine_path.touch()
                metadata_path.touch()

        # Update pipeline
        path_manager.set_pipeline_models("ltx_video_2b", pipeline2_models)

        # In full mode, old models should still exist
        old_transformer_path = path_manager.get_onnx_path("ltx_video_transformer", "fp8")
        old_vae_path = path_manager.get_onnx_path("ltx_video_vae", "fp16")

        assert old_transformer_path.exists(), "Full mode should keep old models"
        assert old_vae_path.exists(), "Full mode should keep old models"

    def test_lean_cache_mode(self, temp_cache_dir: Path):
        """Test lean cache mode cleans up unused models."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), verbose=True, cache_mode="lean")

        # Same test as full mode but with lean mode
        pipeline1_models = {
            "text_encoder": ("t5_text_encoder", "fp16"),
            "transformer": ("ltx_video_transformer", "fp8"),
            "vae": ("ltx_video_vae", "fp16"),
        }

        path_manager.set_pipeline_models("ltx_video_2b", pipeline1_models)

        # Create dummy files
        for role, (model_id, precision) in pipeline1_models.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            engine_path = path_manager.get_engine_path(model_id, precision)
            metadata_path = path_manager.get_metadata_path(model_id, precision)

            onnx_path.touch()
            engine_path.touch()
            metadata_path.touch()

        # Switch to different models
        pipeline2_models = {
            "text_encoder": ("t5_text_encoder", "fp16"),  # Shared
            "transformer": ("flux_transformer", "fp16"),  # Different
            "vae": ("sdxl_vae", "fp16"),  # Different
        }

        # Create dummy files for new models
        for role, (model_id, precision) in pipeline2_models.items():
            if model_id != "t5_text_encoder":
                onnx_path = path_manager.get_onnx_path(model_id, precision)
                engine_path = path_manager.get_engine_path(model_id, precision)
                metadata_path = path_manager.get_metadata_path(model_id, precision)

                onnx_path.touch()
                engine_path.touch()
                metadata_path.touch()

        # Update pipeline - should trigger cleanup
        path_manager.set_pipeline_models("ltx_video_2b", pipeline2_models)

        # In lean mode, old models should be deleted
        old_transformer_path = path_manager.get_onnx_path("ltx_video_transformer", "fp8")
        old_vae_path = path_manager.get_onnx_path("ltx_video_vae", "fp16")
        shared_text_encoder_path = path_manager.get_onnx_path("t5_text_encoder", "fp16")

        assert not old_transformer_path.exists(), "Lean mode should delete unused models"
        assert not old_vae_path.exists(), "Lean mode should delete unused models"
        assert shared_text_encoder_path.exists(), "Lean mode should keep shared models"

    def test_cache_mode_validation(self, temp_cache_dir: Path):
        """Test cache mode validation."""
        with pytest.raises(ValueError, match="cache_mode must be 'lean' or 'full'"):
            PathManager(cache_dir=str(temp_cache_dir), cache_mode="invalid")

    def test_multiple_pipelines_lean_mode(self, temp_cache_dir: Path):
        """Test multiple pipelines in lean mode."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), verbose=True, cache_mode="lean")

        # Create two different pipelines with shared models
        pipeline_a_models = {
            "text_encoder": ("t5_text_encoder", "fp16"),  # Shared
            "transformer": ("model_a_transformer", "fp8"),
            "vae": ("model_a_vae", "fp16"),
        }

        pipeline_b_models = {
            "text_encoder": ("t5_text_encoder", "fp16"),  # Shared
            "transformer": ("model_b_transformer", "fp8"),
            "vae": ("model_b_vae", "fp16"),
        }

        # Set up pipeline A
        path_manager.set_pipeline_models("pipeline_a", pipeline_a_models)
        for role, (model_id, precision) in pipeline_a_models.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            onnx_path.touch()

        # Set up pipeline B
        path_manager.set_pipeline_models("pipeline_b", pipeline_b_models)
        for role, (model_id, precision) in pipeline_b_models.items():
            if model_id != "t5_text_encoder":  # Don't recreate shared
                onnx_path = path_manager.get_onnx_path(model_id, precision)
                onnx_path.touch()

        # Verify coexistence
        shared_text_encoder = path_manager.get_onnx_path("t5_text_encoder", "fp16")
        pipeline_a_transformer = path_manager.get_onnx_path("model_a_transformer", "fp8")
        pipeline_b_transformer = path_manager.get_onnx_path("model_b_transformer", "fp8")

        assert shared_text_encoder.exists(), "Shared model should exist"
        assert pipeline_a_transformer.exists(), "Pipeline A models should exist"
        assert pipeline_b_transformer.exists(), "Pipeline B models should exist"
