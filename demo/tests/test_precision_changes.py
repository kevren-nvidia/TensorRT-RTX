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
Test script to verify precision changes are handled correctly across runs.
"""

import json
from pathlib import Path

import pytest

from demo.utils.path_manager import PathManager


@pytest.mark.integration
class TestPrecisionChanges:
    """Test precision changes across multiple runs."""

    def test_precision_changes_lean_mode(self, temp_cache_dir: Path):
        """Test precision changes across multiple runs in lean mode."""
        # Test 1: Initial run with specific precisions
        path_manager = PathManager(cache_dir=str(temp_cache_dir), verbose=True, cache_mode="lean")

        initial_config = {
            "text_encoder": ("t5_text_encoder", "fp16"),
            "transformer": ("ltx_video_transformer", "fp16"),  # fp16 initially
            "vae": ("ltx_video_vae", "fp16"),
        }

        path_manager.set_pipeline_models("ltx_video_2b", initial_config)

        # Create dummy files for initial models
        for role, (model_id, precision) in initial_config.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            engine_path = path_manager.get_engine_path(model_id, precision)
            metadata_path = path_manager.get_metadata_path(model_id, precision)

            onnx_path.touch()
            engine_path.touch()
            metadata_path.touch()

        # Verify state file was created
        state_file = temp_cache_dir / ".cache_state.json"
        assert state_file.exists(), "State file should be created"

        with open(state_file, "r") as f:
            state = json.load(f)

        expected_state = {
            "ltx_video_2b": {
                "text_encoder": ["t5_text_encoder", "fp16"],
                "transformer": ["ltx_video_transformer", "fp16"],
                "vae": ["ltx_video_vae", "fp16"],
            }
        }
        assert state == expected_state, f"State mismatch: {state} != {expected_state}"

    def test_precision_change_cleanup(self, temp_cache_dir: Path):
        """Test that changing precision cleans up old files in lean mode."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), verbose=True, cache_mode="lean")

        # Initial configuration
        initial_config = {
            "text_encoder": ("t5_text_encoder", "fp16"),
            "transformer": ("ltx_video_transformer", "fp16"),
            "vae": ("ltx_video_vae", "fp16"),
        }

        path_manager.set_pipeline_models("ltx_video_2b", initial_config)

        # Create initial files
        for role, (model_id, precision) in initial_config.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            onnx_path.touch()

        # Change transformer precision
        changed_config = {
            "text_encoder": ("t5_text_encoder", "fp16"),  # Same
            "transformer": ("ltx_video_transformer", "fp8"),  # Changed fp16 -> fp8
            "vae": ("ltx_video_vae", "fp16"),  # Same
        }

        # Check that old fp16 transformer exists before change
        old_transformer_onnx = path_manager.get_onnx_path("ltx_video_transformer", "fp16")
        assert old_transformer_onnx.exists(), "Old fp16 transformer should exist before change"

        path_manager.set_pipeline_models("ltx_video_2b", changed_config)

        # Create new fp8 transformer file
        new_transformer_onnx = path_manager.get_onnx_path("ltx_video_transformer", "fp8")
        new_transformer_onnx.touch()

        # Verify cleanup happened
        assert not old_transformer_onnx.exists(), "Old fp16 transformer should be deleted"
        assert new_transformer_onnx.exists(), "New fp8 transformer should exist"

        # Verify shared models are preserved
        shared_text_encoder = path_manager.get_onnx_path("t5_text_encoder", "fp16")
        shared_vae = path_manager.get_onnx_path("ltx_video_vae", "fp16")
        assert shared_text_encoder.exists(), "Shared text encoder should be preserved"
        assert shared_vae.exists(), "Shared VAE should be preserved"

    def test_multiple_pipelines_preserve_shared_models(self, temp_cache_dir: Path):
        """Test that models needed by other pipelines are preserved."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), verbose=True, cache_mode="lean")

        # First pipeline configuration
        pipeline1_config = {
            "text_encoder": ("t5_text_encoder", "fp16"),
            "transformer": ("ltx_video_transformer", "fp16"),
            "vae": ("ltx_video_vae", "fp16"),
        }

        path_manager.set_pipeline_models("ltx_video_2b", pipeline1_config)

        # Create files
        for role, (model_id, precision) in pipeline1_config.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            onnx_path.touch()

        # Add another pipeline that uses the same transformer precision
        pipeline2_config = {
            "text_encoder": ("t5_text_encoder", "fp16"),
            "transformer": ("ltx_video_transformer", "fp16"),  # Same as pipeline1
            "vae": ("sdxl_vae", "fp16"),  # Different VAE
        }

        path_manager.set_pipeline_models("another_pipeline", pipeline2_config)

        # Create new VAE file
        sdxl_vae_path = path_manager.get_onnx_path("sdxl_vae", "fp16")
        sdxl_vae_path.touch()

        # Now change first pipeline to use different transformer
        changed_pipeline1_config = {
            "text_encoder": ("t5_text_encoder", "fp16"),  # Same
            "transformer": ("flux_transformer", "fp16"),  # Different model entirely
            "vae": ("ltx_video_vae", "fp16"),  # Same
        }

        # Create new flux transformer
        flux_onnx = path_manager.get_onnx_path("flux_transformer", "fp16")
        flux_onnx.touch()

        path_manager.set_pipeline_models("ltx_video_2b", changed_pipeline1_config)

        # Verify ltx_video_transformer fp16 is preserved (needed by another_pipeline)
        preserved_transformer = path_manager.get_onnx_path("ltx_video_transformer", "fp16")
        assert (
            preserved_transformer.exists()
        ), "Transformer should be preserved (needed by other pipeline)"

        # Verify new flux transformer exists
        assert flux_onnx.exists(), "New flux transformer should exist"

    def test_full_mode_keeps_all_models(self, temp_cache_dir: Path):
        """Test that full mode never deletes models regardless of precision changes."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), verbose=True, cache_mode="full")

        # Initial configuration
        initial_config = {
            "transformer": ("test_transformer", "fp16"),
        }

        path_manager.set_pipeline_models("test_pipeline", initial_config)

        # Create initial file
        fp16_path = path_manager.get_onnx_path("test_transformer", "fp16")
        fp16_path.touch()

        # Change precision
        changed_config = {
            "transformer": ("test_transformer", "fp8"),
        }

        path_manager.set_pipeline_models("test_pipeline", changed_config)

        # Create new precision file
        fp8_path = path_manager.get_onnx_path("test_transformer", "fp8")
        fp8_path.touch()

        # In full mode, both should exist
        assert fp16_path.exists(), "Full mode should keep fp16 model"
        assert fp8_path.exists(), "Full mode should have fp8 model"
