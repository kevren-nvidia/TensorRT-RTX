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
Test script to verify the new hierarchical folder structure.
"""

from pathlib import Path

import pytest

from demo.utils.path_manager import PathManager


@pytest.mark.paths
@pytest.mark.unit
class TestHierarchicalStructure:
    """Test hierarchical folder structure functionality."""

    def test_shared_paths_structure(self, path_manager: PathManager):
        """Test that shared paths follow the correct hierarchical structure."""
        shared_onnx = path_manager.get_onnx_path("t5_text_encoder", "fp16")
        shared_engine = path_manager.get_engine_path("t5_text_encoder", "fp16")
        shared_metadata = path_manager.get_metadata_path("t5_text_encoder", "fp16")

        # Verify structure by checking path components
        assert "shared/onnx/t5_text_encoder/fp16/t5_text_encoder.onnx" in str(shared_onnx)
        assert "shared/engines/t5_text_encoder/fp16/t5_text_encoder.engine" in str(shared_engine)
        assert "shared/engines/t5_text_encoder/fp16/t5_text_encoder.metadata.json" in str(
            shared_metadata
        )

    def test_different_precisions_separated(self, path_manager: PathManager):
        """Test that different precisions for same model are properly separated."""
        fp16_onnx = path_manager.get_onnx_path("ltx_video_transformer", "fp16")
        fp8_onnx = path_manager.get_onnx_path("ltx_video_transformer", "fp8")

        # Verify they're in different directories
        assert (
            fp16_onnx.parent != fp8_onnx.parent
        ), "Different precisions should be in different directories"
        assert "fp16" in str(fp16_onnx), "FP16 path should contain 'fp16'"
        assert "fp8" in str(fp8_onnx), "FP8 path should contain 'fp8'"

    def test_directory_creation(self, path_manager: PathManager):
        """Test that directories are created correctly."""
        shared_onnx = path_manager.get_onnx_path("t5_text_encoder", "fp16")
        shared_engine = path_manager.get_engine_path("t5_text_encoder", "fp16")

        assert shared_onnx.parent.exists(), "Shared ONNX directory should be created"
        assert shared_engine.parent.exists(), "Shared engine directory should be created"

    def test_file_naming_consistency(self, path_manager: PathManager):
        """Test that file naming is consistent and clean."""
        shared_onnx = path_manager.get_onnx_path("t5_text_encoder", "fp16")
        shared_engine = path_manager.get_engine_path("t5_text_encoder", "fp16")

        # All files should have clean names without precision suffixes
        assert (
            shared_onnx.name == "t5_text_encoder.onnx"
        ), f"ONNX should have clean name, got: {shared_onnx.name}"
        assert (
            shared_engine.name == "t5_text_encoder.engine"
        ), f"Engine should have clean name, got: {shared_engine.name}"
