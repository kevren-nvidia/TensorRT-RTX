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
Path Manager for cache directory structure and file operations.

Handles:
- Shared cache directory structure for ONNX and engines
- Direct file path access without symlinks
- File operations and cleanup
- Smart caching and usage tracking
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from huggingface_hub import snapshot_download

from demo.utils.model_registry import registry as model_registry


class PathManager:
    """
    Path Manager for shared cache directories.

    Directory structure:
    cache_dir/
    ├── shared/                           # Canonical storage for all models
    │   ├── onnx/
    │   │   ├── model_id/
    │   │   │   └── precision/
    │   │   │       ├── model_id.onnx
    │   │   │       └── model_id.onnx.data (external data)
    │   │   │
    │   └── engines/
    │       ├── model_id/
    │       │   └── precision/
    │       │       ├── model_id.engine
    │       │       └── model_id.metadata.json
    └── .cache_state.json                 # Pipeline usage tracking for cleanup
    """

    def __init__(
        self, cache_dir: str = "./demo_cache", verbose: bool = True, cache_mode: str = "full"
    ):
        """
        Initialize PathManager.

        Args:
            cache_dir: Base cache directory
            verbose: Enable verbose logging
            cache_mode: "lean" (delete unused models) or "full" (keep all models)
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.verbose = verbose
        self.cache_mode = cache_mode

        if cache_mode not in ["lean", "full"]:
            raise ValueError(f"cache_mode must be 'lean' or 'full', got: {cache_mode}")

        # Create base directories
        self.shared_dir = self.cache_dir / "shared"
        self.shared_onnx_dir = self.shared_dir / "onnx"
        self.shared_engines_dir = self.shared_dir / "engines"

        for dir_path in [self.shared_onnx_dir, self.shared_engines_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # State file for persistent tracking
        self.state_file = self.cache_dir / ".cache_state.json"
        self.pipeline_states = self._load_pipeline_states()

        if self.verbose:
            print(f"[INFO] Cache directory: {self.cache_dir}")
            print(f"[INFO] Cache mode: {self.cache_mode}")

    def _load_pipeline_states(self) -> Dict[str, Dict[str, Tuple[str, str]]]:
        """Load pipeline states from persistent storage."""
        if not self.state_file.exists():
            return {}

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            # Convert back to proper format
            result = {}
            for pipeline_name, roles in data.items():
                result[pipeline_name] = {}
                for role, model_info in roles.items():
                    result[pipeline_name][role] = tuple(model_info)

            return result
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to load pipeline states: {e}")
            return {}

    def _save_pipeline_states(self) -> None:
        """Save pipeline states to persistent storage."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self.pipeline_states, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to save pipeline states: {e}")

    def get_onnx_path(self, model_id: str, precision: str) -> Path:
        """Get ONNX file path in shared directory."""
        model_precision_dir = self.shared_onnx_dir / model_id / precision
        model_precision_dir.mkdir(parents=True, exist_ok=True)
        return model_precision_dir / f"{model_id}.onnx"

    def get_engine_path(self, model_id: str, precision: str) -> Path:
        """Get engine file path in shared directory."""
        model_precision_dir = self.shared_engines_dir / model_id / precision
        model_precision_dir.mkdir(parents=True, exist_ok=True)
        return model_precision_dir / f"{model_id}.engine"

    def get_metadata_path(self, model_id: str, precision: str) -> Path:
        """Get metadata file path in shared directory."""
        model_precision_dir = self.shared_engines_dir / model_id / precision
        model_precision_dir.mkdir(parents=True, exist_ok=True)
        return model_precision_dir / f"{model_id}.metadata.json"

    def get_model_directory(self, model_id: str, precision: str, file_type: str) -> Path:
        """Get the directory containing model files."""
        if file_type == "onnx":
            return self.shared_onnx_dir / model_id / precision
        elif file_type == "engine":
            return self.shared_engines_dir / model_id / precision
        else:
            raise ValueError(f"Invalid file_type: {file_type}. Must be 'onnx' or 'engine'")

    def check_cached_files(self, model_id: str, precision: str) -> dict:
        """Check which files exist in cache."""
        onnx_path = self.get_onnx_path(model_id, precision)
        engine_path = self.get_engine_path(model_id, precision)
        metadata_path = self.get_metadata_path(model_id, precision)

        return {
            "onnx": onnx_path.exists(),
            "engine": engine_path.exists(),
            "metadata": metadata_path.exists(),
        }

    def delete_cached_onnx_files(self, model_id: str, precision: str) -> None:
        """Delete cached ONNX files for a model."""
        if self.verbose:
            print(f"[WARN] Deleting cached ONNX files for {model_id}_{precision}")

        onnx_path = self.get_onnx_path(model_id, precision)
        model_dir = self.get_model_directory(model_id, precision, "onnx")

        # Delete the main ONNX file
        if onnx_path.exists():
            onnx_path.unlink()

        # Delete all related files in the directory
        if model_dir.exists():
            for related_file in model_dir.iterdir():
                if related_file.is_file():
                    related_file.unlink()

        self._cleanup_empty_dirs(model_id, precision, "onnx")

    def delete_cached_engine_files(self, model_id: str, precision: str) -> None:
        """Delete cached engine files for a model."""
        if self.verbose:
            print(f"[WARN] Deleting cached engine files for {model_id}_{precision}")

        engine_path = self.get_engine_path(model_id, precision)
        metadata_path = self.get_metadata_path(model_id, precision)

        for path in [engine_path, metadata_path]:
            if path.exists():
                path.unlink()

        self._cleanup_empty_dirs(model_id, precision, "engine")

    def delete_cached_files(self, model_id: str, precision: str) -> None:
        """Delete all cached files (both ONNX and engine files) for a model."""
        if self.verbose:
            print(f"[WARN] Deleting all cached files for {model_id}_{precision}")

        self.delete_cached_onnx_files(model_id, precision)
        self.delete_cached_engine_files(model_id, precision)

    def list_cached_models(self) -> dict:
        """List all cached models."""
        cached = {"onnx": [], "engines": []}

        # Scan shared/onnx/model_id/precision/*.onnx
        if self.shared_onnx_dir.exists():
            for model_dir in self.shared_onnx_dir.iterdir():
                if model_dir.is_dir():
                    for precision_dir in model_dir.iterdir():
                        if precision_dir.is_dir() and any(precision_dir.glob("*.onnx")):
                            cached["onnx"].append(f"{model_dir.name}_{precision_dir.name}")

        # Scan shared/engines/model_id/precision/*.engine
        if self.shared_engines_dir.exists():
            for model_dir in self.shared_engines_dir.iterdir():
                if model_dir.is_dir():
                    for precision_dir in model_dir.iterdir():
                        if precision_dir.is_dir() and any(precision_dir.glob("*.engine")):
                            cached["engines"].append(f"{model_dir.name}_{precision_dir.name}")

        return cached

    def print_cache_summary(self) -> None:
        """Print cache summary."""
        print(f"\n[INFO] Cache Summary ({self.cache_dir}):")

        cached = self.list_cached_models()

        print(f"[INFO] ONNX models: {len(cached.get('onnx', []))}")
        for model in cached.get("onnx", []):
            print(f"[INFO]   {model}")

        print(f"[INFO] Engine models: {len(cached.get('engines', []))}")
        for model in cached.get("engines", []):
            print(f"[INFO]   {model}")

    def acquire_onnx_file(
        self,
        pipeline_source: str,
        model_id: str,
        precision: str,
        onnx_source: str,
        verbose: bool = False,
        hf_token: Optional[str] = None,
    ) -> bool:
        """
        Acquire ONNX file from source (URL or local path) and store in shared cache.

        For local paths: Copies ALL files from the source directory.
        For URLs: Downloads the specified files and moves them to the target location.
        """
        remote_onnx_subfolder = model_registry.get_onnx_url(pipeline_source, model_id, precision)
        onnx_path = self.get_onnx_path(model_id, precision)

        if onnx_path.exists():
            if verbose:
                print(f"[INFO] ONNX already exists: {onnx_path}")
            return True

        try:
            # Check if source is a local file path
            if Path(onnx_source).exists():
                source_path = Path(onnx_source)
                source_dir = source_path.parent

                if verbose:
                    print(f"[INFO] Copying ONNX from: {source_dir}")

                # Copy the main ONNX file
                shutil.copy2(onnx_source, onnx_path)

                # Copy ALL files from the source directory
                for source_file in source_dir.iterdir():
                    if source_file.is_file() and source_file != source_path:
                        target_file = onnx_path.parent / source_file.name
                        shutil.copy2(source_file, target_file)

                return True

            else:
                # Download from URL
                if verbose:
                    print(
                        f"[INFO] Downloading ONNX from: {pipeline_source}/{remote_onnx_subfolder}"
                    )

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_download_path = Path(temp_dir)

                    snapshot_download(
                        repo_id=pipeline_source,
                        allow_patterns=os.path.join(remote_onnx_subfolder, "*"),
                        local_dir=temp_download_path,
                        token=hf_token,
                    )

                    # Find the downloaded files in the nested structure and move them
                    source_files_dir = temp_download_path / remote_onnx_subfolder
                    if source_files_dir.exists():
                        for source_file in source_files_dir.iterdir():
                            if source_file.is_file():
                                target_file = onnx_path.parent / source_file.name
                                shutil.move(str(source_file), str(target_file))

                return onnx_path.exists()

        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to acquire ONNX file: {e}")
            return False

    def set_pipeline_models(
        self, pipeline_name: str, model_configs: Dict[str, Tuple[str, str]]
    ) -> None:
        """
        Set the models needed for current pipeline and handle cleanup in lean mode.

        Args:
            pipeline_name: Name of the pipeline
            model_configs: Dict of {role: (model_id, precision)} for this pipeline
        """
        old_config = self.pipeline_states.get(pipeline_name, {})

        if self.cache_mode == "lean" and old_config:
            # Find models that changed or were removed
            models_to_cleanup = set()

            for old_role, (old_model_id, old_precision) in old_config.items():
                if old_role not in model_configs:
                    models_to_cleanup.add((old_model_id, old_precision))
                else:
                    new_model_id, new_precision = model_configs[old_role]
                    if (old_model_id, old_precision) != (new_model_id, new_precision):
                        models_to_cleanup.add((old_model_id, old_precision))

            # Update pipeline state BEFORE checking what's needed by other pipelines
            self.pipeline_states[pipeline_name] = model_configs.copy()

            # Check if any models to cleanup are still needed by other pipelines
            all_needed_models = set()
            for other_config in self.pipeline_states.values():
                for model_id, precision in other_config.values():
                    all_needed_models.add((model_id, precision))

            # Only cleanup models that are truly unused
            truly_unused = models_to_cleanup - all_needed_models

            if truly_unused:
                if self.verbose:
                    print(f"[INFO] Cleaning up {len(truly_unused)} unused models in lean mode")

                for model_id, precision in truly_unused:
                    self._cleanup_unused_model(model_id, precision)

            self._save_pipeline_states()
        else:
            # Full mode or no previous config - just update state
            self.pipeline_states[pipeline_name] = model_configs.copy()
            self._save_pipeline_states()

    def clean_cache_for_precision_change(
        self, pipeline_name: str, role: str, old_precision: str, new_precision: str, model_id: str
    ) -> None:
        """Clean up cache when precision changes for a specific role."""
        if self.verbose:
            print(
                f"[INFO] Cleaning cache for precision change: {pipeline_name}:{role} {old_precision} -> {new_precision}"
            )

        if self.cache_mode == "lean":
            old_model = (model_id, old_precision)

            # Get all currently needed models (excluding the changed one)
            all_needed_models = set()
            for other_pipeline, other_config in self.pipeline_states.items():
                for other_role, (other_model_id, other_precision) in other_config.items():
                    if not (other_pipeline == pipeline_name and other_role == role):
                        all_needed_models.add((other_model_id, other_precision))

            # Add the new precision
            all_needed_models.add((model_id, new_precision))

            if old_model not in all_needed_models:
                if self.verbose:
                    print(f"[INFO] Old precision {old_precision} no longer needed, cleaning up")
                self._cleanup_unused_model(model_id, old_precision)

    def _cleanup_unused_model(self, model_id: str, precision: str) -> None:
        """Clean up an unused model in lean mode."""
        if self.verbose:
            print(f"[INFO] Cleaning up unused model: {model_id}_{precision}")

        files_exist = self.check_cached_files(model_id, precision)

        if any(files_exist.values()):
            self.delete_cached_files(model_id, precision)

    def _cleanup_empty_dirs(self, model_id: str, precision: str, file_type: str) -> None:
        """Clean up empty directories after model deletion."""
        if file_type == "onnx":
            precision_dir = self.shared_onnx_dir / model_id / precision
            model_dir = self.shared_onnx_dir / model_id
        elif file_type == "engine":
            precision_dir = self.shared_engines_dir / model_id / precision
            model_dir = self.shared_engines_dir / model_id
        else:
            return

        # Clean up precision directory if empty
        if precision_dir.exists():
            try:
                if not any(precision_dir.iterdir()):
                    precision_dir.rmdir()
            except OSError:
                pass

        # Clean up model directory if empty
        if model_dir.exists():
            try:
                if not any(model_dir.iterdir()):
                    model_dir.rmdir()
            except OSError:
                pass
