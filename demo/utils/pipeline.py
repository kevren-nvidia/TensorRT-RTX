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
Simple Pipeline for ONNX -> TRT-RTX flow
"""

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import torch
from cuda import cudart

from .model_registry import registry
from .path_manager import PathManager


class Pipeline(ABC):
    """
    Simple pipeline for ONNX -> TRT-RTX flow.

    Usage:
    1. Create Pipeline with precision config
    2. Load Engines (builds if needed)
    3. Activate Engines (load into memory)
    4. Load Resources (allocate buffers)
    5. Run Inference
    """

    def __init__(
        self,
        pipeline_name: str,
        cache_dir: str = "./demo_cache",
        precision_config: Optional[Dict[str, str]] = None,
        device: str = "cuda",
        verbose: bool = True,
        cache_mode: str = "full",
    ):
        """
        Initialize pipeline.

        Args:
            pipeline_name: Name of the pipeline
            cache_dir: Directory for caching models
            precision_config: Model precision configuration
            device: Device to run on
            verbose: Enable verbose logging
            cache_mode: Cache management mode ("full" or "lean")
        """
        # Initialize path manager
        self.path_manager = PathManager(cache_dir=cache_dir, verbose=verbose, cache_mode=cache_mode)

        # Core attributes
        self.pipeline_name = pipeline_name
        self.device = device
        self.verbose = verbose

        # Pipeline state
        self.engines = {}
        self.model_instances = {}
        self.shape_dicts = {}
        self.current_shapes = None
        self.stream = None
        self.shared_device_memory = None

        # Get default precision config from registry
        try:
            self.precision_config = registry.get_default_precisions(pipeline_name)
        except (KeyError, AttributeError):
            # Fallback if pipeline not in registry
            self.precision_config = {}

        # Override with user-provided precision config
        if precision_config:
            self.precision_config.update(precision_config)

        if self.verbose:
            print(f"[INFO] Pipeline: {pipeline_name}")
            print(f"[INFO] Initial Precision Config: {self.precision_config}")

    def get_model_names(self) -> list:
        """Return list of model names used by this pipeline"""
        raise NotImplementedError("Subclasses must implement get_model_names")

    def initialize_models(self):
        """Initialize model objects - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement initialize_models")

    @abstractmethod
    def load_engines(self) -> None:
        """Load TensorRT engines, building if necessary."""
        raise NotImplementedError("Subclasses must implement load_engines")

    def activate_engines(self, shared_device_memory=None) -> Dict[str, float]:
        """Activate all engines (load into memory)"""
        jit_times = {}

        if shared_device_memory is None:
            max_device_memory = self.calculate_max_device_memory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)

        self.shared_device_memory = shared_device_memory

        if self.verbose:
            print("\n[INFO] Activating engines...")

        # Load and activate TensorRT engines
        for model_name, engine in self.engines.items():
            if self.verbose:
                print(f"[INFO]   Activating {model_name}")

            jit_time = engine.activate(device_memory=self.shared_device_memory)
            jit_times[model_name] = jit_time

        # Initialize CUDA stream
        self.stream = cudart.cudaStreamCreate()[1]

        if self.verbose:
            total_jit_time = sum(jit_times.values())
            print(f"[INFO] All engines activated (total JIT time: {total_jit_time:.3f}s)")

        return jit_times

    @abstractmethod
    def load_resources(self, **shape_params) -> None:
        """
        Allocate buffers for inference.

        Args:
            **shape_params: Shape parameters (e.g., batch_size=1, height=512, width=512)
        """
        raise NotImplementedError("Subclasses must implement load_resources")

    def calculate_max_device_memory(self) -> int:
        """Calculate maximum device memory needed across all engines"""
        if not self.engines:
            return 0

        max_device_memory = 0
        total_engine_memory = 0

        print(f"\n[MEMORY] Calculating shared workspace requirements:")

        for model_name, engine in self.engines.items():
            engine_memory = engine.engine.device_memory_size_v2
            total_engine_memory += engine_memory
            max_device_memory = max(max_device_memory, engine_memory)

            print(f"[MEMORY]   {model_name}: {engine_memory / (1024 ** 3):.3f} GB")

        workspace_savings = (total_engine_memory - max_device_memory) / (1024**3)
        print(f"[MEMORY] Shared workspace (max): {max_device_memory / (1024 ** 3):.3f} GB")
        print(f"[MEMORY] Memory savings: {workspace_savings:.3f} GB")

        return max_device_memory

    def run_engine(
        self, model_name: str, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Run inference on a specific engine"""
        engine = self.engines[model_name]
        return engine.infer(inputs, self.stream, use_cuda_graph=False)

    def infer(self, *args, **kwargs):
        """Run the full pipeline inference - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement infer")

    def cleanup(self) -> None:
        """Clean up all resources"""
        if self.verbose:
            print("[INFO] Cleaning up pipeline...")

        # Deactivate engines
        for model_name, engine in self.engines.items():
            if self.verbose:
                print(f"[INFO]   Deactivating {model_name}")
            engine.deallocate_buffers()
            engine.deactivate()
            engine.unload()
            del engine

        if self.shared_device_memory is not None:
            print(f"[MEMORY] Freeing shared workspace memory")
            cudart.cudaFree(self.shared_device_memory)
            self.shared_device_memory = None

        cudart.cudaStreamDestroy(self.stream)
        self.stream = None

        # Clear state
        self.engines.clear()
        self.model_instances.clear()
        self.current_shapes = None

        # Clear GPU memory
        print(f"[MEMORY] Clearing GPU cache")
        torch.cuda.empty_cache()
        gc.collect()

    def __del__(self):
        self.cleanup()

    def calculate_total_gpu_vram_usage(self) -> dict:
        """Calculate total GPU VRAM usage estimate using formula: weights + workspace + buffers"""
        # Model weights (serialized engine sizes)
        total_weights = 0
        engine_weights = {}

        for model_name, engine in self.engines.items():
            engine_path = Path(engine.engine_path)
            if engine_path.exists():
                weight_size = engine_path.stat().st_size
                engine_weights[model_name] = weight_size
                total_weights += weight_size

        # Shared workspace memory
        max_workspace = self.calculate_max_device_memory()

        # Buffer allocations
        total_buffers = 0
        buffer_breakdown = {}

        for model_name, engine in self.engines.items():
            model_buffer_size = 0
            for tensor in engine.tensors.values():
                model_buffer_size += tensor.numel() * tensor.element_size()
            buffer_breakdown[model_name] = model_buffer_size
            total_buffers += model_buffer_size

        # Total VRAM estimate
        total_vram = total_weights + max_workspace + total_buffers

        return {
            "total_weights_gb": total_weights / (1024**3),
            "max_workspace_gb": max_workspace / (1024**3),
            "total_buffers_gb": total_buffers / (1024**3),
            "total_vram_gb": total_vram / (1024**3),
            "breakdown": {
                "weights": {name: size / (1024**3) for name, size in engine_weights.items()},
                "buffers": {name: size / (1024**3) for name, size in buffer_breakdown.items()},
            },
        }

    def print_gpu_vram_summary(self):
        """Print GPU VRAM usage summary"""
        vram_info = self.calculate_total_gpu_vram_usage()

        print(f"\n[VRAM] Total estimated usage: {vram_info['total_vram_gb']:.2f} GB")
        print(f"[VRAM]   Model weights: {vram_info['total_weights_gb']:.2f} GB")
        print(f"[VRAM]   Shared workspace: {vram_info['max_workspace_gb']:.2f} GB")
        print(f"[VRAM]   Buffer allocations: {vram_info['total_buffers_gb']:.2f} GB")

        print(f"[VRAM] Per-model breakdown:")
        for model_name in self.engines.keys():
            weights = vram_info["breakdown"]["weights"].get(model_name, 0)
            buffers = vram_info["breakdown"]["buffers"].get(model_name, 0)
            print(f"[VRAM]   {model_name}: {weights:.2f}GB weights + {buffers:.2f}GB buffers")

        return vram_info
