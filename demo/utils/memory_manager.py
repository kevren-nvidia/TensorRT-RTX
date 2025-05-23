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


import time

from cuda import cudart


class ModelMemoryManager:
    """
    Context manager for efficiently loading and unloading models to optimize VRAM usage.

    This class provides two memory optimization modes:

    1. **T5 Offload Mode**: Only for LTX-Video Pipeline, optimized for T5 text encoder workflow
       - At startup: Load all models without workspace allocation
       - At runtime: Allocate T5 buffers/workspace only, run T5,
         then deallocate T5 and allocate workspace for remaining models

    2. **Low VRAM Mode**: Just-in-time model loading
       - Load/allocate models only when needed
       - Deallocate immediately after use

    Args:
        pipeline: The pipeline instance containing engines and model instances
        model_names (list): List of model names to manage
        low_vram (bool): Whether to enable low VRAM mode
        t5_offload (bool): Whether to enable T5 offload mode
    """

    def __init__(self, pipeline, model_name, low_vram=False, t5_offload=False):
        self.pipeline = pipeline
        self.model_name = model_name
        self.low_vram = low_vram
        self.t5_offload = t5_offload
        self.timing = {}
        self.midway_t5_offload = False

        assert (
            self.model_name in self.pipeline.engines
        ), f"Model {self.model_name} not found in pipeline.engines"
        assert isinstance(self.model_name, str), "model_name must be a string"

        # T5 offload logic only applies when managing text_encoder
        self.apply_t5_offload = t5_offload and self.model_name == "text_encoder"

    def __enter__(self):
        if not self.low_vram and not self.apply_t5_offload:
            return self

        if self.apply_t5_offload:
            return self._enter_t5_offload()
        elif self.low_vram:
            return self._enter_low_vram()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.low_vram and not self.apply_t5_offload:
            return

        if self.apply_t5_offload:
            self._exit_t5_offload()
        elif self.low_vram:
            self._exit_low_vram()

    def _enter_t5_offload(self):
        """Handle T5 offload mode entry - allocate T5 workspace and buffers"""
        if self.pipeline.verbose:
            print("[MEMORY] T5 Offload: Allocating T5 workspace and buffers...")

        engine = self.pipeline.engines["text_encoder"]

        start_time = time.time()

        # Free any existing shared memory
        if self.pipeline.shared_device_memory is not None:
            cudart.cudaFree(self.pipeline.shared_device_memory)
            self.pipeline.shared_device_memory = None

        # Allocate memory for text_encoder's workspace
        device_memory_size = engine.engine.device_memory_size
        _, shared_device_memory = cudart.cudaMalloc(device_memory_size)
        self.pipeline.shared_device_memory = shared_device_memory

        # Activate engine with allocated memory
        engine.activate(device_memory=shared_device_memory)

        # Get shape dict for text encoder
        shape_dict = self.pipeline.shape_dicts["text_encoder"]

        # Allocate buffers
        engine.allocate_buffers(shape_dict, device=self.pipeline.device)

        malloc_time = time.time() - start_time
        self.timing["t5_setup"] = malloc_time

        if self.pipeline.verbose:
            memory_gb = device_memory_size / (1024**3)
            print(
                f"[MEMORY] T5 Offload: Allocated {memory_gb:.2f} GB workspace in {malloc_time:.3f}s"
            )

        self.midway_t5_offload = True
        return self

    def _exit_t5_offload(self):
        """Handle T5 offload mode exit - deallocate T5 and setup remaining models"""
        if not self.midway_t5_offload:
            raise RuntimeError("Inconsistent state: Exiting T5 offload context without entering it")

        if self.pipeline.verbose:
            print("[MEMORY] T5 Offload: Deallocating T5 and setting up remaining models...")

        start_time = time.time()

        # Deallocate T5's buffers and deactivate
        self.pipeline.engines["text_encoder"].deallocate_buffers()
        self.pipeline.engines["text_encoder"].deactivate()

        # Free T5's memory
        if self.pipeline.shared_device_memory is not None:
            cudart.cudaFree(self.pipeline.shared_device_memory)
            self.pipeline.shared_device_memory = None

        # Calculate memory needed for remaining models and set their workspace memory
        remaining_engines = {k: v for k, v in self.pipeline.engines.items() if k != "text_encoder"}
        if remaining_engines:
            max_device_memory = max(
                engine.engine.device_memory_size for engine in remaining_engines.values()
            )

            # Allocate shared memory for remaining models
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
            self.pipeline.shared_device_memory = shared_device_memory

            # Set workspace memory for remaining engines (buffers should already be allocated)
            for model_name, engine in remaining_engines.items():
                if hasattr(engine, "context") and engine.context is not None:
                    engine.context.device_memory = shared_device_memory

                    if self.pipeline.verbose:
                        print(f"[MEMORY] T5 Offload: Set workspace for {model_name}")

        cleanup_time = time.time() - start_time
        self.timing["t5_cleanup"] = cleanup_time

        if self.pipeline.verbose:
            print(f"[MEMORY] T5 Offload: Setup completed in {cleanup_time:.3f}s")

        self.midway_t5_offload = False

    def _enter_low_vram(self):
        """Handle low VRAM mode entry - load and allocate specified models"""
        if self.pipeline.verbose:
            print(f"[MEMORY] Low VRAM: Loading models {self.model_name}...")

        if self.model_name not in self.pipeline.shape_dicts:
            raise RuntimeError(f"Model {self.model_name} not found in pipeline.shape_dicts")

        engine = self.pipeline.engines[self.model_name]
        shape_dict = self.pipeline.shape_dicts[self.model_name]

        # Allocate device memory for this model
        start_time = time.time()
        device_memory_size = engine.engine.device_memory_size
        _, device_memory = cudart.cudaMalloc(device_memory_size)
        self.pipeline.shared_device_memory = device_memory

        # Activate engine with allocated memory
        engine.activate(device_memory=device_memory)

        # Allocate buffers
        engine.allocate_buffers(shape_dict, device=self.pipeline.device)

        setup_time = time.time() - start_time
        self.timing[f"{self.model_name}_setup"] = setup_time

        if self.pipeline.verbose:
            memory_gb = device_memory_size / (1024**3)
            print(
                f"[MEMORY] Low VRAM: {self.model_name} allocated {memory_gb:.2f} GB in {setup_time:.3f}s"
            )

        return self

    def _exit_low_vram(self):
        """Handle low VRAM mode exit - deallocate specified models"""
        if self.pipeline.verbose:
            print(f"[MEMORY] Low VRAM: Deallocating models {self.model_name}...")

        engine = self.pipeline.engines[self.model_name]

        start_time = time.time()

        # Deallocate buffers
        engine.deallocate_buffers()

        # Deactivate engine
        engine.deactivate()

        cudart.cudaFree(self.pipeline.shared_device_memory)
        self.pipeline.shared_device_memory = None

        if self.pipeline.verbose:
            print(f"[MEMORY] Low VRAM: Freed workspace memory")

        cleanup_time = time.time() - start_time
        self.timing[f"{self.model_name}_cleanup"] = cleanup_time

        if self.pipeline.verbose:
            print(f"[MEMORY] Low VRAM: {self.model_name} deallocated in {cleanup_time:.3f}s")

    def get_timing_summary(self):
        """Get timing summary for memory operations"""
        return self.timing.copy()
