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
LTX Video 2B Pipeline

Comprehensive pipeline for LTX Video generation (text-to-video and image-to-video)
using the ONNX → TRT flow.
"""

import gc
import inspect
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from cuda import cudart

# Add the parent directory to the Python path so we can import utils and models
sys.path.append(str(Path(__file__).parent.parent))

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.video_processor import VideoProcessor
from tqdm.auto import tqdm
from transformers import T5TokenizerFast

from demo.models.ltx_video.ltx_video_model_params import LTXVideo2BModelParams
from demo.models.ltx_video.ltx_video_transformer import LTXVideoTransformer
from demo.models.ltx_video.ltx_video_vae import LTXVideoVAEDecoder, LTXVideoVAEEncoder
from demo.models.ltx_video.t5_text_encoder import T5TextEncoder
from demo.utils.engine import Engine
from demo.utils.engine_metadata import metadata_manager
from demo.utils.memory_manager import ModelMemoryManager
from demo.utils.model_registry import registry as model_registry
from demo.utils.pipeline import Pipeline


@dataclass
class InferenceTimingData:
    """Container for detailed inference timing data using CUDA events."""

    # Per-component timings (in milliseconds)
    text_encoder_time: float = 0.0
    transformer_times: List[float] = field(default_factory=list)  # Per-iteration times
    vae_decoder_time: float = 0.0
    vae_encoder_time: float = 0.0  # For img2vid

    # Total times
    total_inference_time: float = 0.0

    # Metadata
    num_inference_steps: int = 0
    timesteps: List[int] = field(default_factory=list)
    height: int = 0
    width: int = 0
    num_frames: int = 0
    batch_size: int = 0
    guidance_scale: float = 0.0

    def get_transformer_stats(self) -> Dict[str, float]:
        """Get statistics for transformer iterations."""
        if not self.transformer_times:
            return {}

        times = np.array(self.transformer_times)
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "total": float(np.sum(times)),
        }

    def plot_transformer_timing(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot transformer timing per iteration.

        Args:
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if not self.transformer_times:
            print("No transformer timing data available for plotting")
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot timing per iteration
        iterations = list(range(1, len(self.transformer_times) + 1))
        ax.plot(iterations, self.transformer_times, "b-o", linewidth=2, markersize=4)
        ax.set_xlabel("Inference Iteration")
        ax.set_ylabel("Duration (ms)")
        ax.set_title(
            r"${{\bf Transformer\ Inference\ Duration\ per\ Iteration}}$"
            f"\n({self.height}x{self.width}), {self.num_frames} frames"
        )
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats = self.get_transformer_stats()
        stats_text = (
            r"${{\bf Mean}}$" + f": {stats['mean']:.2f}ms\n"
            r"${{\bf Std}}$" + f": {stats['std']:.2f}ms\n"
            r"${{\bf Min}}$" + f": {stats['min']:.2f}ms\n"
            r"${{\bf Max}}$" + f": {stats['max']:.2f}ms"
        )
        ax.text(
            0.85,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def print_summary(self):
        """Print a summary of timing data."""
        print("\n" + "=" * 60)
        print("INFERENCE TIMING SUMMARY")
        print("=" * 60)
        print(f"Configuration: {self.height}x{self.width}, {self.num_frames} frames")
        print(f"Batch size: {self.batch_size}, Guidance scale: {self.guidance_scale}")
        print(f"Inference steps: {self.num_inference_steps}")
        print("-" * 60)

        print(f"Text Encoder:     {self.text_encoder_time:8.2f} ms")
        if self.vae_encoder_time > 0:
            print(f"VAE Encoder:      {self.vae_encoder_time:8.2f} ms")

        if self.transformer_times:
            stats = self.get_transformer_stats()
            print(f"Transformer:      {stats['total']:8.2f} ms (total)")
            print(f"  - Per iteration: {stats['mean']:8.2f} ± {stats['std']:.2f} ms")
            print(f"  - Range:         {stats['min']:.2f} - {stats['max']:.2f} ms")

        print(f"VAE Decoder:      {self.vae_decoder_time:8.2f} ms")
        print("-" * 60)
        print(f"Total Inference:  {self.total_inference_time:8.2f} ms")

        # Calculate throughput
        if self.total_inference_time > 0:
            fps = (self.num_frames * 1000) / self.total_inference_time
            print(f"Throughput:       {fps:8.2f} frames/second")

        print("=" * 60)


class LTXVideo2BPipeline(Pipeline):
    """
    LTX Video 2B Pipeline

    Supports video generation with:
    - Text-to-video (txt2vid): Generate video from text prompt
    - Image-to-video (img2vid): Generate video from text prompt + reference image
    - Smart caching using PathManager
    - Engine metadata tracking
    - Model registry integration
    - Precision-specific configurations
    - Interactive workflow
    """

    def __init__(
        self,
        pipeline_type: str = "txt2vid",  # "txt2vid" or "img2vid"
        cache_dir: str = "./demo_cache",
        precision_config: Optional[Dict[str, str]] = None,
        device: str = "cuda",
        verbose: bool = True,
        cache_mode: str = "full",
        memory_mode: str = "normal",  # "normal", "t5_offload", "low_vram"
    ):
        """
        Initialize LTX Video 2B Pipeline.

        Args:
            pipeline_type: Type of pipeline ("txt2vid" or "img2vid")
            cache_dir: Directory for caching engines and ONNX models
            precision_config: Precision configuration per model
            device: Device to run on ("cuda" or "cpu")
            verbose: Enable verbose logging
            cache_mode: Cache mode ("full" or "lean")
            memory_mode: Memory management mode ("normal", "t5_offload", "low_vram")
        """
        # Validate pipeline type
        if pipeline_type not in ["txt2vid", "img2vid"]:
            raise ValueError(
                f"Invalid pipeline_type: {pipeline_type}. Must be 'txt2vid' or 'img2vid'"
            )

        # Validate memory mode for LTX (supports t5_offload)
        valid_memory_modes = ["normal", "t5_offload", "low_vram"]
        if memory_mode not in valid_memory_modes:
            raise ValueError(
                f"Invalid memory_mode: {memory_mode}. Must be one of {valid_memory_modes}"
            )

        # Determine pipeline name based on type
        pipeline_name = f"ltx_video_2b_{pipeline_type}"

        super().__init__(
            pipeline_name=pipeline_name,
            cache_dir=cache_dir,
            device=device,
            verbose=verbose,
            precision_config=precision_config,
            cache_mode=cache_mode,
        )

        assert model_registry.get_pipeline_config(
            self.pipeline_name
        ), f"Pipeline {self.pipeline_name} not registered"

        self.pipeline_type = pipeline_type
        self.memory_mode = memory_mode

        # Memory management properties
        self.t5_offload = memory_mode == "t5_offload"
        self.low_vram = memory_mode == "low_vram"

        # Pipeline state
        self.scheduler = None
        self.tokenizer = None

        # Model parameters
        self.model_params = LTXVideo2BModelParams()

        self.video_processor = VideoProcessor()

        # Default values
        self.num_inference_steps = self.model_params.NUM_INFERENCE_STEPS
        self.guidance_scale = self.model_params.GUIDANCE_SCALE
        self.num_videos_per_prompt = self.model_params.NUM_VIDEOS_PER_PROMPT

        # Timing infrastructure
        self.enable_timing = True  # Can be disabled for production
        self.timing_data = None
        self._cuda_events = {}  # Store CUDA events for reuse

    def _create_cuda_events(self, name: str) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """Create and cache CUDA events for timing."""
        if name not in self._cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            self._cuda_events[name] = (start_event, end_event)
        return self._cuda_events[name]

    def _record_cuda_timing(self, name: str, func, *args, **kwargs):
        """
        Record CUDA timing for a function call using events.

        Args:
            name: Name for the timing event
            func: Function to time
            *args, **kwargs: Arguments for the function

        Returns:
            Function result and elapsed time in milliseconds
        """
        if not self.enable_timing:
            return func(*args, **kwargs), 0.0

        start_event, end_event = self._create_cuda_events(name)

        # Ensure CUDA is synchronized before starting
        torch.cuda.synchronize()
        start_event.record()

        result = func(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)
        return result, elapsed_time

    def reset_timing_data(self):
        """Reset timing data for a new inference run."""
        self.timing_data = InferenceTimingData()

    def model_memory_manager(self, model_names, low_vram=False):
        """Returns a context manager for model memory management.

        This helper method creates a ModelMemoryManager instance for efficient
        loading and unloading of models to optimize VRAM usage.

        Args:
            model_names (list): List of model names to manage with this context.
            low_vram (bool, optional): Whether to enable VRAM optimization. Defaults to False.

        Returns:
            ModelMemoryManager: Context manager for model memory management.
        """
        # Pass t5_offload flag to the ModelMemoryManager
        return ModelMemoryManager(self, model_names, low_vram=low_vram, t5_offload=self.t5_offload)

    def _get_model_configs(self) -> Dict[str, Tuple[str, str]]:
        """Get model configurations from registry."""
        # Get from registry
        pipeline_config = model_registry.get_pipeline_config(self.pipeline_name)
        model_configs = {}

        for role, model_id in pipeline_config.items():
            precision = self.precision_config.get(role)
            model_configs[role] = (model_id, precision)

        return model_configs

    def _initialize_models(self):
        """Initialize model instances."""
        if self.verbose:
            print(f"[I] Initializing models for {self.pipeline_name}...")

        # Initialize Tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained(
            self.model_params.PIPELINE_SOURCE, subfolder="tokenizer"
        )

        # Initialize Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_params.PIPELINE_SOURCE, subfolder="scheduler"
        )

        # Create model instances
        self.model_instances = {
            "text_encoder": T5TextEncoder(
                name="ltx_video_t5_text_encoder",
                device=self.device,
                verbose=self.verbose,
                max_sequence_length=self.model_params.MAX_SEQUENCE_LENGTH,
                model_params=self.model_params,
            ),
            "transformer": LTXVideoTransformer(
                name=f"ltx_video_transformer_2b_{self.pipeline_type}",
                device=self.device,
                verbose=self.verbose,
                do_classifier_free_guidance=self.guidance_scale > 1.0,
                is_image_pipeline=(self.pipeline_type == "img2vid"),
                model_params=self.model_params,
            ),
            "vae_decoder": LTXVideoVAEDecoder(
                name="ltx_video_vae_2b_decoder",
                device=self.device,
                verbose=self.verbose,
                model_params=self.model_params,
            ),
        }

        # Add VAE encoder for img2vid pipeline
        if self.pipeline_type == "img2vid":
            self.model_instances["vae_encoder"] = LTXVideoVAEEncoder(
                name="ltx_video_vae_2b_encoder",
                device=self.device,
                verbose=self.verbose,
                model_params=self.model_params,
            )

    def load_engines(
        self,
        precisions: Optional[Dict[str, str]] = None,
        opt_batch_size: int = 1,
        opt_height: int = 512,
        opt_width: int = 704,
        opt_num_frames: int = 161,
        static_shape: Optional[bool] = None,  # Deprecated, use shape_config instead
        shape_config: Optional[Dict[str, str]] = None,
        extra_args: Optional[Set[str]] = None,
    ):
        """
        Load or build TensorRT engines with smart caching.

        Args:
            precisions: Precision configuration per model
            opt_batch_size: Optimal batch size
            opt_height: Optimal image height
            opt_width: Optimal image width
            opt_num_frames: Optimal number of frames
            static_shape: Global static shape setting (deprecated, use shape_config)
            shape_config: Per-model shape configuration {"model_role": "static"/"dynamic"}
            extra_args: Additional polygraphy arguments
        """
        if precisions is not None:
            assert isinstance(precisions, dict), "precisions must be a dictionary"
            assert all(
                isinstance(role, str) and isinstance(precision, str)
                for role, precision in precisions.items()
            ), "precisions must be a dictionary of role: precision pairs"
            assert all(
                role in model_registry.get_pipeline_roles(self.pipeline_name)
                for role in precisions.keys()
            ), f"Invalid role in precisions: {precisions.keys()}, options: {model_registry.get_pipeline_roles(self.pipeline_name)}"
            assert all(
                precision in model_registry.get_available_precisions(self.pipeline_name, role)
                for role, precision in precisions.items()
            ), f"Invalid precision for role {role}: {precision}, options: {model_registry.get_available_precisions(self.pipeline_name, role)}"

        # Handle shape configuration
        if shape_config is not None and static_shape is not None:
            raise ValueError(
                "Cannot specify both shape_config and static_shape. Use shape_config for per-model control."
            )

        if shape_config is not None:
            # Validate shape_config
            assert isinstance(shape_config, dict), "shape_config must be a dictionary"
            valid_modes = ["static", "dynamic"]
            for role, mode in shape_config.items():
                assert role in model_registry.get_pipeline_roles(
                    self.pipeline_name
                ), f"Invalid role in shape_config: {role}, options: {model_registry.get_pipeline_roles(self.pipeline_name)}"
                assert (
                    mode in valid_modes
                ), f"Invalid shape mode for {role}: {mode}, options: {valid_modes}"
        elif static_shape is not None:
            # Convert legacy static_shape to shape_config
            if self.verbose:
                print("[W] static_shape parameter is deprecated, use shape_config instead")
            shape_mode = "static" if static_shape else "dynamic"
            shape_config = {
                role: shape_mode for role in model_registry.get_pipeline_roles(self.pipeline_name)
            }
        else:
            # Default: all models use static shapes
            shape_config = {
                role: "static" for role in model_registry.get_pipeline_roles(self.pipeline_name)
            }

        # Set precisions, with priority to precisions passed in
        self.precision_config = {**self.precision_config, **(precisions or {})}

        # Get model configurations
        model_configs = self._get_model_configs()

        # Update path manager with current pipeline models
        self.path_manager.set_pipeline_models(self.pipeline_name, model_configs)

        # Initialize models
        self._initialize_models()

        # Process each model
        for role, (model_id, precision) in model_configs.items():
            if self.verbose:
                print(f"\n[I] Processing {role} ({model_id}_{precision})...")

            # Get shape mode for this model
            use_static_shape = shape_config[role] == "static"

            # Get file paths directly from path manager
            onnx_path = self.path_manager.get_onnx_path(model_id, precision)
            engine_path = self.path_manager.get_engine_path(model_id, precision)

            # Check if ONNX exists
            if not onnx_path.exists():
                onnx_source = model_registry.get_onnx_url(self.pipeline_name, role, precision)
                if onnx_source is not None:
                    if self.verbose:
                        print(f"[I] Acquiring ONNX from registry: {onnx_source}")

                    success = self.path_manager.acquire_onnx_file(
                        self.pipeline_name, model_id, precision, onnx_source, verbose=self.verbose
                    )

                    if not success:
                        raise AssertionError(
                            f"[E] Failed to acquire ONNX for {model_id}_{precision}"
                        )
                else:
                    raise ValueError(f"[E] Model {model_id}_{precision} not found in registry")

            # Check if engine needs rebuilding
            engine = None
            if engine_path.exists():
                # Check metadata compatibility
                target_shapes = (
                    self.model_instances[role].get_input_profile(
                        opt_batch_size, opt_height, opt_width, opt_num_frames, static_shape=True
                    )
                    if role != "text_encoder"
                    else self.model_instances[role].get_input_profile(
                        opt_batch_size, static_shape=True
                    )
                )

                is_compatible, reason = metadata_manager.check_engine_compatibility(
                    engine_path=engine_path,
                    target_shapes=target_shapes,
                    static_shape=use_static_shape,
                    extra_args=extra_args,
                )

                if is_compatible:
                    if self.verbose:
                        print(f"[I] Using cached engine: {engine_path.name}")
                else:
                    if self.verbose:
                        print(f"[I] Recompiling {model_id}_{precision}: {reason}")
                    self.path_manager.delete_cached_engine_files(model_id, precision)

            # Build engine if needed
            if not engine_path.exists():
                if onnx_path.exists():
                    if self.verbose:
                        build_reason = (
                            "No cached engine found"
                            if not engine_path.with_suffix(".metadata.json").exists()
                            else "Engine cache cleared"
                        )
                        shape_type = "static" if use_static_shape else "dynamic"
                        print(
                            f"[I] Building {model_id}_{precision} engine ({shape_type} shapes): {build_reason}"
                        )
                else:
                    if self.verbose:
                        print(
                            f"[E] Cannot build engine: ONNX file not found for {model_id}_{precision}"
                        )
                    continue

                # Get input profile
                if role == "text_encoder":
                    input_profile = self.model_instances[role].get_input_profile(
                        opt_batch_size, static_shape=use_static_shape
                    )
                else:
                    input_profile = self.model_instances[role].get_input_profile(
                        opt_batch_size,
                        opt_height,
                        opt_width,
                        opt_num_frames,
                        static_shape=use_static_shape,
                    )

                engine = Engine(engine_path, precision, model_id)

                engine.build(
                    onnx_path=str(onnx_path),
                    input_profile=input_profile,
                    static_shape=use_static_shape,
                    verbose=self.verbose,
                    extra_args=extra_args,
                )

            # Create engine instance and load it
            if engine is None:
                engine = Engine(engine_path, precision, model_id)

            engine.load()

            # Store engine in pipeline
            self.engines[role] = engine

        # Store shape configuration for later use
        self.shape_config = shape_config

        if self.verbose:
            print(f"\n[I] All engines loaded for {self.pipeline_name}")
            print(f"[I] Shape configuration: {shape_config}")
            self.path_manager.print_cache_summary()

        print("[I] Activating engines...")
        self.activate_engines()
        print("[I] Engines activated successfully")

    def load_resources(
        self,
        batch_size: int = 1,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        seed: int = 0,
    ):
        """Load resources needed for inference."""
        if seed is not None:
            self.seed = seed

        self.generator = torch.Generator(device="cuda").manual_seed(self.seed)

        self.stream = cudart.cudaStreamCreate()[1]

        if self.verbose:
            print(f"\n[INFO] Loading resources: {batch_size}, {height}, {width}, {num_frames}")

        current_shapes = (batch_size, height, width, num_frames)

        # Check if we need to reallocate
        if self.current_shapes is not None and self.current_shapes:
            if self.current_shapes == current_shapes:
                if self.verbose:
                    print("[INFO] Resources already allocated")
                return
            else:
                if self.verbose:
                    print(
                        f"[INFO] Current shapes: {self.current_shapes}, new shapes: {current_shapes}"
                    )

                # Deallocate existing buffers
                for engine in self.engines.values():
                    engine.deallocate_buffers()

        #  Get shape dicts and allocate buffers for engines that have contexts
        for model_name, engine in self.engines.items():
            model = self.model_instances[model_name]
            if model_name == "text_encoder":
                shape_dict = model.get_shape_dict(batch_size)
            else:
                shape_dict = model.get_shape_dict(batch_size, height, width, num_frames)

            self.shape_dicts[model_name] = shape_dict

            if self.low_vram:
                if self.verbose:
                    print(f"[INFO]   Skipping buffer allocation (low VRAM mode)")
                continue

            # Skip text_encoder in T5 offload mode - it has no context yet
            if self.t5_offload and model_name == "text_encoder":
                if self.verbose:
                    print(f"[INFO]   Skipping {model_name} (T5 offload mode - no context yet)")
                continue

            engine.allocate_buffers(shape_dict, device=self.device)

            if self.verbose:
                print(f"[INFO]   Allocated buffers for {model_name}")

        self.current_shapes = current_shapes

        if self.verbose:
            print(f"\n[INFO] Resources loaded successfully")
            # Print comprehensive GPU VRAM usage summary
            self.print_gpu_vram_summary()

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        dtype: Optional[torch.dtype] = None,
    ):
        """Generate T5 embeddings for the prompt.

        Args:
            prompt (str or List[str]): The text prompt to encode
            num_videos_per_prompt (int): Number of videos per prompt
            dtype (torch.dtype): Data type for embeddings

        Returns:
            Tuple containing prompt embeddings and attention mask
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model_params.MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(self.device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.model_params.MAX_SEQUENCE_LENGTH - 1 : -1]
            )
            print(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {self.model_params.MAX_SEQUENCE_LENGTH} tokens: {removed_text}"
            )

        outputs = self.run_engine("text_encoder", {"input_ids": text_input_ids.to(self.device)})
        prompt_embeds = outputs["text_embeddings"]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Encode the text prompt into embeddings.

        Args:
            prompt (str or List[str]): The text prompt to encode
            negative_prompt (str or List[str], optional): Negative text prompt
            do_classifier_free_guidance (bool): Whether to use classifier-free guidance
            num_videos_per_prompt (int): Number of videos per prompt
            prompt_embeds (torch.Tensor, optional): Pre-computed prompt embeddings
            negative_prompt_embeds (torch.Tensor, optional): Pre-computed negative prompt embeddings
            prompt_attention_mask (torch.Tensor, optional): Attention mask for prompt embeddings
            negative_prompt_attention_mask (torch.Tensor, optional): Attention mask for negative prompt embeddings
            dtype (torch.dtype, optional): Data type for embeddings

        Returns:
            Tuple containing prompt embeddings, attention mask, negative prompt embeddings, and negative prompt attention mask
        """
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                dtype=dtype,
            )
        else:
            assert (
                prompt_attention_mask is not None
            ), "prompt_attention_mask must be provided if prompt_embeds is provided"
            prompt_embeds = prompt_embeds.to(self.device)
            prompt_attention_mask = prompt_attention_mask.to(self.device)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                [negative_prompt] * len(prompt)
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif len(prompt) != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {len(prompt)}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            (
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                dtype=dtype,
            )

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def encode_image(
        self,
        image: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode reference image for img2vid pipeline.

        Args:
            image: Reference image tensor
            generator: Random generator for VAE sampling

        Returns:
            (image_latents, conditioning_mask) for the first frame

        TODO: Implement image encoding logic
        """
        if self.pipeline_type != "img2vid":
            raise ValueError("encode_image is only for img2vid pipeline")

        raise NotImplementedError("TODO: Implement image encoding for img2vid")

    def prepare_latents_img2vid(self, *args, **kwargs):
        raise NotImplementedError("TODO: Implement image encoding for img2vid")

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepare latents for the diffusion process.

        Args:
            batch_size (int): Batch size
            num_channels_latents (int): Number of channels in latents
            height (int): Height of the output image
            width (int): Width of the output image
            num_frames (int): Number of video frames
            dtype (torch.dtype, optional): Data type for latents
            generator (torch.Generator, optional): Random number generator
            latents (torch.Tensor, optional): Pre-generated latents

        Returns:
            torch.Tensor: Prepared latents
        """
        if latents is not None:
            return latents.to(device=self.device, dtype=dtype)

        height = height // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        width = width // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        num_frames = (num_frames - 1) // self.model_params.VAE_TEMPORAL_COMPRESSION_RATIO + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Generate random noise
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=dtype)

        # Pack latents
        latents = self._pack_latents(
            latents,
            self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE,
            self.model_params.TRANSFORMER_TEMPORAL_PATCH_SIZE,
        )
        return latents

    def denoise_latents(
        self,
        latents,
        latent_height,
        latent_width,
        latent_num_frames,
        timesteps,
        text_embeddings,
        attention_mask,
        rope_interpolation_scale,
        conditioning_mask=None,
    ):
        """Run the denoising process.

        Args:
            latents (torch.Tensor): Initial latents
            latent_height (int): Height of the latents
            latent_width (int): Width of the latents
            latent_num_frames (int): Number of frames in the latents
            timesteps (torch.Tensor): Timesteps for the diffusion process
            text_embeddings (torch.Tensor): Text embeddings
            attention_mask (torch.Tensor): Attention mask for text embeddings
            rope_interpolation_scale (tuple): Interpolation scale for RoPE
            conditioning_mask (Optional[torch.Tensor]): Conditioning mask for the latents
        Returns:
            torch.Tensor: Denoised latents
        """
        if self.pipeline_type == "img2vid":
            assert (
                conditioning_mask is not None
            ), "Conditioning mask is required for image-to-video pipelines"

        do_classifier_free_guidance = self.guidance_scale > 1.0
        latent_model_input = latents
        batch_size = latents.shape[0]

        # Generate video coordinates for rotary position embeddings
        grid_bs = 2 * batch_size if do_classifier_free_guidance else batch_size
        cos_freqs, sin_freqs = self._generate_video_coords(
            latent_height,
            latent_width,
            latent_num_frames,
            rope_interpolation_scale,
            grid_bs,
        )

        # Run denoising for each timestep
        num_inference_steps = len(timesteps)

        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self._current_timestep = t

                # Double the latent for classifier-free guidance if needed
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)

                # Prepare timestep (expanding to batch)
                timestep = t.expand(latent_model_input.shape[0])
                if self.pipeline_type == "img2vid":
                    timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask)

                latent_model_input = latent_model_input.to(text_embeddings.dtype)
                # text_embeddings = text_embeddings.to(latent_model_input.dtype)

                # Use TensorRT engine for inference with timing
                inputs = {
                    "hidden_states": latent_model_input,
                    "encoder_hidden_states": text_embeddings,
                    "timestep": timestep,
                    "encoder_attention_mask": attention_mask,
                    "image_rotary_emb_cos": cos_freqs,
                    "image_rotary_emb_sin": sin_freqs,
                }

                # Record transformer timing for this iteration
                def run_transformer():
                    return self.run_engine("transformer", inputs)

                outputs, iteration_time = self._record_cuda_timing(
                    f"transformer_iter_{i}", run_transformer
                )
                noise_pred = outputs["latent"]

                # Store timing data
                if self.timing_data is not None:
                    self.timing_data.transformer_times.append(iteration_time)

                noise_pred = noise_pred.float()

                # Apply classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    if self.pipeline_type == "img2vid":
                        timestep, _ = timestep.chunk(2)

                # Noise and latent processing + scheduler step
                if self.pipeline_type == "img2vid":
                    # compute the previous noisy sample x_t -> x_t-1
                    noise_pred = self._unpack_latents(
                        noise_pred,
                        latent_num_frames,
                        latent_height,
                        latent_width,
                        self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE,
                        self.model_params.TRANSFORMER_TEMPORAL_PATCH_SIZE,
                    )
                    latents = self._unpack_latents(
                        latents,
                        latent_num_frames,
                        latent_height,
                        latent_width,
                        self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE,
                        self.model_params.TRANSFORMER_TEMPORAL_PATCH_SIZE,
                    )

                    # Scheduler step
                    noise_pred = noise_pred[:, :, 1:]
                    noise_latents = latents[:, :, 1:]
                    pred_latents = self.scheduler.step(
                        noise_pred, t, noise_latents, return_dict=False
                    )[0]
                    latents = torch.cat([latents[:, :, :1], pred_latents], dim=2)
                    latents = self._pack_latents(
                        latents,
                        self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE,
                        self.model_params.TRANSFORMER_TEMPORAL_PATCH_SIZE,
                    )
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Update progress bar with timing info
                if self.enable_timing and iteration_time > 0:
                    progress_bar.set_postfix({"iter_time": f"{iteration_time:.1f}ms"})
                progress_bar.update()

        return latents

    def decode_latents(self, latents, timestep=None):
        """Decode latents to video frames.

        Args:
            latents (torch.Tensor): Latents to decode
            timestep (torch.Tensor, optional): Timestep tensor for conditional decoding

        Returns:
            torch.Tensor: Decoded video frames
        """
        inputs = {"latent": latents}
        if timestep is not None:
            inputs["timestep"] = timestep
        video = self.run_engine("vae_decoder", inputs)["video"]

        return video

    # Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
    def retrieve_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):
        r"""
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
                `num_inference_steps` and `sigmas` must be `None`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(
                inspect.signature(self.scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {self.scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            self.scheduler.set_timesteps(timesteps=timesteps, device=self.device, **kwargs)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(
                inspect.signature(self.scheduler.set_timesteps).parameters.keys()
            )
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {self.scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            self.scheduler.set_timesteps(sigmas=sigmas, device=self.device, **kwargs)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, **kwargs)
            timesteps = self.scheduler.timesteps
        return timesteps, num_inference_steps

    def run_demo(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[torch.Tensor] = None,  # For img2vid pipeline
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        frame_rate: int = 24,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        guidance_scale: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Run the LTX Video generation pipeline.

        Args:
            prompt: Text prompt(s) for generation
            negative_prompt: Negative prompt(s) for guidance
            image: Reference image for img2vid pipeline
            height: Output video height
            width: Output video width
            num_frames: Number of frames to generate
            frame_rate: Frame rate of the output video
            num_inference_steps: Number of denoising steps
            timesteps: Custom timestep schedule
            guidance_scale: Classifier-free guidance scale
            num_videos_per_prompt: Videos to generate per prompt
            seed: Random seed for reproducibility
            generator: Random number generator for reproducibility
        Returns:
            Generated videos and timing information
        """
        # Validate inputs
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"Height and width must be divisible by 32, got {height}x{width}")

        if self.pipeline_type == "img2vid" and image is None:
            raise ValueError("img2vid pipeline requires a reference image")
        elif self.pipeline_type == "txt2vid" and image is not None:
            print("[W] txt2vid pipeline ignores reference image")
            image = None

        # Convert single prompts to list
        if not isinstance(prompt, list):
            prompt = [prompt]

        if negative_prompt is not None and not isinstance(negative_prompt, list):
            negative_prompt = [negative_prompt]

        batch_size = len(prompt)

        # Update parameters if provided
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps
        if guidance_scale is not None:
            assert (
                guidance_scale > 1.0
            ), f"Currently, guidance_scale for LTX-Video must be greater than 1.0, got {guidance_scale}"
            self.guidance_scale = guidance_scale
        if num_videos_per_prompt is not None:
            self.num_videos_per_prompt = num_videos_per_prompt

        # Run load_resources to refresh shapes as needed
        print("[I] Loading resources for inference...")
        self.load_resources(
            batch_size=(batch_size * self.num_videos_per_prompt),
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
        )

        do_classifier_free_guidance = self.guidance_scale > 1.0

        # Initialize timing data
        if self.enable_timing:
            self.reset_timing_data()
            self.timing_data.num_inference_steps = self.num_inference_steps
            self.timing_data.height = height
            self.timing_data.width = width
            self.timing_data.num_frames = num_frames
            self.timing_data.batch_size = batch_size
            self.timing_data.guidance_scale = self.guidance_scale

        # Start timing
        start_time = time.perf_counter()
        total_start_event, total_end_event = self._create_cuda_events("total_inference")
        torch.cuda.synchronize()
        total_start_event.record()

        # Compute latent dimensions
        latent_height = height // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        latent_width = width // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        latent_num_frames = (num_frames - 1) // self.model_params.VAE_TEMPORAL_COMPRESSION_RATIO + 1

        with torch.inference_mode():
            rope_interpolation_scale = (
                self.model_params.VAE_TEMPORAL_COMPRESSION_RATIO / frame_rate,
                self.model_params.VAE_SPATIAL_COMPRESSION_RATIO,
                self.model_params.VAE_SPATIAL_COMPRESSION_RATIO,
            )

            # 1. Encode prompt with memory management and timing
            if self.verbose:
                print("[I] Encoding prompt...")

            def encode_prompt_func():
                with self.model_memory_manager("text_encoder", low_vram=self.low_vram):
                    return self.encode_prompt(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        num_videos_per_prompt=self.num_videos_per_prompt,
                    )

            (
                text_embeds,
                attention_mask,
                negative_embeds,
                negative_attention_mask,
            ), text_encoder_time = self._record_cuda_timing("text_encoder", encode_prompt_func)

            if self.timing_data is not None:
                self.timing_data.text_encoder_time = text_encoder_time

            # Concatenate for classifier-free guidance if needed
            if do_classifier_free_guidance:
                text_embeds = torch.cat([negative_embeds, text_embeds], dim=0)
                attention_mask = torch.cat([negative_attention_mask, attention_mask], dim=0)

            # 2. Prepare timesteps
            if self.verbose:
                print(f"[I] Preparing {self.num_inference_steps} denoising steps...")
            sigmas = torch.linspace(1.0, 1 / self.num_inference_steps, self.num_inference_steps)
            video_sequence_length = latent_num_frames * latent_height * latent_width
            mu = self.calculate_shift(
                video_sequence_length,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )

            timesteps, num_inference_steps = self.retrieve_timesteps(
                num_inference_steps=self.num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                mu=mu,
            )

            self._num_timesteps = len(timesteps)

            # Store timesteps for timing data
            if self.timing_data is not None:
                self.timing_data.timesteps = timesteps.cpu().tolist()

            # 3. Prepare latents
            if self.verbose:
                print("[I] Preparing initial latents...")

            if self.pipeline_type == "txt2vid":
                latents = self.prepare_latents(
                    batch_size=batch_size,
                    num_channels_latents=self.model_instances["transformer"].config["in_channels"],
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    generator=self.generator,
                )
                conditioning_mask = None
            else:
                image = self.video_processor.preprocess(image, height=height, width=width)
                image = image.to(
                    device=self.device,
                    dtype=model_registry.get_torch_dtype(self.precision_config["vae_encoder"]),
                )

                def encode_image_func():
                    with self.model_memory_manager("vae_encoder", low_vram=self.low_vram):
                        return self.prepare_latents_img2vid(
                            image=image,
                            batch_size=batch_size,
                            num_channels_latents=self.model_instances["transformer"].config[
                                "in_channels"
                            ],
                            height=height,
                            width=width,
                            num_frames=num_frames,
                            generator=self.generator,
                        )

                (latents, conditioning_mask), vae_encoder_time = self._record_cuda_timing(
                    "vae_encoder", encode_image_func
                )

                if self.timing_data is not None:
                    self.timing_data.vae_encoder_time = vae_encoder_time

                del image

                if self.do_classifier_free_guidance:
                    conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])

            # 4. Denoising loop with memory management (transformer timing recorded inside denoise_latents)
            if self.verbose:
                print("[I] Running denoising loop...")

            with self.model_memory_manager("transformer", low_vram=self.low_vram):
                latents = self.denoise_latents(
                    latents=latents,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    latent_num_frames=latent_num_frames,
                    timesteps=timesteps,
                    text_embeddings=text_embeds,
                    attention_mask=attention_mask,
                    rope_interpolation_scale=rope_interpolation_scale,
                    conditioning_mask=conditioning_mask,
                )

            del (
                text_embeds,
                timesteps,
                attention_mask,
                negative_embeds,
                negative_attention_mask,
                conditioning_mask,
            )

            gc.collect()
            torch.cuda.empty_cache()

            # 5. Decode latents with memory management and timing
            if self.verbose:
                print("[I] Decoding latents to video...")
            latents = self._unpack_latents(
                latents,
                latent_num_frames,
                latent_height,
                latent_width,
                self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE,
                self.model_params.TRANSFORMER_TEMPORAL_PATCH_SIZE,
            )

            # Denormalize latents if needed
            latents = self._denormalize_latents(
                latents,
                self.model_instances["vae_decoder"].latents_mean,
                self.model_instances["vae_decoder"].latents_std,
                self.model_instances["vae_decoder"].config.get("scaling_factor", 1.0),
            )

            # Convert to the dtype expected by the VAE
            latents = latents.to(
                model_registry.get_torch_dtype(self.precision_config["vae_decoder"])
            )

            # Prepare timestep conditioning for VAE if needed
            timestep = None

            # Decode to video frames with memory management and timing
            def decode_latents_func():
                with self.model_memory_manager("vae_decoder", low_vram=self.low_vram):
                    return self.decode_latents(latents, timestep=timestep)

            videos, vae_decoder_time = self._record_cuda_timing("vae_decoder", decode_latents_func)

            if self.timing_data is not None:
                self.timing_data.vae_decoder_time = vae_decoder_time

            # 6. Post-process outputs
            videos = self.video_processor.postprocess_video(videos, output_type="pil")

        # Calculate total timing
        total_end_event.record()
        torch.cuda.synchronize()
        total_inference_time = total_start_event.elapsed_time(total_end_event)

        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # ms

        if self.timing_data is not None:
            self.timing_data.total_inference_time = total_inference_time

        if self.verbose:
            print(
                f"[I] Generation completed in {inference_time:.2f}ms (CPU) / {total_inference_time:.2f}ms (CUDA)"
            )
            print(f"[I] Generated {num_frames} frames")
            print(f"[I] Throughput: {num_frames / (total_inference_time / 1000):.2f} frames/s")

        return videos

    def warmup(self, height: int = 512, width: int = 704, num_frames: int = 161, num_runs: int = 3):
        """
        Run warmup iterations to initialize CUDA kernels.

        Args:
            height: Height for warmup
            width: Width for warmup
            num_frames: Number of frames
            num_runs: Number of warmup iterations
        """
        if self.verbose:
            print(f"[I] Running {num_runs} warmup iterations...")

        # Create dummy image for img2vid
        dummy_image = None
        if self.pipeline_type == "img2vid":
            dummy_image = torch.randn(1, 3, height, width, device=self.device)

        for i in range(num_runs):
            _ = self.run_demo(
                prompt="warmup",
                image=dummy_image,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=2,  # Minimal steps for warmup
                return_dict=False,
            )

        if self.verbose:
            print("[I] Warmup completed")

    def get_timing_data(self) -> Optional[InferenceTimingData]:
        """
        Get the timing data from the last inference run.

        Returns:
            InferenceTimingData object or None if no timing data available
        """
        return self.timing_data

    def print_timing_summary(self):
        """Print a summary of the last inference timing."""
        if self.timing_data is None:
            print("No timing data available. Run inference first.")
            return

        self.timing_data.print_summary()

    def plot_timing(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot timing data from the last inference run.

        Args:
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if self.timing_data is None:
            print("No timing data available. Run inference first.")
            return None

        return self.timing_data.plot_transformer_timing(save_path=save_path, show=show)

    def save_timing_data(self, filepath: str):
        """
        Save timing data to a JSON file.

        Args:
            filepath: Path to save the timing data
        """
        if self.timing_data is None:
            print("No timing data available. Run inference first.")
            return

        import json
        from dataclasses import asdict

        timing_dict = asdict(self.timing_data)

        with open(filepath, "w") as f:
            json.dump(timing_dict, f, indent=2)

        print(f"Timing data saved to: {filepath}")

    def load_timing_data(self, filepath: str) -> InferenceTimingData:
        """
        Load timing data from a JSON file.

        Args:
            filepath: Path to load the timing data from

        Returns:
            InferenceTimingData object
        """
        import json

        with open(filepath, "r") as f:
            timing_dict = json.load(f)

        # Reconstruct the InferenceTimingData object
        timing_data = InferenceTimingData(**timing_dict)
        return timing_data

    def compare_timing_runs(
        self,
        timing_data_list: List[InferenceTimingData],
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Compare timing data from multiple runs.

        Args:
            timing_data_list: List of InferenceTimingData objects to compare
            labels: Optional labels for each run
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if not timing_data_list:
            print("No timing data provided for comparison.")
            return None

        if labels is None:
            labels = [f"Run {i+1}" for i in range(len(timing_data_list))]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Transformer timing comparison
        colors = plt.cm.tab10(np.linspace(0, 1, len(timing_data_list)))

        for i, (timing_data, label) in enumerate(zip(timing_data_list, labels)):
            if timing_data.transformer_times:
                iterations = list(range(1, len(timing_data.transformer_times) + 1))
                ax1.plot(
                    iterations,
                    timing_data.transformer_times,
                    "o-",
                    color=colors[i],
                    linewidth=2,
                    markersize=3,
                    label=label,
                )

        ax1.set_xlabel("Inference Iteration")
        ax1.set_ylabel("Duration (ms)")
        ax1.set_title("Transformer Timing Comparison")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Component timing comparison
        component_names = ["Text Encoder", "Transformer (Total)", "VAE Decoder"]
        x_pos = np.arange(len(component_names))
        bar_width = 0.8 / len(timing_data_list)

        for i, (timing_data, label) in enumerate(zip(timing_data_list, labels)):
            component_times = [
                timing_data.text_encoder_time,
                sum(timing_data.transformer_times) if timing_data.transformer_times else 0,
                timing_data.vae_decoder_time,
            ]

            if timing_data.vae_encoder_time > 0:
                if i == 0:  # Only extend for first run to avoid multiple extensions
                    component_names.append("VAE Encoder")
                    x_pos = np.arange(len(component_names))
                component_times.append(timing_data.vae_encoder_time)

            ax2.bar(
                x_pos + i * bar_width,
                component_times,
                bar_width,
                label=label,
                color=colors[i],
                alpha=0.8,
            )

        ax2.set_xlabel("Component")
        ax2.set_ylabel("Duration (ms)")
        ax2.set_title("Component Timing Comparison")
        ax2.set_xticks(x_pos + bar_width * (len(timing_data_list) - 1) / 2)
        ax2.set_xticklabels(component_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comparison plot saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def _pack_latents(
        self, latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        """Pack latents into tokens.

        Args:
            latents (torch.Tensor): Latent vectors to pack
            patch_size (int): Spatial patch size
            patch_size_t (int): Temporal patch size

        Returns:
            torch.Tensor: Packed latents
        """
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens
        batch_size, _, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    def _unpack_latents(
        self,
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        """Unpack latents from tokens to video tensor.

        Args:
            latents (torch.Tensor): Packed latents to unpack
            num_frames (int): Number of frames
            height (int): Height of the video
            width (int): Width of the video
            patch_size (int): Spatial patch size
            patch_size_t (int): Temporal patch size

        Returns:
            torch.Tensor: Unpacked latents
        """
        # Packed latents of shape [B, S, D] are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]
        batch_size = latents.size(0)
        latents = latents.reshape(
            batch_size,
            num_frames,
            height,
            width,
            -1,
            patch_size_t,
            patch_size,
            patch_size,
        )
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    def _normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        """Normalize latents across the channel dimension.

        Args:
            latents (torch.Tensor): Latents to normalize
            latents_mean (torch.Tensor): Mean for normalization
            latents_std (torch.Tensor): Standard deviation for normalization
            scaling_factor (float): Scaling factor

        Returns:
            torch.Tensor: Normalized latents
        """
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    def _denormalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        """Denormalize latents across the channel dimension.

        Args:
            latents (torch.Tensor): Latents to denormalize
            latents_mean (torch.Tensor): Mean for denormalization
            latents_std (torch.Tensor): Standard deviation for denormalization
            scaling_factor (float): Scaling factor

        Returns:
            torch.Tensor: Denormalized latents
        """
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def _generate_video_coords(
        self,
        latent_height,
        latent_width,
        latent_num_frames,
        rope_interpolation_scale,
        grid_bs,
    ):
        """Generate video coordinates for rotary position embeddings.

        Args:
            latent_height (int): Height of the latents
            latent_width (int): Width of the latents
            latent_num_frames (int): Number of frames
            rope_interpolation_scale (tuple): Interpolation scale for RoPE
            grid_bs (int): Batch size for the grid

        Returns:
            Tuple of cosine and sine position embeddings
        """
        import math

        grid_h = torch.arange(latent_height, dtype=torch.float32, device=self.device)
        grid_w = torch.arange(latent_width, dtype=torch.float32, device=self.device)
        grid_f = torch.arange(latent_num_frames, dtype=torch.float32, device=self.device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        grid = grid.unsqueeze(0).repeat(grid_bs, 1, 1, 1, 1)

        if rope_interpolation_scale is not None:
            grid[:, 0:1] = (
                grid[:, 0:1]
                * rope_interpolation_scale[0]
                * self.model_params.TRANSFORMER_TEMPORAL_PATCH_SIZE
                / 20
            )
            grid[:, 1:2] = (
                grid[:, 1:2]
                * rope_interpolation_scale[1]
                * self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE
                / 2048
            )
            grid[:, 2:3] = (
                grid[:, 2:3]
                * rope_interpolation_scale[2]
                * self.model_params.TRANSFORMER_SPATIAL_PATCH_SIZE
                / 2048
            )

        grid = grid.flatten(2, 4).transpose(1, 2)

        start = 1.0
        end = 10000.0
        freqs = 10000.0 ** torch.linspace(
            math.log(start, 10000),
            math.log(end, 10000.0),
            self.model_instances["transformer"].inner_dim // 6,
            device=self.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
        freqs = freqs.transpose(-1, -2).flatten(2)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        inner_dim = self.model_instances["transformer"].inner_dim
        if inner_dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : inner_dim % 6])
            sin_padding = torch.zeros_like(cos_freqs[:, :, : inner_dim % 6])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs

    def activate_engines(self, shared_device_memory=None):
        """Activate all engines.

        For t5_offload mode, this activates all engines except the text encoder,
        which will be activated separately during inference.

        For low_vram mode, this does nothing - engines are activated by memory manager.
        """
        jit_times = {}
        if shared_device_memory is not None and self.t5_offload:
            raise AssertionError(
                "Enabling t5 offload with pre-specified shared device mem is not allowed"
            )

        if self.t5_offload:
            # T5 offload mode: activate all engines except text encoder with deferred memory
            for model_name, engine in self.engines.items():
                if model_name != "text_encoder":
                    jit_time = engine.activate(defer_memory_allocation=True)
                    jit_times[model_name] = jit_time

            if self.verbose:
                total_jit_time = sum(jit_times.values())
                print(
                    f"[INFO] T5 Offload engines activated (total JIT time: {total_jit_time:.3f}s)"
                )
        elif self.low_vram:
            if self.verbose:
                print(
                    f"[INFO] Low VRAM mode: Engines will be activated just-in-time by memory manager"
                )
        else:
            # Normal mode: activate all engines
            jit_times = super().activate_engines(shared_device_memory)

        return jit_times
