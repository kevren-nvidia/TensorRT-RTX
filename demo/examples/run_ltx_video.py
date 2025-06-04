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
Launcher script for LTX Video 2B Pipeline
This script demonstrates shape configuration and memory management features.
"""

import sys
from pathlib import Path

# Add parent directory to Python path to access utils and models
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffusers.utils import export_to_video

from demo.pipelines.ltx_video_2b import LTXVideo2BPipeline

# Example usage with different configurations
print("LTX Video 2B Pipeline - Advanced Configuration Demo")
print("=" * 50)

########################################################################################
# Configuration 1: High Performance with T5 Offload
########################################################################################
print("\n1. High Performance Setup (T5 Offload + Mixed Precision)")
pipeline = LTXVideo2BPipeline(
    pipeline_type="txt2vid",
    cache_dir="./demo_cache",
    verbose=True,
    precision_config={
        "text_encoder": "bf16",  # Standard precision for text
        "transformer": "fp8",  # High performance for main model
        "vae_decoder": "fp16",  # Balance for video output
    },
    memory_mode="t5_offload",  # Efficient T5 memory management
    cache_mode="full",  # Clean up unused models
)

shape_config = {
    "text_encoder": "static",  # Text encoder can be static
    "transformer": "static",  # Transformer needs flexibility for different resolutions
    "vae_decoder": "static",  # VAE decoder can be static
}

height = 480
width = 704
num_frames = 161

# Load engines with per-model shape configuration
pipeline.load_engines(
    shape_config=shape_config, opt_height=height, opt_width=width, opt_num_frames=num_frames
)

pipeline.activate_engines()

pipeline.load_resources(
    batch_size=1,
    height=height,
    width=width,
    num_frames=num_frames,
    seed=42,
)

# Generate video
video = pipeline.run_demo(
    prompt="A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom. The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks.",
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    frame_rate=24,
    num_inference_steps=50,
    guidance_scale=3.0,
    num_videos_per_prompt=1,
    height=height,
    width=width,
    num_frames=num_frames,
)

print("[I] Video generation complete")

# Save the video file
export_to_video(video[0], "output_t5_offload.mp4", fps=24)
print(f"[I] Video saved to output_t5_offload.mp4")

pipeline.cleanup()

########################################################################################
# Configuration 2: Low VRAM Mode Example
########################################################################################
print("\n2. Low VRAM Setup (Just-in-time Loading)")
pipeline_low_vram = LTXVideo2BPipeline(
    pipeline_type="txt2vid",
    cache_dir="./demo_cache",
    verbose=True,
    precision_config={"transformer": "fp8", "vae_decoder": "fp16", "text_encoder": "bf16"},
    memory_mode="low_vram",
    cache_mode="lean",
)

pipeline_low_vram.load_engines(
    shape_config={"text_encoder": "static", "transformer": "dynamic", "vae_decoder": "static"},
    opt_height=512,
    opt_width=704,
    opt_num_frames=161,
)

# JIT-Allocated Resources and generate video
video_low_vram = pipeline_low_vram.run_demo(
    prompt="A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom. The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks.",
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    num_videos_per_prompt=1,
    height=512,
    width=704,
    num_frames=161,
    num_inference_steps=30,
    guidance_scale=3.0,
    frame_rate=24,
    seed=42,
)

export_to_video(video_low_vram[0], "output_low_vram.mp4", fps=24)
print(f"[I] Low VRAM video saved to output_low_vram.mp4")

# Print detailed timing summary
print("\n[3] Timing Analysis:")
pipeline_low_vram.print_timing_summary()

# Generate and save timing plots
print("\n[4] Generating timing plots...")
fig = pipeline_low_vram.plot_timing(
    save_path="ltx_timing_analysis.png", show=False  # Set to True if you want to display the plot
)

pipeline_low_vram.cleanup()
