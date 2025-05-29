"""
Configuration file for FantasyTalking RunPod worker
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path("/")
MODELS_DIR = BASE_DIR / "models"

# Model paths
WAN_MODEL_PATH = MODELS_DIR / "Wan2.1-I2V-14B-720P"
WAV2VEC_MODEL_PATH = MODELS_DIR / "wav2vec2-base-960h"
FANTASYTALKING_MODEL_PATH = MODELS_DIR / "fantasytalking_model" / "fantasytalking_model.ckpt"

# Default inference settings
DEFAULT_CONFIG = {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "prompt_cfg_scale": 5.0,
    "audio_cfg_scale": 5.0,
    "height": 512,
    "width": 512,
    "num_frames": 81,
    "fps": 8,
    "torch_dtype": "torch.bfloat16",
    "num_persistent_param_in_dit": 7000000000,  # 7B for memory optimization
}

# Supported file formats
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".m4a"]

# Memory optimization settings
MEMORY_OPTIMIZATION = {
    "enable_attention_slicing": True,
    "enable_cpu_offload": True,
    "enable_sequential_cpu_offload": False,
    "low_mem_mode": True
}

# Output settings
OUTPUT_CONFIG = {
    "video_format": "mp4",
    "audio_sample_rate": 24000,
    "video_codec": "libx264",
    "audio_codec": "aac"
}
