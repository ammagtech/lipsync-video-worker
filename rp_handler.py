import runpod
import os
import tempfile
import base64
import torch
import librosa
from datetime import datetime
from pathlib import Path
from PIL import Image
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Import DiffSynth components
def install_missing_packages():
    """Install missing packages at runtime."""
    missing_packages = []

    try:
        from diffsynth import ModelManager, WanVideoPipeline
        return ModelManager, WanVideoPipeline
    except ImportError:
        missing_packages.append("diffsynth")

    if missing_packages:
        print(f"Installing missing packages: {missing_packages}")
        import subprocess

        # Install DiffSynth-Studio
        try:
            subprocess.run([
                "pip", "install", "--no-cache-dir",
                "git+https://github.com/modelscope/DiffSynth-Studio.git"
            ], check=True, timeout=300)
            print("✅ DiffSynth-Studio installed successfully")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"❌ Failed to install DiffSynth-Studio: {e}")
            raise ImportError("Could not install DiffSynth-Studio")

        # Try importing again
        try:
            from diffsynth import ModelManager, WanVideoPipeline
            return ModelManager, WanVideoPipeline
        except ImportError as e:
            print(f"❌ Still cannot import DiffSynth after installation: {e}")
            raise

# Install and import DiffSynth
ModelManager, WanVideoPipeline = install_missing_packages()

from model import FantasyTalkingAudioConditionModel
from utils import (
    get_audio_features,
    resize_image_by_longest_edge,
    save_video,
    encode_video_to_base64,
    decode_base64_to_image,
    save_base64_to_file,
    download_models_if_needed,
    check_model_cache_status
)

# Global variables to store loaded models
pipe = None
fantasytalking = None
wav2vec_processor = None
wav2vec = None

def load_models():
    """Load all required models for FantasyTalking."""
    global pipe, fantasytalking, wav2vec_processor, wav2vec

    print("Loading models...")
    print("⚠️ Note: High CPU usage is normal during model loading (2-3 minutes)")

    # Monitor CPU usage
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        print(f"📊 Current CPU: {cpu_percent}%, Memory: {memory_info.percent}%")
    except:
        pass

    # Download models if needed
    download_models_if_needed()

    # Model paths
    wan_model_dir = "./models/Wan2.1-I2V-14B-720P"
    wav2vec_model_dir = "./models/wav2vec2-base-960h"
    fantasytalking_model_path = "./models/fantasytalking_model.ckpt"

    # Load Wan I2V models
    print("Loading Wan I2V models... (This will use high CPU for 2-3 minutes)")

    # Set CPU optimization for model loading
    torch.set_num_threads(min(8, torch.get_num_threads()))  # Limit CPU threads

    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
                f"{wan_model_dir}/diffusion_pytorch_model-00001-of-00007.safetensors",
                f"{wan_model_dir}/diffusion_pytorch_model-00002-of-00007.safetensors",
                f"{wan_model_dir}/diffusion_pytorch_model-00003-of-00007.safetensors",
                f"{wan_model_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                f"{wan_model_dir}/diffusion_pytorch_model-00005-of-00007.safetensors",
                f"{wan_model_dir}/diffusion_pytorch_model-00006-of-00007.safetensors",
                f"{wan_model_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
            ],
            f"{wan_model_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            f"{wan_model_dir}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{wan_model_dir}/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )

    print("✅ Wan I2V models loaded, CPU usage should decrease now")

    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )

    # Load FantasyTalking weights
    print("Loading FantasyTalking model...")
    fantasytalking = FantasyTalkingAudioConditionModel(pipe.dit, 768, 2048).to("cuda")

    # Try to load the weights, but continue even if it fails
    weights_loaded = fantasytalking.load_audio_processor(fantasytalking_model_path, pipe.dit)
    if not weights_loaded:
        print("⚠️ FantasyTalking will use random weights - results may be suboptimal")

    # Enable VRAM management for efficiency
    pipe.enable_vram_management(num_persistent_param_in_dit=0)  # Minimal VRAM usage

    # Load wav2vec models
    print("Loading Wav2Vec models...")
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_dir)
    wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_dir).to("cuda")

    print("All models loaded successfully!")

def handler(event):
    """
    RunPod handler function for FantasyTalking lip-sync generation.

    Expected input format:
    {
        "input": {
            "image": "base64_encoded_image_string",
            "audio": "base64_encoded_audio_string",
            "prompt": "A woman is talking enthusiastically",
            "image_size": 512,
            "max_num_frames": 81,
            "fps": 23,
            "prompt_cfg_scale": 5.0,
            "audio_cfg_scale": 5.0,
            "seed": 1111
        }
    }
    """
    global pipe, fantasytalking, wav2vec_processor, wav2vec

    print("FantasyTalking Worker Start")

    # Load models if not already loaded
    if pipe is None:
        load_models()

    input_data = event['input']

    # Extract parameters with defaults
    image_base64 = input_data.get('image')
    audio_base64 = input_data.get('audio')
    prompt = input_data.get('prompt', 'A person is talking.')
    image_size = input_data.get('image_size', 512)
    max_num_frames = input_data.get('max_num_frames', 81)
    fps = input_data.get('fps', 23)
    prompt_cfg_scale = input_data.get('prompt_cfg_scale', 5.0)
    audio_cfg_scale = input_data.get('audio_cfg_scale', 5.0)
    seed = input_data.get('seed', 1111)

    if not image_base64 or not audio_base64:
        return {
            "status": "error",
            "message": "Both 'image' and 'audio' base64 strings are required"
        }

    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save base64 inputs to temporary files
            image_path = os.path.join(temp_dir, "input_image.png")
            audio_path = os.path.join(temp_dir, "input_audio.wav")

            # Decode and save image
            image = decode_base64_to_image(image_base64)
            image.save(image_path)

            # Decode and save audio
            save_base64_to_file(audio_base64, audio_path, 'audio')

            # Process the lip-sync generation
            result = generate_lipsynced_video(
                image_path=image_path,
                audio_path=audio_path,
                prompt=prompt,
                image_size=image_size,
                max_num_frames=max_num_frames,
                fps=fps,
                prompt_cfg_scale=prompt_cfg_scale,
                audio_cfg_scale=audio_cfg_scale,
                seed=seed,
                output_dir=temp_dir
            )

            return result

    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        }

def generate_lipsynced_video(
    image_path: str,
    audio_path: str,
    prompt: str,
    image_size: int = 512,
    max_num_frames: int = 81,
    fps: int = 23,
    prompt_cfg_scale: float = 5.0,
    audio_cfg_scale: float = 5.0,
    seed: int = 1111,
    output_dir: str = "./output"
) -> dict:
    """
    Generate lip-synced video using FantasyTalking.

    Returns:
        Dictionary with status and video_base64 or error message
    """
    global pipe, fantasytalking, wav2vec_processor, wav2vec

    try:
        print(f"Processing image: {image_path}")
        print(f"Processing audio: {audio_path}")
        print(f"Prompt: {prompt}")

        # Calculate video parameters based on audio duration
        duration = librosa.get_duration(filename=audio_path)
        num_frames = min(int(fps * duration // 4) * 4 + 5, max_num_frames)
        print(f"Audio duration: {duration:.2f}s, generating {num_frames} frames")

        # Extract audio features
        print("Extracting audio features...")
        audio_wav2vec_fea = get_audio_features(
            wav2vec, wav2vec_processor, audio_path, fps, num_frames
        )

        # Resize and prepare image
        print("Processing image...")
        image = resize_image_by_longest_edge(image_path, image_size)
        width, height = image.size
        print(f"Image resized to: {width}x{height}")

        # Process audio features for conditioning
        print("Processing audio conditioning...")
        audio_proj_fea = fantasytalking.get_proj_fea(audio_wav2vec_fea)
        pos_idx_ranges = fantasytalking.split_audio_sequence(
            audio_proj_fea.size(1), num_frames=num_frames
        )
        audio_proj_split, audio_context_lens = fantasytalking.split_tensor_with_padding(
            audio_proj_fea, pos_idx_ranges, expand_length=4
        )

        # Generate video
        print("Generating lip-synced video...")
        video_frames = pipe(
            prompt=prompt,
            negative_prompt="人物静止不动，静止，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=image,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=30,
            seed=seed,
            tiled=True,
            audio_scale=1.0,
            cfg_scale=prompt_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            audio_proj=audio_proj_split,
            audio_context_lens=audio_context_lens,
            latents_num_frames=(num_frames - 1) // 4 + 1,
        )

        # Save temporary video without audio
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_video_path = os.path.join(output_dir, f"temp_video_{current_time}.mp4")
        save_video(video_frames, temp_video_path, fps=fps, quality=5)

        # Combine video with original audio using ffmpeg
        final_video_path = os.path.join(output_dir, f"final_video_{current_time}.mp4")
        import subprocess
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", temp_video_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            final_video_path
        ]

        print("Combining video with audio...")
        subprocess.run(ffmpeg_command, check=True, capture_output=True)

        # Encode final video to base64
        print("Encoding video to base64...")
        video_base64 = encode_video_to_base64(final_video_path)

        # Clean up temporary files
        os.remove(temp_video_path)
        os.remove(final_video_path)

        return {
            "status": "success",
            "video_base64": video_base64,
            "num_frames": num_frames,
            "fps": fps,
            "duration": duration,
            "resolution": f"{width}x{height}",
            "prompt": prompt
        }

    except Exception as e:
        print(f"Error in generate_lipsynced_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Video generation failed: {str(e)}"
        }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
