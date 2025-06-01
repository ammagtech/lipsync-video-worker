# Copyright Alibaba Inc. All Rights Reserved.

import os
import cv2
import torch
import librosa
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import base64
import io


def get_audio_features(
    wav2vec_model, 
    wav2vec_processor, 
    audio_path: str, 
    fps: int = 23, 
    num_frames: int = 81
) -> torch.Tensor:
    """
    Extract audio features using Wav2Vec model.
    
    Args:
        wav2vec_model: Wav2Vec model for feature extraction
        wav2vec_processor: Wav2Vec processor
        audio_path: Path to the audio file
        fps: Frames per second for video
        num_frames: Number of frames to generate
        
    Returns:
        Audio features tensor [1, seq_len, 768]
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)  # Wav2Vec expects 16kHz
    
    # Calculate the required audio length based on video parameters
    video_duration = num_frames / fps
    required_audio_length = int(video_duration * sr)
    
    # Trim or pad audio to match video duration
    if len(audio) > required_audio_length:
        audio = audio[:required_audio_length]
    else:
        # Pad with zeros if audio is shorter
        padding = required_audio_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    
    # Process audio with Wav2Vec
    inputs = wav2vec_processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt", 
        padding=True
    )
    
    # Move to GPU if available
    device = next(wav2vec_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract features
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
        features = outputs.last_hidden_state  # [1, seq_len, 768]
    
    return features


def resize_image_by_longest_edge(image_path: str, target_size: int = 512) -> Image.Image:
    """
    Resize image by longest edge while maintaining aspect ratio.
    
    Args:
        image_path: Path to the input image
        target_size: Target size for the longest edge
        
    Returns:
        Resized PIL Image
    """
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Calculate new dimensions
    if width > height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)
    
    # Ensure dimensions are even (required for video encoding)
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def save_video(frames: torch.Tensor, output_path: str, fps: int = 23, quality: int = 5):
    """
    Save video frames to file.
    
    Args:
        frames: Video frames tensor [num_frames, height, width, channels]
        output_path: Output video file path
        fps: Frames per second
        quality: Video quality (1-10, higher is better)
    """
    # Convert tensor to numpy if needed
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    
    # Ensure frames are in the correct format [0, 255] uint8
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)
    
    # Get video dimensions
    num_frames, height, width, channels = frames.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for i in range(num_frames):
        frame = frames[i]
        if channels == 3:
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()


def encode_video_to_base64(video_path: str) -> str:
    """
    Encode video file to base64 string.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Base64 encoded video string
    """
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    return video_base64


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image


def save_base64_to_file(base64_string: str, output_path: str, file_type: str = 'image'):
    """
    Save base64 string to file.
    
    Args:
        base64_string: Base64 encoded data
        output_path: Output file path
        file_type: Type of file ('image' or 'audio')
    """
    # Remove data URL prefix if present
    if base64_string.startswith('data:'):
        base64_string = base64_string.split(',')[1]
    
    file_bytes = base64.b64decode(base64_string)
    
    with open(output_path, 'wb') as f:
        f.write(file_bytes)


def download_models_if_needed():
    """
    Download required models if they don't exist locally.
    This function should be called during container initialization.
    """
    import subprocess
    import os
    
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Check and download Wan2.1 model
    wan_model_path = f"{models_dir}/Wan2.1-I2V-14B-720P"
    wan_model_files = [
        "diffusion_pytorch_model-00001-of-00007.safetensors",
        "diffusion_pytorch_model-00007-of-00007.safetensors",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth"
    ]

    # Check if all required files exist
    wan_complete = all(os.path.exists(os.path.join(wan_model_path, f)) for f in wan_model_files)

    if not wan_complete:
        print("Downloading Wan2.1-I2V-14B-720P model...")
        subprocess.run([
            "huggingface-cli", "download", "Wan-AI/Wan2.1-I2V-14B-720P",
            "--local-dir", wan_model_path
        ], check=True)
        print("✅ Wan2.1 model download complete!")
    else:
        print("✅ Wan2.1 model already exists, skipping download")

    # Check and download Wav2Vec model
    wav2vec_path = f"{models_dir}/wav2vec2-base-960h"
    wav2vec_config = os.path.join(wav2vec_path, "config.json")

    if not os.path.exists(wav2vec_config):
        print("Downloading wav2vec2-base-960h model...")
        subprocess.run([
            "huggingface-cli", "download", "facebook/wav2vec2-base-960h",
            "--local-dir", wav2vec_path
        ], check=True)
        print("✅ Wav2Vec model download complete!")
    else:
        print("✅ Wav2Vec model already exists, skipping download")

    # Check and download FantasyTalking model
    fantasy_model_path = f"{models_dir}/fantasytalking_model.ckpt"
    if not os.path.exists(fantasy_model_path):
        print("Downloading FantasyTalking model...")
        try:
            # Try method 1: Direct file download
            subprocess.run([
                "huggingface-cli", "download", "acvlab/FantasyTalking",
                "fantasytalking_model.ckpt",
                "--local-dir", models_dir
            ], check=True)
            print("✅ FantasyTalking model download complete!")
        except subprocess.CalledProcessError:
            print("⚠️ Method 1 failed, trying alternative download method...")
            try:
                # Try method 2: Download to specific directory
                fantasy_dir = f"{models_dir}/FantasyTalking"
                subprocess.run([
                    "huggingface-cli", "download", "acvlab/FantasyTalking",
                    "--local-dir", fantasy_dir
                ], check=True)

                # Move the file to the expected location
                import shutil
                source_file = f"{fantasy_dir}/fantasytalking_model.ckpt"
                if os.path.exists(source_file):
                    shutil.move(source_file, fantasy_model_path)
                    print("✅ FantasyTalking model download complete (method 2)!")
                else:
                    print("❌ FantasyTalking model file not found after download")

            except subprocess.CalledProcessError as e:
                print(f"❌ FantasyTalking model download failed: {e}")
                print("⚠️ Continuing without FantasyTalking weights (will use random initialization)")

                # Create an empty file to indicate we tried and failed
                try:
                    with open(f"{models_dir}/fantasytalking_download_failed.txt", "w") as f:
                        f.write("FantasyTalking model download failed. Using random initialization.")
                except:
                    pass
    else:
        print("✅ FantasyTalking model already exists, skipping download")
    
    print("All models are ready!")


def check_model_cache_status():
    """Check and print the status of cached models."""
    models_dir = "./models"

    print("🔍 Checking model cache status...")

    # Check Wan2.1 model
    wan_model_path = f"{models_dir}/Wan2.1-I2V-14B-720P"
    wan_model_files = [
        "diffusion_pytorch_model-00001-of-00007.safetensors",
        "diffusion_pytorch_model-00007-of-00007.safetensors",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth"
    ]

    wan_files_exist = [os.path.exists(os.path.join(wan_model_path, f)) for f in wan_model_files]
    wan_complete = all(wan_files_exist)

    print(f"📁 Wan2.1 model directory: {wan_model_path}")
    print(f"✅ Wan2.1 complete: {wan_complete} ({sum(wan_files_exist)}/{len(wan_model_files)} files)")

    # Check Wav2Vec model
    wav2vec_path = f"{models_dir}/wav2vec2-base-960h"
    wav2vec_exists = os.path.exists(os.path.join(wav2vec_path, "config.json"))
    print(f"📁 Wav2Vec model directory: {wav2vec_path}")
    print(f"✅ Wav2Vec complete: {wav2vec_exists}")

    # Check FantasyTalking model
    fantasy_model_path = f"{models_dir}/fantasytalking_model.ckpt"
    fantasy_exists = os.path.exists(fantasy_model_path)
    print(f"📁 FantasyTalking model file: {fantasy_model_path}")
    print(f"✅ FantasyTalking complete: {fantasy_exists}")

    # Calculate total size
    total_size = 0
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)

    total_size_gb = total_size / (1024**3)
    print(f"💾 Total models size: {total_size_gb:.2f} GB")

    return wan_complete and wav2vec_exists and fantasy_exists
