#!/usr/bin/env python3
"""
Model download script for FantasyTalking
Downloads required models from HuggingFace and ModelScope
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from modelscope import snapshot_download as ms_snapshot_download

def create_models_dir():
    """Create models directory structure"""
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    return models_dir

def download_wan_model(models_dir):
    """Download Wan2.1-I2V-14B-720P base model"""
    print("Downloading Wan2.1-I2V-14B-720P model...")
    wan_dir = models_dir / "Wan2.1-I2V-14B-720P"
    
    try:
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-720P",
            local_dir=str(wan_dir),
            local_dir_use_symlinks=False
        )
        print("✓ Wan2.1-I2V-14B-720P downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download Wan2.1-I2V-14B-720P: {e}")
        return False
    return True

def download_wav2vec_model(models_dir):
    """Download Wav2Vec2 audio encoder"""
    print("Downloading Wav2Vec2 model...")
    wav2vec_dir = models_dir / "wav2vec2-base-960h"
    
    try:
        snapshot_download(
            repo_id="facebook/wav2vec2-base-960h",
            local_dir=str(wav2vec_dir),
            local_dir_use_symlinks=False
        )
        print("✓ Wav2Vec2 downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download Wav2Vec2: {e}")
        return False
    return True

def download_fantasytalking_model(models_dir):
    """Download FantasyTalking checkpoint"""
    print("Downloading FantasyTalking model...")
    ft_dir = models_dir / "fantasytalking_model"
    ft_dir.mkdir(exist_ok=True)
    
    try:
        hf_hub_download(
            repo_id="acvlab/FantasyTalking",
            filename="fantasytalking_model.ckpt",
            local_dir=str(ft_dir),
            local_dir_use_symlinks=False
        )
        print("✓ FantasyTalking model downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download FantasyTalking model: {e}")
        return False
    return True

def main():
    """Main download function"""
    print("Starting model downloads for FantasyTalking...")
    
    models_dir = create_models_dir()
    
    # Download all required models
    success = True
    success &= download_wan_model(models_dir)
    success &= download_wav2vec_model(models_dir)
    success &= download_fantasytalking_model(models_dir)
    
    if success:
        print("\n✓ All models downloaded successfully!")
        print("Models are ready for inference.")
    else:
        print("\n✗ Some models failed to download.")
        print("Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
