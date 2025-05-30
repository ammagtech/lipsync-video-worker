#!/usr/bin/env python3
"""
Model download script for FantasyTalking
Downloads required models from HuggingFace and ModelScope
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from modelscope import snapshot_download as ms_snapshot_download

def download_with_retry(download_func, max_retries=3):
    """Wrapper to retry downloads with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return download_func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = 2 ** attempt
            print(f"Download failed, retrying in {wait_time} seconds...")
            time.sleep(wait_time)

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
        def download():
            return snapshot_download(
                repo_id="Wan-AI/Wan2.1-I2V-14B-720P",
                local_dir=str(wan_dir),
                local_dir_use_symlinks=False
            )
        download_with_retry(download)
        print("✓ Wan2.1-I2V-14B-720P downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download Wan2.1-I2V-14B-720P: {e}")
        return False

def download_wav2vec2_model(models_dir):
    """Download wav2vec2-base-960h model"""
    print("Downloading wav2vec2-base-960h model...")
    try:
        wav2vec_path = models_dir / "wav2vec2-base-960h"
        wav2vec_path.mkdir(exist_ok=True)
        
        def download():
            return snapshot_download(
                repo_id="facebook/wav2vec2-base-960h",
                local_dir=str(wav2vec_path),
                local_dir_use_symlinks=False
            )
        download_with_retry(download)
        print("✓ wav2vec2-base-960h model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading wav2vec2-base-960h model: {e}")
        return False

def download_fantasytalking_model(models_dir):
    """Download FantasyTalking checkpoint"""
    print("Downloading FantasyTalking model...")
    ft_dir = models_dir / "fantasytalking_model"
    ft_dir.mkdir(exist_ok=True)
    
    try:
        def download():
            return hf_hub_download(
                repo_id="acvlab/FantasyTalking",
                filename="fantasytalking_model.ckpt",
                local_dir=str(ft_dir),
                local_dir_use_symlinks=False
            )
        download_with_retry(download)
        print("✓ FantasyTalking model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download FantasyTalking model: {e}")
        return False

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
    models_dir = create_models_dir()
    success = True
    
    # Download all required models
    success &= download_wan_model(models_dir)
    success &= download_wav2vec2_model(models_dir)
    
    if not success:
        print("✗ Some models failed to download")
        sys.exit(1)
    print("✓ All models downloaded successfully")
