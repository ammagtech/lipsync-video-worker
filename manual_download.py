#!/usr/bin/env python3
"""
Manual model download script for troubleshooting.
Use this if automatic downloads keep failing.
"""

import os
import subprocess
import sys
from pathlib import Path

def download_with_git():
    """Download models using git clone as fallback."""
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("🔄 Trying git clone method...")
    
    # Download Wan2.1 model with git
    wan_path = f"{models_dir}/Wan2.1-I2V-14B-720P"
    if not os.path.exists(wan_path):
        try:
            print("📥 Cloning Wan2.1 model with git...")
            subprocess.run([
                "git", "clone", 
                "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P",
                wan_path
            ], check=True, timeout=3600)  # 1 hour timeout
            print("✅ Wan2.1 model cloned successfully")
        except Exception as e:
            print(f"❌ Git clone failed: {e}")
            return False
    
    # Download Wav2Vec model with git
    wav2vec_path = f"{models_dir}/wav2vec2-base-960h"
    if not os.path.exists(wav2vec_path):
        try:
            print("📥 Cloning Wav2Vec model with git...")
            subprocess.run([
                "git", "clone",
                "https://huggingface.co/facebook/wav2vec2-base-960h",
                wav2vec_path
            ], check=True, timeout=600)
            print("✅ Wav2Vec model cloned successfully")
        except Exception as e:
            print(f"❌ Wav2Vec git clone failed: {e}")
            return False
    
    return True

def download_with_wget():
    """Download specific files with wget."""
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("🔄 Trying wget method for key files...")
    
    # Key files to download
    files_to_download = [
        {
            "url": "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/diffusion_pytorch_model-00001-of-00007.safetensors",
            "path": f"{models_dir}/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors"
        },
        {
            "url": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json",
            "path": f"{models_dir}/wav2vec2-base-960h/config.json"
        },
        {
            "url": "https://huggingface.co/acvlab/FantasyTalking/resolve/main/fantasytalking_model.ckpt",
            "path": f"{models_dir}/fantasytalking_model.ckpt"
        }
    ]
    
    for file_info in files_to_download:
        url = file_info["url"]
        path = file_info["path"]
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if not os.path.exists(path):
            try:
                print(f"📥 Downloading {os.path.basename(path)}...")
                subprocess.run([
                    "wget", "-O", path, url, "--timeout=300"
                ], check=True)
                print(f"✅ Downloaded {os.path.basename(path)}")
            except Exception as e:
                print(f"❌ Failed to download {os.path.basename(path)}: {e}")
                return False
    
    return True

def check_requirements():
    """Check if required tools are available."""
    tools = ["git", "wget", "curl"]
    available = []
    
    for tool in tools:
        try:
            subprocess.run([tool, "--version"], 
                         capture_output=True, check=True)
            available.append(tool)
            print(f"✅ {tool} is available")
        except:
            print(f"❌ {tool} is not available")
    
    return available

def main():
    print("🛠️ Manual Model Download Tool")
    print("=" * 40)
    
    # Check available tools
    available_tools = check_requirements()
    
    if not available_tools:
        print("❌ No download tools available")
        return 1
    
    # Try different methods
    success = False
    
    if "git" in available_tools:
        print("\n🔄 Attempting git clone method...")
        if download_with_git():
            success = True
            print("✅ Git clone method successful!")
        else:
            print("❌ Git clone method failed")
    
    if not success and "wget" in available_tools:
        print("\n🔄 Attempting wget method...")
        if download_with_wget():
            success = True
            print("✅ Wget method successful!")
        else:
            print("❌ Wget method failed")
    
    if success:
        print("\n🎉 Manual download completed!")
        print("You can now try running the main worker again.")
        return 0
    else:
        print("\n❌ All manual download methods failed")
        print("Please check:")
        print("- Internet connectivity")
        print("- Disk space (need 40GB+)")
        print("- HuggingFace access")
        return 1

if __name__ == "__main__":
    exit(main())
