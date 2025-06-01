#!/usr/bin/env python3
"""
Test script to debug model download issues.
Run this to test model downloads independently.
"""

import os
import sys
import subprocess
from utils import download_models_if_needed, check_model_cache_status

def test_huggingface_cli():
    """Test if huggingface-cli is working."""
    print("🔧 Testing huggingface-cli...")
    try:
        result = subprocess.run(["huggingface-cli", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ huggingface-cli version: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ huggingface-cli failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ huggingface-cli not found")
        return False

def test_individual_downloads():
    """Test downloading each model individually."""
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n🧪 Testing individual model downloads...")
    
    # Test 1: FantasyTalking model (the problematic one)
    print("\n1️⃣ Testing FantasyTalking model download...")
    fantasy_model_path = f"{models_dir}/test_fantasytalking_model.ckpt"
    
    try:
        # Method 1: Direct file download
        print("   Trying method 1: Direct file download...")
        subprocess.run([
            "huggingface-cli", "download", "acvlab/FantasyTalking",
            "fantasytalking_model.ckpt",
            "--local-dir", models_dir,
            "--local-dir-use-symlinks", "False"
        ], check=True, timeout=300)  # 5 minute timeout
        
        if os.path.exists(f"{models_dir}/fantasytalking_model.ckpt"):
            print("   ✅ Method 1 successful!")
            return True
        else:
            print("   ❌ Method 1 failed - file not found")
            
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Method 1 failed: {e}")
        
    except subprocess.TimeoutExpired:
        print("   ❌ Method 1 timed out")
        
    # Method 2: Download entire repo
    try:
        print("   Trying method 2: Download entire repo...")
        fantasy_dir = f"{models_dir}/FantasyTalking_test"
        subprocess.run([
            "huggingface-cli", "download", "acvlab/FantasyTalking",
            "--local-dir", fantasy_dir,
            "--local-dir-use-symlinks", "False"
        ], check=True, timeout=300)
        
        source_file = f"{fantasy_dir}/fantasytalking_model.ckpt"
        if os.path.exists(source_file):
            print("   ✅ Method 2 successful!")
            return True
        else:
            print("   ❌ Method 2 failed - file not found in downloaded repo")
            
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Method 2 failed: {e}")
        
    except subprocess.TimeoutExpired:
        print("   ❌ Method 2 timed out")
    
    return False

def test_alternative_download():
    """Test alternative download methods."""
    print("\n🔄 Testing alternative download methods...")
    
    # Method 3: Using wget/curl
    try:
        print("   Trying method 3: Direct URL download...")
        models_dir = "./models"
        
        # Try to download directly from HuggingFace
        url = "https://huggingface.co/acvlab/FantasyTalking/resolve/main/fantasytalking_model.ckpt"
        output_file = f"{models_dir}/fantasytalking_model_direct.ckpt"
        
        subprocess.run([
            "wget", "-O", output_file, url
        ], check=True, timeout=300)
        
        if os.path.exists(output_file):
            print("   ✅ Method 3 (wget) successful!")
            return True
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"   ❌ Method 3 (wget) failed: {e}")
        
    # Method 4: Using curl
    try:
        print("   Trying method 4: curl download...")
        subprocess.run([
            "curl", "-L", "-o", output_file, url
        ], check=True, timeout=300)
        
        if os.path.exists(output_file):
            print("   ✅ Method 4 (curl) successful!")
            return True
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"   ❌ Method 4 (curl) failed: {e}")
    
    return False

def main():
    print("🧪 FantasyTalking Model Download Test")
    print("=" * 50)
    
    # Test 1: Check huggingface-cli
    if not test_huggingface_cli():
        print("\n❌ huggingface-cli is not working. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"], check=True)
            print("✅ huggingface-cli installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install huggingface-cli")
            return 1
    
    # Test 2: Check current cache status
    print("\n📁 Current cache status:")
    check_model_cache_status()
    
    # Test 3: Try individual downloads
    if test_individual_downloads():
        print("\n✅ Individual download test passed!")
    else:
        print("\n❌ Individual download test failed, trying alternatives...")
        if test_alternative_download():
            print("\n✅ Alternative download method worked!")
        else:
            print("\n❌ All download methods failed")
            return 1
    
    # Test 4: Run full download process
    print("\n🚀 Testing full download process...")
    try:
        download_models_if_needed()
        print("\n✅ Full download process completed!")
    except Exception as e:
        print(f"\n❌ Full download process failed: {e}")
        return 1
    
    # Final status check
    print("\n📊 Final cache status:")
    if check_model_cache_status():
        print("\n🎉 All tests passed! Models are ready.")
        return 0
    else:
        print("\n⚠️ Some models are missing, but basic functionality should work.")
        return 0

if __name__ == "__main__":
    exit(main())
