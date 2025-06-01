#!/usr/bin/env python3
"""
Troubleshooting script for model download issues.
Run this to diagnose download problems.
"""

import subprocess
import os
import sys
import urllib.request
import shutil

def test_basic_connectivity():
    """Test basic internet connectivity."""
    print("🌐 Testing basic connectivity...")
    
    urls_to_test = [
        "https://google.com",
        "https://huggingface.co",
        "https://github.com"
    ]
    
    for url in urls_to_test:
        try:
            response = urllib.request.urlopen(url, timeout=10)
            print(f"✅ {url} - OK ({response.getcode()})")
        except Exception as e:
            print(f"❌ {url} - Failed: {e}")
            return False
    
    return True

def test_huggingface_cli():
    """Test HuggingFace CLI functionality."""
    print("\n🔧 Testing HuggingFace CLI...")
    
    # Check if CLI is installed
    try:
        result = subprocess.run(["huggingface-cli", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ CLI version: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ CLI not working: {e}")
        return False
    
    # Test login status
    try:
        result = subprocess.run(["huggingface-cli", "whoami"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Logged in as: {result.stdout.strip()}")
        else:
            print("⚠️ Not logged in (may be required for some models)")
    except Exception as e:
        print(f"⚠️ Cannot check login status: {e}")
    
    return True

def test_disk_space():
    """Test available disk space."""
    print("\n💾 Testing disk space...")
    
    try:
        total, used, free = shutil.disk_usage("./")
        free_gb = free // (1024**3)
        total_gb = total // (1024**3)
        used_gb = used // (1024**3)
        
        print(f"📊 Disk usage:")
        print(f"   Total: {total_gb}GB")
        print(f"   Used:  {used_gb}GB")
        print(f"   Free:  {free_gb}GB")
        
        if free_gb < 50:
            print(f"❌ Insufficient disk space: {free_gb}GB (need 50GB+)")
            return False
        else:
            print(f"✅ Sufficient disk space: {free_gb}GB")
            return True
            
    except Exception as e:
        print(f"❌ Cannot check disk space: {e}")
        return False

def test_model_access():
    """Test access to specific models."""
    print("\n🎯 Testing model access...")
    
    models_to_test = [
        "Wan-AI/Wan2.1-I2V-14B-720P",
        "facebook/wav2vec2-base-960h",
        "acvlab/FantasyTalking"
    ]
    
    for model in models_to_test:
        print(f"\n🧪 Testing {model}...")
        try:
            # Try to download just the README
            test_dir = f"./test_{model.replace('/', '_')}"
            os.makedirs(test_dir, exist_ok=True)
            
            result = subprocess.run([
                "huggingface-cli", "download", model,
                "README.md", "--local-dir", test_dir
            ], check=True, timeout=60, capture_output=True, text=True)
            
            print(f"✅ {model} - Accessible")
            
            # Clean up
            shutil.rmtree(test_dir, ignore_errors=True)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {model} - Failed (exit code {e.returncode})")
            if e.stderr:
                print(f"   Error: {e.stderr[:200]}...")
        except subprocess.TimeoutExpired:
            print(f"❌ {model} - Timeout")
        except Exception as e:
            print(f"❌ {model} - Error: {e}")

def test_git_access():
    """Test git clone access."""
    print("\n🔄 Testing git clone access...")
    
    try:
        # Test git availability
        result = subprocess.run(["git", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Git version: {result.stdout.strip()}")
        
        # Test git clone of a small repo
        test_dir = "./test_git_clone"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            
        result = subprocess.run([
            "git", "clone", "--depth", "1",
            "https://huggingface.co/facebook/wav2vec2-base-960h",
            test_dir
        ], check=True, timeout=120, capture_output=True, text=True)
        
        print("✅ Git clone works")
        
        # Clean up
        shutil.rmtree(test_dir, ignore_errors=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git clone failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr[:200]}...")
        return False
    except Exception as e:
        print(f"❌ Git not available: {e}")
        return False

def suggest_solutions():
    """Suggest solutions based on test results."""
    print("\n💡 Suggested Solutions:")
    print("=" * 40)
    print("1. **Network Issues:**")
    print("   - Try a different RunPod region")
    print("   - Check if RunPod has network restrictions")
    print("   - Wait and try again (HuggingFace may be down)")
    print()
    print("2. **Authentication Issues:**")
    print("   - Some models may require HuggingFace login")
    print("   - Run: huggingface-cli login")
    print("   - Use HF_TOKEN environment variable")
    print()
    print("3. **Disk Space Issues:**")
    print("   - Increase container disk size to 80GB+")
    print("   - Clean up existing files")
    print()
    print("4. **Alternative Download Methods:**")
    print("   - Use git clone instead of huggingface-cli")
    print("   - Download models manually and upload")
    print("   - Use different model repositories")

def main():
    print("🔍 FantasyTalking Download Troubleshooter")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        ("Basic Connectivity", test_basic_connectivity),
        ("HuggingFace CLI", test_huggingface_cli),
        ("Disk Space", test_disk_space),
        ("Git Access", test_git_access),
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_tests_passed = False
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            all_tests_passed = False
    
    # Test model access (separate because it's more likely to fail)
    test_model_access()
    
    # Provide suggestions
    suggest_solutions()
    
    print(f"\n📊 Overall Status: {'✅ PASS' if all_tests_passed else '❌ ISSUES DETECTED'}")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    exit(main())
