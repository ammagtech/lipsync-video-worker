#!/usr/bin/env python3
"""
Utility script to encode image and audio files to base64 for testing
"""

import base64
import json
import argparse
from pathlib import Path

def encode_file_to_base64(file_path):
    """Encode a file to base64 string"""
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        return base64.b64encode(file_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {file_path}: {e}")
        return None

def create_test_input(image_path, audio_path, output_path="test_input_custom.json"):
    """Create a test input JSON file with encoded image and audio"""
    
    print(f"Encoding image: {image_path}")
    image_base64 = encode_file_to_base64(image_path)
    if not image_base64:
        return False
    
    print(f"Encoding audio: {audio_path}")
    audio_base64 = encode_file_to_base64(audio_path)
    if not audio_base64:
        return False
    
    # Create test input structure
    test_input = {
        "input": {
            "image": image_base64,
            "audio": audio_base64,
            "prompt": "The person is speaking enthusiastically with natural facial expressions",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "prompt_cfg_scale": 5.0,
            "audio_cfg_scale": 5.0,
            "num_frames": 81,
            "fps": 8
        }
    }
    
    # Save to file
    try:
        with open(output_path, 'w') as f:
            json.dump(test_input, f, indent=2)
        print(f"✓ Test input saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving test input: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Encode image and audio files for FantasyTalking testing")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--output", default="test_input_custom.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    # Create test input
    success = create_test_input(args.image, args.audio, args.output)
    
    if success:
        print("\n✓ Test input file created successfully!")
        print(f"You can now test with: python3 rp_handler.py")
        print(f"Make sure to rename {args.output} to test_input.json or modify the handler to use your custom file.")
    else:
        print("\n✗ Failed to create test input file.")

if __name__ == "__main__":
    main()
