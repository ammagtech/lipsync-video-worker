#!/usr/bin/env python3
"""
Helper script to create test_input.json with actual base64 encoded image and audio files.
This script helps you prepare the test input for the FantasyTalking RunPod worker.
"""

import base64
import json
import argparse
from pathlib import Path


def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 string."""
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
        return base64.b64encode(file_bytes).decode('utf-8')


def create_test_input(image_path: str, audio_path: str, output_path: str = "test_input.json"):
    """
    Create test_input.json with base64 encoded image and audio.
    
    Args:
        image_path: Path to the input image file
        audio_path: Path to the input audio file
        output_path: Path to save the test_input.json file
    """
    
    # Validate input files
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"Encoding image: {image_path}")
    image_base64 = encode_file_to_base64(image_path)
    
    print(f"Encoding audio: {audio_path}")
    audio_base64 = encode_file_to_base64(audio_path)
    
    # Create test input structure
    test_input = {
        "input": {
            "image": image_base64,
            "audio": audio_base64,
            "prompt": "A person is talking enthusiastically with expressive gestures",
            "image_size": 512,
            "max_num_frames": 81,
            "fps": 23,
            "prompt_cfg_scale": 5.0,
            "audio_cfg_scale": 5.0,
            "seed": 1111
        }
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(test_input, f, indent=2)
    
    print(f"Test input saved to: {output_path}")
    print(f"Image size: {len(image_base64)} characters")
    print(f"Audio size: {len(audio_base64)} characters")
    print("\nYou can now run: python3 rp_handler.py")


def main():
    parser = argparse.ArgumentParser(description="Create test input for FantasyTalking RunPod worker")
    parser.add_argument("--image", "-i", required=True, help="Path to input image file")
    parser.add_argument("--audio", "-a", required=True, help="Path to input audio file")
    parser.add_argument("--output", "-o", default="test_input.json", help="Output file path")
    parser.add_argument("--prompt", "-p", default="A person is talking enthusiastically with expressive gestures", 
                       help="Prompt for video generation")
    parser.add_argument("--image-size", type=int, default=512, help="Image size for processing")
    parser.add_argument("--max-frames", type=int, default=81, help="Maximum number of frames")
    parser.add_argument("--fps", type=int, default=23, help="Frames per second")
    parser.add_argument("--prompt-cfg", type=float, default=5.0, help="Prompt CFG scale")
    parser.add_argument("--audio-cfg", type=float, default=5.0, help="Audio CFG scale")
    parser.add_argument("--seed", type=int, default=1111, help="Random seed")
    
    args = parser.parse_args()
    
    try:
        # Validate input files
        if not Path(args.image).exists():
            raise FileNotFoundError(f"Image file not found: {args.image}")
        
        if not Path(args.audio).exists():
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        
        print(f"Encoding image: {args.image}")
        image_base64 = encode_file_to_base64(args.image)
        
        print(f"Encoding audio: {args.audio}")
        audio_base64 = encode_file_to_base64(args.audio)
        
        # Create test input structure with custom parameters
        test_input = {
            "input": {
                "image": image_base64,
                "audio": audio_base64,
                "prompt": args.prompt,
                "image_size": args.image_size,
                "max_num_frames": args.max_frames,
                "fps": args.fps,
                "prompt_cfg_scale": args.prompt_cfg,
                "audio_cfg_scale": args.audio_cfg,
                "seed": args.seed
            }
        }
        
        # Save to file
        with open(args.output, 'w') as f:
            json.dump(test_input, f, indent=2)
        
        print(f"\n✅ Test input saved to: {args.output}")
        print(f"📷 Image size: {len(image_base64)} characters")
        print(f"🎵 Audio size: {len(audio_base64)} characters")
        print(f"⚙️  Parameters:")
        print(f"   - Prompt: {args.prompt}")
        print(f"   - Image size: {args.image_size}")
        print(f"   - Max frames: {args.max_frames}")
        print(f"   - FPS: {args.fps}")
        print(f"   - Prompt CFG: {args.prompt_cfg}")
        print(f"   - Audio CFG: {args.audio_cfg}")
        print(f"   - Seed: {args.seed}")
        print(f"\n🚀 You can now run: python3 rp_handler.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
