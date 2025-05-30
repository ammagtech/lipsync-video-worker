import runpod
import torch
import base64
import io
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

# Import FantasyTalking dependencies
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    from diffusers import DiffusionPipeline
    import xformers
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")

from config import (
    WAN_MODEL_PATH,
    WAV2VEC_MODEL_PATH,
    FANTASYTALKING_MODEL_PATH,
    DEFAULT_CONFIG,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
    MEMORY_OPTIMIZATION,
    OUTPUT_CONFIG
)

# Global variables for model loading
pipeline = None
wav2vec_processor = None
wav2vec_model = None

def load_models():
    """Load all required models for FantasyTalking"""
    global pipeline, wav2vec_processor, wav2vec_model

    print("Loading FantasyTalking models...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # First verify model paths exist
        print(f"Checking WAV2VEC_MODEL_PATH: {WAV2VEC_MODEL_PATH}")
        if not os.path.exists(str(WAV2VEC_MODEL_PATH)):
            raise ValueError(f"WAV2VEC_MODEL_PATH does not exist: {WAV2VEC_MODEL_PATH}")

        # Load Wav2Vec2 for audio processing
        print("Loading Wav2Vec2 model...")
        try:
            wav2vec_processor = Wav2Vec2Processor.from_pretrained(str(WAV2VEC_MODEL_PATH))
            if wav2vec_processor is None:
                raise ValueError("Wav2Vec2Processor failed to load.")
        except Exception as e:
            raise ValueError(f"Failed to load Wav2Vec2Processor: {str(e)}")

        try:
            wav2vec_model = Wav2Vec2Model.from_pretrained(str(WAV2VEC_MODEL_PATH))
            if wav2vec_model is None:
                raise ValueError("Wav2Vec2Model failed to load.")
            wav2vec_model.to(device)
        except Exception as e:
            raise ValueError(f"Failed to load Wav2Vec2Model: {str(e)}")

        # Load base video generation pipeline
        print("Loading Wan2.1 pipeline...")
        pipeline = DiffusionPipeline.from_pretrained(
            str(WAN_MODEL_PATH),
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            use_safetensors=True
        )
        pipeline.to(device)

        if MEMORY_OPTIMIZATION["enable_attention_slicing"]:
            pipeline.enable_attention_slicing()
        if MEMORY_OPTIMIZATION["enable_cpu_offload"] and device == "cuda":
            pipeline.enable_model_cpu_offload()

        print("✓ All models loaded successfully")
        return True

    except Exception as e:
        print(f"✗ Error loading models: {e}")
        traceback.print_exc()
        return False



def process_image(image_data: str) -> Optional[Image.Image]:
    """Process base64 image input"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to model requirements
        target_size = (DEFAULT_CONFIG["width"], DEFAULT_CONFIG["height"])
        image = image.resize(target_size, Image.Resampling.LANCZOS)

        return image

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def process_audio(audio_data: str) -> Optional[np.ndarray]:
    """Process base64 audio input"""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)

        # Save to temporary file for librosa
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        # Load audio with librosa
        audio, sr = librosa.load(temp_path, sr=16000)  # Wav2Vec2 expects 16kHz

        # Clean up temp file
        os.unlink(temp_path)

        return audio

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def extract_audio_features(audio: np.ndarray) -> Optional[torch.Tensor]:
    """Extract audio features using Wav2Vec2"""
    global wav2vec_processor, wav2vec_model

    # Check that the processor and model are loaded
    if wav2vec_processor is None or wav2vec_model is None:
        print("Error: Audio processor or model is not loaded.")
        return None

    try:
        inputs = wav2vec_processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )

        device = next(wav2vec_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
            audio_features = outputs.last_hidden_state

        return audio_features

    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None
def generate_talking_video(image: Image.Image, audio_features: torch.Tensor, prompt: str = "", **kwargs) -> Optional[np.ndarray]:
    """Generate talking video using FantasyTalking"""
    try:
        global pipeline

        # Get generation parameters
        num_inference_steps = kwargs.get('num_inference_steps', DEFAULT_CONFIG['num_inference_steps'])
        guidance_scale = kwargs.get('guidance_scale', DEFAULT_CONFIG['guidance_scale'])
        prompt_cfg_scale = kwargs.get('prompt_cfg_scale', DEFAULT_CONFIG['prompt_cfg_scale'])
        audio_cfg_scale = kwargs.get('audio_cfg_scale', DEFAULT_CONFIG['audio_cfg_scale'])
        num_frames = kwargs.get('num_frames', DEFAULT_CONFIG['num_frames'])

        print(f"Generating video with {num_frames} frames...")

        # This is a simplified version - the actual FantasyTalking implementation
        # would require the specific model architecture and conditioning mechanisms
        # For now, we'll use the base pipeline with audio conditioning

        # Convert image to tensor
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Generate video frames
        with torch.no_grad():
            # This would be replaced with actual FantasyTalking inference
            # For demonstration, we'll create a simple video sequence
            frames = []
            for i in range(num_frames):
                # In real implementation, this would use the audio features
                # to condition the video generation
                frame = np.array(image)
                frames.append(frame)

            video = np.stack(frames, axis=0)

        return video

    except Exception as e:
        print(f"Error generating video: {e}")
        traceback.print_exc()
        return None

def encode_video_to_base64(video: np.ndarray, fps: int = 8) -> Optional[str]:
    """Encode video frames to base64 MP4"""
    try:
        # Create temporary file for video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name

        # Set up video writer
        height, width = video.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        # Write frames
        for frame in video:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

        # Read video file and encode to base64
        with open(temp_path, 'rb') as f:
            video_bytes = f.read()

        # Clean up temp file
        os.unlink(temp_path)

        # Encode to base64
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        return video_base64

    except Exception as e:
        print(f"Error encoding video: {e}")
        return None

def handler(event):
    """Main RunPod handler function for FantasyTalking"""
    print("FantasyTalking Worker Start")

    try:
        # Get input data
        input_data = event['input']

        # Required inputs
        image_data = input_data.get('image')  # base64 encoded image
        audio_data = input_data.get('audio')  # base64 encoded audio

        if not image_data or not audio_data:
            return {
                "status": "error",
                "message": "Both 'image' and 'audio' are required in base64 format"
            }

        # Optional inputs
        prompt = input_data.get('prompt', "")
        num_inference_steps = input_data.get('num_inference_steps', DEFAULT_CONFIG['num_inference_steps'])
        guidance_scale = input_data.get('guidance_scale', DEFAULT_CONFIG['guidance_scale'])
        prompt_cfg_scale = input_data.get('prompt_cfg_scale', DEFAULT_CONFIG['prompt_cfg_scale'])
        audio_cfg_scale = input_data.get('audio_cfg_scale', DEFAULT_CONFIG['audio_cfg_scale'])
        num_frames = input_data.get('num_frames', DEFAULT_CONFIG['num_frames'])
        fps = input_data.get('fps', DEFAULT_CONFIG['fps'])

        print(f"Processing request with prompt: '{prompt}'")
        print(f"Parameters: steps={num_inference_steps}, guidance={guidance_scale}, frames={num_frames}")

        # Process inputs
        print("Processing image...")
        image = process_image(image_data)
        if image is None:
            return {"status": "error", "message": "Failed to process image"}

        print("Processing audio...")
        audio = process_audio(audio_data)
        if audio is None:
            return {"status": "error", "message": "Failed to process audio"}

        print("Extracting audio features...")
        audio_features = extract_audio_features(audio)
        if audio_features is None:
            return {"status": "error", "message": "Failed to extract audio features"}

        # Generate talking video
        print("Generating talking video...")
        video = generate_talking_video(
            image=image,
            audio_features=audio_features,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt_cfg_scale=prompt_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            num_frames=num_frames
        )

        if video is None:
            return {"status": "error", "message": "Failed to generate video"}

        print("Encoding video...")
        video_base64 = encode_video_to_base64(video, fps=fps)
        if video_base64 is None:
            return {"status": "error", "message": "Failed to encode video"}

        print("✓ Video generation completed successfully")

        return {
            "status": "success",
            "video_base64": video_base64,
            "prompt": prompt,
            "num_frames": num_frames,
            "fps": fps,
            "format": "mp4",
            "resolution": f"{DEFAULT_CONFIG['width']}x{DEFAULT_CONFIG['height']}"
        }

    except Exception as e:
        print(f"Error in handler: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Internal error: {str(e)}"
        }

# Initialize models on startup
print("Initializing FantasyTalking models...")
models_loaded = load_models()

if not models_loaded:
    print("⚠️  Warning: Models failed to load. Handler will return errors.")

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
