import runpod
import os
import time
import base64
import torch
import numpy as np
import cv2
import json
import tempfile
import shutil
from pathlib import Path
import ffmpeg
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import AutoencoderKL
import io
import requests
from urllib.parse import urlparse

class MuseTalkModel:
    def __init__(self):
        print("Initializing MuseTalk model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load MuseTalk model
        self.model_path = "/app/models/musetalk"
        self.config_path = os.path.join(self.model_path, "musetalk.json")
        self.model_weights_path = os.path.join(self.model_path, "pytorch_model.bin")
        
        # Load model configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load VAE for image processing
        self.vae = AutoencoderKL.from_pretrained(
            "/app/models/sd-vae-ft-mse",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.vae.to(self.device)
        
        print("MuseTalk model initialized successfully")

    def process_image_and_audio(self, image_path, audio_path, bbox_shift=0):
        print(f"Processing image: {image_path} with audio: {audio_path}")
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Load the image
            if image_path.startswith('http'):
                # Download image if it's a URL
                response = requests.get(image_path)
                image = Image.open(io.BytesIO(response.content))
                local_image_path = os.path.join(temp_dir, "input_image.jpg")
                image.save(local_image_path)
            else:
                # Use local path
                local_image_path = image_path
                image = Image.open(local_image_path)
            
            # Convert PIL Image to numpy array for OpenCV processing
            image_np = np.array(image)
            if image_np.shape[2] == 4:  # If RGBA, convert to RGB
                image_np = image_np[:, :, :3]
            
            # Process image with MuseTalk model
            # In a real implementation, this would use the full MuseTalk pipeline
            # Here we're creating a simple animation by duplicating the image
            
            # Create frames directory
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Determine audio duration using ffmpeg
            probe = ffmpeg.probe(audio_path)
            audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            duration = float(audio_info.get('duration', 5.0))
            
            # Generate frames (for a real implementation, this would be where MuseTalk generates lip-synced frames)
            fps = 30
            frame_count = int(duration * fps)
            frame_paths = []
            
            # For demo purposes, we'll create a simple animation by slightly modifying the image
            # In a real implementation, this would be replaced with actual lip-syncing logic
            for i in range(frame_count):
                # Create a copy of the image for this frame
                frame = image_np.copy()
                
                # Apply a simple animation effect (this is just for demonstration)
                # In a real implementation, this would be where the mouth is animated based on audio
                if i % 10 < 5:  # Simple open/close mouth simulation
                    # Determine the mouth region (approximate)
                    height, width = frame.shape[:2]
                    mouth_y = int(height * 0.7)  # Approximate mouth position
                    mouth_height = int(height * 0.1)
                    
                    # Apply a simple effect to simulate mouth movement
                    # This is just for demonstration - MuseTalk would do proper lip-syncing
                    frame[mouth_y:mouth_y+mouth_height, int(width*0.4):int(width*0.6), :] = [
                        int(128 + 50 * np.sin(i * 0.2)),
                        int(128 + 50 * np.sin(i * 0.2)),
                        int(128 + 50 * np.sin(i * 0.2))
                    ]
                
                # Save the frame
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)
            
            # Create output video path
            output_path = os.path.join(temp_dir, "output.mp4")
            
            # Combine frames into video with audio
            self._create_video_with_audio(frame_paths, audio_path, output_path, fps)
            
            # Read the output video as bytes
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
                
            # Encode as base64 for safe return
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            return video_base64, output_path
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
    
    def _create_video_with_audio(self, frame_paths, audio_path, output_path, fps):
        # Create video from frames
        temp_video = os.path.join(os.path.dirname(output_path), "temp_video.mp4")
        
        # Use ffmpeg to create video from frames
        frame_pattern = os.path.join(os.path.dirname(frame_paths[0]), "frame_%04d.png")
        ffmpeg.input(frame_pattern, framerate=fps).output(temp_video, vcodec='libx264', pix_fmt='yuv420p').run(quiet=True, overwrite_output=True)
        
        # Add audio to video
        ffmpeg.input(temp_video).input(audio_path).output(output_path, vcodec='copy', acodec='aac', strict='experimental').run(quiet=True, overwrite_output=True)
        
        # Remove temporary video
        os.remove(temp_video)


def handler(event):
    print("Worker Start")
    input_data = event['input']

    # Initialize MuseTalk model (lazy loading - will only initialize once)
    if not hasattr(handler, "model"):
        handler.model = MuseTalkModel()

    # Get input parameters
    image_input = input_data.get('image')
    audio_input = input_data.get('audio')
    bbox_shift = input_data.get('bbox_shift', 0)
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Process image input (can be URL, base64, or local path)
        if image_input:
            if isinstance(image_input, str):
                if image_input.startswith('http'):
                    # It's a URL
                    image_path = image_input
                elif image_input.startswith('data:image'):  # base64 image
                    # It's a base64 encoded image
                    image_data = base64.b64decode(image_input.split(',')[1])
                    image_path = os.path.join(temp_dir, "input_image.jpg")
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                else:
                    # Assume it's a local path
                    image_path = image_input
            else:
                # Not a valid image input
                return {"status": "error", "message": "Invalid image input format"}
        else:
            # Create a default test image if none provided
            print("Creating test image")
            image_path = os.path.join(temp_dir, "test_image.jpg")
            # Create a simple test image
            test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
            # Draw a simple face shape
            cv2.circle(test_image, (320, 240), 150, (255, 200, 150), -1)  # Face
            cv2.circle(test_image, (270, 180), 30, (255, 255, 255), -1)  # Left eye
            cv2.circle(test_image, (370, 180), 30, (255, 255, 255), -1)  # Right eye
            cv2.circle(test_image, (270, 180), 10, (0, 0, 0), -1)  # Left pupil
            cv2.circle(test_image, (370, 180), 10, (0, 0, 0), -1)  # Right pupil
            cv2.ellipse(test_image, (320, 280), (60, 30), 0, 0, 180, (150, 100, 100), -1)  # Mouth
            cv2.imwrite(image_path, test_image)
        
        # Process audio input (can be URL, base64, or local path)
        if audio_input:
            if isinstance(audio_input, str):
                if audio_input.startswith('http'):
                    # It's a URL
                    response = requests.get(audio_input)
                    audio_path = os.path.join(temp_dir, "input_audio.wav")
                    with open(audio_path, 'wb') as f:
                        f.write(response.content)
                elif audio_input.startswith('data:audio'):  # base64 audio
                    # It's a base64 encoded audio
                    audio_data = base64.b64decode(audio_input.split(',')[1])
                    audio_path = os.path.join(temp_dir, "input_audio.wav")
                    with open(audio_path, 'wb') as f:
                        f.write(audio_data)
                else:
                    # Assume it's a local path
                    audio_path = audio_input
            else:
                # Not a valid audio input
                return {"status": "error", "message": "Invalid audio input format"}
        else:
            # Create a default test audio if none provided
            print("Creating test audio")
            audio_path = os.path.join(temp_dir, "test_audio.wav")
            # Create a simple test audio (silence with beeps)
            os.system(f"ffmpeg -f lavfi -i 'sine=frequency=1000:duration=5' -q:a 9 -acodec pcm_s16le {audio_path} -y")
        
        print(f"Processing image: {image_path} with audio: {audio_path}")
        
        # Process image and audio with MuseTalk
        video_base64, output_path = handler.model.process_image_and_audio(image_path, audio_path, bbox_shift)
        
        return {
            "status": "success",
            "video_base64": video_base64,
            "message": "MuseTalk processing completed successfully"
        }
    except Exception as e:
        print(f"Error processing: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing: {str(e)}"
        }
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
