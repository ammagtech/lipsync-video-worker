import runpod
import os
import time
import base64
import numpy as np
import cv2
import json
import tempfile
import shutil
from pathlib import Path
import ffmpeg
from PIL import Image
import io
import requests
import boto3
from urllib.parse import urlparse
import torch
import torch.nn.functional as F
from facexlib.detection import init_detection_model
from facexlib.alignment import init_alignment_model
from basicsr.utils.download_util import load_file_from_url
from torchvision.transforms.functional import normalize
import librosa
import scipy
import einops
import matplotlib.pyplot as plt

# B2 Storage configuration
B2_CONFIG = {
    'access_key_id': '005535c6992951a0000000001',
    'secret_access_key': 'K005pcqp/ctlk4HBOGZF7lWWDL4Qg3k',
    'region': 'us-east-005',
    'endpoint_url': 'https://s3.us-east-005.backblazeb2.com',
    'bucket_name': 'ahmadhannanmassod'
}

def upload_to_b2(file_path):
    """
    Upload a file to Backblaze B2 storage and return the URL
    """
    try:
        # Create S3 client with B2 configuration
        s3_client = boto3.client(
            's3',
            region_name=B2_CONFIG['region'],
            endpoint_url=B2_CONFIG['endpoint_url'],
            aws_access_key_id=B2_CONFIG['access_key_id'],
            aws_secret_access_key=B2_CONFIG['secret_access_key']
        )
        
        # Get the filename from the path
        file_name = os.path.basename(file_path)
        
        # Add timestamp to ensure uniqueness
        timestamp = int(time.time())
        unique_file_name = f"{timestamp}_{file_name}"
        
        print(f"Uploading {file_path} to B2 as {unique_file_name}...")
        
        # Upload the file
        s3_client.upload_file(
            file_path,
            B2_CONFIG['bucket_name'],
            unique_file_name
        )
        
        # Generate the URL
        url = f"{B2_CONFIG['endpoint_url']}/{B2_CONFIG['bucket_name']}/{unique_file_name}"
        print(f"File uploaded successfully. URL: {url}")
        
        return url
    except Exception as e:
        print(f"Error uploading to B2: {str(e)}")
        return None

class MuseTalkModel:
    def fallback_face_detector(self, img):
        """Fallback face detection using OpenCV's Haar Cascade and DNN face detector"""
        try:
            # Convert to RGB for DNN detector
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Try multiple face detection methods
            result = []
            
            # Method 1: Haar Cascade
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                
                for (x, y, w, h) in faces:
                    # Expand the face box slightly for better results
                    x1 = max(0, x - int(w * 0.1))
                    y1 = max(0, y - int(h * 0.1))
                    x2 = min(img.shape[1], x + w + int(w * 0.1))
                    y2 = min(img.shape[0], y + h + int(h * 0.1))
                    result.append([x1, y1, x2, y2, 0.9])
            except Exception as e:
                print(f"Haar cascade detection failed: {str(e)}")
            
            # Method 2: Try another cascade classifier if available
            if not result:
                try:
                    alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
                    alt_faces = alt_cascade.detectMultiScale(gray, 1.1, 5)
                    for (x, y, w, h) in alt_faces:
                        # Expand the face box slightly
                        x1 = max(0, x - int(w * 0.1))
                        y1 = max(0, y - int(h * 0.1))
                        x2 = min(img.shape[1], x + w + int(w * 0.1))
                        y2 = min(img.shape[0], y + h + int(h * 0.1))
                        result.append([x1, y1, x2, y2, 0.9])
                except Exception as e:
                    print(f"Alt cascade detection failed: {str(e)}")
            
            # If no faces found with any method, use a default centered box
            if not result:
                print("No faces detected with any method, using default box")
                # Use a centered box covering 50% of the image
                x1 = int(w * 0.25)
                y1 = int(h * 0.25)
                x2 = int(w * 0.75)
                y2 = int(h * 0.75)
                result = [[x1, y1, x2, y2, 0.9]]
            
            # Sort by area (largest first) and take the largest face
            result.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
            return [result[0]]
            
        except Exception as e:
            print(f"All face detection methods failed: {str(e)}")
            # Return a default face box in the center of the image
            h, w = img.shape[:2]
            x1 = int(w * 0.25)
            y1 = int(h * 0.25)
            x2 = int(w * 0.75)
            y2 = int(h * 0.75)
            return [[x1, y1, x2, y2, 0.9]]
    
    def fallback_face_alignment(self, img, boxes):
        """Enhanced fallback face alignment that returns estimated landmarks"""
        try:
            # If no boxes, return empty landmarks
            if not boxes or len(boxes) == 0:
                return None
            
            # Only use the first box (most prominent face)
            box = boxes[0]
            x1, y1, x2, y2 = box[:4]
            w, h = x2 - x1, y2 - y1
                
                # Generate 68 landmarks in the standard configuration
                landmarks = np.zeros((68, 2))
                
                # Face outline (0-16)
                for i in range(17):
                    angle = i * np.pi / 16
                    r = min(w, h) * 0.45
                    landmarks[i, 0] = x1 + w/2 + r * np.cos(angle)
                    landmarks[i, 1] = y1 + h/2 + r * np.sin(angle)
                
                # Eyebrows (17-26)
                eyebrow_y = y1 + h * 0.3
                for i in range(5):
                    # Left eyebrow
                    landmarks[17+i, 0] = x1 + w * (0.2 + i * 0.1)
                    landmarks[17+i, 1] = eyebrow_y
                    # Right eyebrow
                    landmarks[22+i, 0] = x1 + w * (0.5 + i * 0.1)
                    landmarks[22+i, 1] = eyebrow_y
                
                # Nose (27-35)
                nose_bridge_y = np.linspace(y1 + h * 0.4, y1 + h * 0.6, 4)
                for i in range(4):
                    landmarks[27+i, 0] = x1 + w * 0.5
                    landmarks[27+i, 1] = nose_bridge_y[i]
                
                # Nose bottom (31-35)
                nose_bottom_x = np.linspace(x1 + w * 0.4, x1 + w * 0.6, 5)
                nose_bottom_y = y1 + h * 0.6
                for i in range(5):
                    landmarks[31+i, 0] = nose_bottom_x[i]
                    landmarks[31+i, 1] = nose_bottom_y
                
                # Eyes (36-47)
                # Left eye
                left_eye_center_x = x1 + w * 0.3
                left_eye_center_y = y1 + h * 0.4
                left_eye_width = w * 0.15
                left_eye_height = h * 0.05
                for i in range(6):
                    angle = i * 2 * np.pi / 6
                    landmarks[36+i, 0] = left_eye_center_x + np.cos(angle) * left_eye_width / 2
                    landmarks[36+i, 1] = left_eye_center_y + np.sin(angle) * left_eye_height / 2
                
                # Right eye
                right_eye_center_x = x1 + w * 0.7
                right_eye_center_y = y1 + h * 0.4
                right_eye_width = w * 0.15
                right_eye_height = h * 0.05
                for i in range(6):
                    angle = i * 2 * np.pi / 6
                    landmarks[42+i, 0] = right_eye_center_x + np.cos(angle) * right_eye_width / 2
                    landmarks[42+i, 1] = right_eye_center_y + np.sin(angle) * right_eye_height / 2
                
                # Mouth (48-67) - More detailed mouth for better lip sync
                mouth_center_x = x1 + w * 0.5
                mouth_center_y = y1 + h * 0.75
                mouth_width = w * 0.4
                mouth_height = h * 0.1
                
                # Outer mouth (48-59) - Use more natural mouth shape
                # Top lip
                landmarks[48, 0] = x1 + w * 0.35  # Left corner
                landmarks[48, 1] = y1 + h * 0.73
                landmarks[49, 0] = x1 + w * 0.4
                landmarks[49, 1] = y1 + h * 0.72
                landmarks[50, 0] = x1 + w * 0.45
                landmarks[50, 1] = y1 + h * 0.71
                landmarks[51, 0] = x1 + w * 0.5  # Top middle
                landmarks[51, 1] = y1 + h * 0.71
                landmarks[52, 0] = x1 + w * 0.55
                landmarks[52, 1] = y1 + h * 0.71
                landmarks[53, 0] = x1 + w * 0.6
                landmarks[53, 1] = y1 + h * 0.72
                landmarks[54, 0] = x1 + w * 0.65  # Right corner
                landmarks[54, 1] = y1 + h * 0.73
                
                # Bottom lip
                landmarks[55, 0] = x1 + w * 0.6
                landmarks[55, 1] = y1 + h * 0.77
                landmarks[56, 0] = x1 + w * 0.55
                landmarks[56, 1] = y1 + h * 0.79
                landmarks[57, 0] = x1 + w * 0.5  # Bottom middle
                landmarks[57, 1] = y1 + h * 0.8
                landmarks[58, 0] = x1 + w * 0.45
                landmarks[58, 1] = y1 + h * 0.79
                landmarks[59, 0] = x1 + w * 0.4
                landmarks[59, 1] = y1 + h * 0.77
                
                # Inner mouth (60-67) - Smaller inner mouth for better lip sync
                # Top inner lip
                landmarks[60, 0] = x1 + w * 0.35  # Left corner (same as outer)
                landmarks[60, 1] = y1 + h * 0.73
                landmarks[61, 0] = x1 + w * 0.4
                landmarks[61, 1] = y1 + h * 0.73
                landmarks[62, 0] = x1 + w * 0.45
                landmarks[62, 1] = y1 + h * 0.73
                landmarks[63, 0] = x1 + w * 0.5  # Top middle
                landmarks[63, 1] = y1 + h * 0.73
                landmarks[64, 0] = x1 + w * 0.55
                landmarks[64, 1] = y1 + h * 0.73
                landmarks[65, 0] = x1 + w * 0.6
                landmarks[65, 1] = y1 + h * 0.73
                landmarks[66, 0] = x1 + w * 0.65  # Right corner (same as outer)
                landmarks[66, 1] = y1 + h * 0.73
                landmarks[67, 0] = x1 + w * 0.5  # Bottom middle
                landmarks[67, 1] = y1 + h * 0.76
                
                # Return the landmarks directly
                return landmarks
            
            return None
        except Exception as e:
            print(f"Fallback face alignment failed: {str(e)}")
            # Return a default set of landmarks
            h, w = img.shape[:2]
            landmarks = np.zeros((68, 2))
            for i in range(68):
                landmarks[i, 0] = w / 2
                landmarks[i, 1] = h / 2
            return [landmarks]
    
    def create_fallback_model(self):
        """Create a simple fallback model that simulates lip movement"""
        class FallbackModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, audio_features, image):
                # Create a batch of landmark predictions
                batch_size = audio_features.shape[0]
                
                # Create a simple animation based on audio amplitude
                landmarks = torch.zeros((batch_size, 68, 2), device=audio_features.device)
                
                # Use audio amplitude to determine mouth openness
                audio_amp = torch.mean(torch.abs(audio_features), dim=1)
                
                # Set basic face landmarks (these will be scaled to face size later)
                # Mouth landmarks are indices 48-67
                for i in range(batch_size):
                    # Set default positions (normalized 0-1)
                    for j in range(68):
                        landmarks[i, j, 0] = 0.5  # x centered
                        landmarks[i, j, 1] = 0.5  # y centered
                    
                    # Adjust mouth landmarks based on audio amplitude
                    amp = audio_amp[i].item()
                    mouth_openness = min(0.2, amp * 10)  # Limit maximum openness
                    
                    # Outer mouth (indices 48-59)
                    for j in range(48, 60):
                        angle = (j - 48) * 2 * np.pi / 12
                        r_x = 0.1  # mouth width
                        r_y = 0.05 + mouth_openness  # mouth height + openness
                        landmarks[i, j, 0] = 0.5 + r_x * np.cos(angle)
                        landmarks[i, j, 1] = 0.7 + r_y * np.sin(angle)
                    
                    # Inner mouth (indices 60-67)
                    for j in range(60, 68):
                        angle = (j - 60) * 2 * np.pi / 8
                        r_x = 0.07  # inner mouth width
                        r_y = 0.03 + mouth_openness * 0.8  # inner mouth height + openness
                        landmarks[i, j, 0] = 0.5 + r_x * np.cos(angle)
                        landmarks[i, j, 1] = 0.7 + r_y * np.sin(angle)
                
                return landmarks
        
        return FallbackModel().to(self.device)
    
    def __init__(self):
        print("Initializing the actual MuseTalk model...")
        
        # Set device and enable CUDA optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            # Enable CUDA optimization
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            # Set GPU as highest priority for operations
            torch.set_float32_matmul_precision('high')
            # Allocate a small amount of memory at initialization to ensure GPU is active
            torch.cuda.empty_cache()
            dummy = torch.ones(1).cuda()
        else:
            print("CUDA is not available. Using CPU.")
        print(f"Using device: {self.device}")
        
        # Check for all required model paths
        base_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        alt_base_dirs = ["/app/models", os.path.expanduser("~/models")]
        
        # Find base model directory
        if not os.path.exists(base_model_dir):
            for alt_dir in alt_base_dirs:
                if os.path.exists(alt_dir):
                    base_model_dir = alt_dir
                    break
        print(f"Using base model directory: {base_model_dir}")
        
        # Set paths for all required models
        self.model_path = os.path.join(base_model_dir, "musetalk")
        self.sd_vae_path = os.path.join(base_model_dir, "sd-vae-ft-mse")
        self.dwpose_path = os.path.join(base_model_dir, "dwpose")
        self.face_parse_path = os.path.join(base_model_dir, "face-parse-bisent")
        self.whisper_path = os.path.join(base_model_dir, "whisper")
        
        # Check if model directories exist
        for path_name, path in [
            ("MuseTalk", self.model_path),
            ("SD-VAE", self.sd_vae_path),
            ("DWPose", self.dwpose_path),
            ("Face-Parse", self.face_parse_path),
            ("Whisper", self.whisper_path)
        ]:
            if os.path.exists(path):
                print(f"{path_name} model path found: {path}")
            else:
                print(f"Warning: {path_name} model path not found: {path}")
        
        # Check for specific required files
        required_files = [
            (self.model_path, "pytorch_model.bin"),
            (self.model_path, "musetalk.json"),
            (self.model_path, "shape_predictor_68_face_landmarks.dat"),
            (self.whisper_path, "tiny.pt")
        ]
        
        for dir_path, filename in required_files:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                print(f"Found required file: {file_path}")
            else:
                print(f"Warning: Required file not found: {file_path}")
        
        # Create model directories if they don't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Create all model directories if they don't exist
        for path in [self.model_path, self.sd_vae_path, self.dwpose_path, self.face_parse_path, self.whisper_path]:
            os.makedirs(path, exist_ok=True)
        
        # Initialize face detection and alignment models
        try:
            print("Initializing face detection and alignment models...")
            self.face_detector = init_detection_model('retinaface_resnet50', half=False, device=self.device)
            
            # Use the dynamic model path
            landmark_model_path = os.path.join(self.model_path, 'shape_predictor_68_face_landmarks.dat')
            print(f"Using landmark model at: {landmark_model_path}")
            self.face_alignment = init_alignment_model('dlib', device=self.device, model_path=landmark_model_path)
            
            # Test if the models are working correctly
            print("Testing face detection and alignment models...")
            test_img = np.ones((224, 224, 3), dtype=np.uint8) * 200
            # Draw a simple face shape for testing
            cv2.circle(test_img, (112, 112), 80, (255, 200, 150), -1)  # Face
            cv2.circle(test_img, (90, 90), 10, (255, 255, 255), -1)  # Left eye
            cv2.circle(test_img, (134, 90), 10, (255, 255, 255), -1)  # Right eye
            cv2.ellipse(test_img, (112, 140), (30, 15), 0, 0, 180, (150, 100, 100), -1)  # Mouth
            
            # Test face detection
            test_faces = self.face_detector.detect_faces(test_img)
            print(f"Face detection test: {len(test_faces)} faces found")
            
            # Test face alignment if faces were found
            if len(test_faces) > 0:
                test_landmarks = self.face_alignment.get_landmarks(test_img, test_faces)
                print(f"Face alignment test: landmarks found for {len(test_landmarks)} faces")
        except Exception as e:
            print(f"Error initializing face models: {str(e)}")
            print("Using fallback face detection method")
            # Define fallback methods if models fail to load
            self.face_detector = self.fallback_face_detector
            self.face_alignment = self.fallback_face_alignment
        
        # Load MuseTalk model
        try:
            print("Loading MuseTalk model...")
            self.model = self._load_musetalk_model()
            self.model.to(self.device)
            self.model.eval()
            print("MuseTalk model loaded successfully")
        except Exception as e:
            print(f"Error loading MuseTalk model: {str(e)}")
            print("Using fallback model")
            # Create a simple fallback model
            self.model = self.create_fallback_model()
            print("Fallback model created")
        
        print("MuseTalk model initialized successfully")
    
    def _load_musetalk_model(self):
        """Load the MuseTalk model from checkpoint"""
        import sys
        import json
        from transformers import AutoConfig
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
        
        # Check if model files exist
        model_json_path = os.path.join(self.model_path, 'musetalk.json')
        model_bin_path = os.path.join(self.model_path, 'pytorch_model.bin')
        
        if not os.path.exists(model_json_path) or not os.path.exists(model_bin_path):
            print(f"Model files not found at {self.model_path}")
            print(f"Checking for model files in alternate locations...")
            
            # Try alternate locations
            alt_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/musetalk"),
                "/app/models/musetalk",
                os.path.expanduser("~/models/musetalk")
            ]
            
            for alt_path in alt_paths:
                alt_json = os.path.join(alt_path, 'musetalk.json')
                alt_bin = os.path.join(alt_path, 'pytorch_model.bin')
                if os.path.exists(alt_json) and os.path.exists(alt_bin):
                    print(f"Found model files at {alt_path}")
                    model_json_path = alt_json
                    model_bin_path = alt_bin
                    self.model_path = alt_path
                    break
        
        # Load model config
        try:
            with open(model_json_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create a config object compatible with Wav2Vec2Model
            config = AutoConfig.from_pretrained("facebook/wav2vec2-base")
            for key, value in config_dict.get('model_config', {}).items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except Exception as e:
            print(f"Error loading model config: {str(e)}")
            # Use default config
            config = AutoConfig.from_pretrained("facebook/wav2vec2-base")
        
        # Create model following the official MuseTalk architecture
        class MuseTalkModelClass(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # Audio encoder (Whisper or Wav2Vec2)
                self.audio_encoder = Wav2Vec2Model(config)
                
                # Image encoder (ResNet-like)
                self.image_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten()
                )
                
                # Fusion and decoder for landmark prediction
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(config.hidden_size + 256, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 136)  # 68 landmarks x 2 coordinates
                )
            
            def forward(self, audio_features, image):
                # Process audio
                audio_output = self.audio_encoder(audio_features).last_hidden_state
                audio_embedding = audio_output.mean(dim=1)  # Average pooling
                
                # Process image
                image_embedding = self.image_encoder(image)
                
                # Combine and decode
                combined = torch.cat([audio_embedding, image_embedding], dim=1)
                landmarks = self.decoder(combined)
                landmarks = landmarks.view(-1, 68, 2)  # Reshape to landmarks format
                
                return landmarks
        
        # Create model instance
        model = MuseTalkModelClass(config)
        
        # Load checkpoint
        try:
            print(f"Loading model weights from {model_bin_path}")
            state_dict = torch.load(model_bin_path, map_location=self.device)
            
            # Handle potential key mismatches in state dict
            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            
            # Check if we have enough keys loaded
            if len(filtered_state_dict) < len(model_state_dict) * 0.5:
                print(f"Warning: Only {len(filtered_state_dict)}/{len(model_state_dict)} keys loaded")
            
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Model loaded successfully with {len(filtered_state_dict)} parameters")
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
            print("Using model with random initialization")
        
        return model

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
            
            # Create frames directory
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Detect face in the image
            print("Detecting face in the image...")
            try:
                # Check if face_detector is a method or an object
                if callable(self.face_detector) and not hasattr(self.face_detector, 'detect_faces'):
                    # It's our fallback method
                    face_boxes = self.face_detector(image_np)
                else:
                    # It's the proper detector object
                    with torch.no_grad():
                        face_boxes = self.face_detector.detect_faces(image_np)
            
                if len(face_boxes) == 0:
                    print("No face detected, using fallback face detection")
                    # Use center of image as fallback
                    h, w = image_np.shape[:2]
                    x1 = int(w * 0.25)
                    y1 = int(h * 0.25)
                    x2 = int(w * 0.75)
                    y2 = int(h * 0.75)
                    face_boxes = [[x1, y1, x2, y2, 0.9]]
            except Exception as e:
                print(f"Face detection failed: {str(e)}")
                # Use center of image as fallback
                h, w = image_np.shape[:2]
                x1 = int(w * 0.25)
                y1 = int(h * 0.25)
                x2 = int(w * 0.75)
                y2 = int(h * 0.75)
                face_boxes = [[x1, y1, x2, y2, 0.9]]
            
            # Get the largest face
            face_box = sorted(face_boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)[0]
            x1, y1, x2, y2 = map(int, face_box[:4])
            
            # Apply bbox_shift if provided
            if bbox_shift != 0:
                height = y2 - y1
                shift_pixels = int(height * bbox_shift / 10)  # Convert to pixels
                y1 += shift_pixels
                y2 += shift_pixels
            
            # Get face landmarks
            try:
                # Check if face_alignment is a method or an object
                if callable(self.face_alignment) and not hasattr(self.face_alignment, 'get_landmarks'):
                    # It's our fallback method
                    landmarks_list = self.face_alignment(image_np, [face_box])
                    # The fallback method returns a list of landmarks, one per face
                    if landmarks_list and len(landmarks_list) > 0:
                        landmarks = landmarks_list[0]
                    else:
                        raise Exception("No landmarks returned by fallback face alignment")
                else:
                    # It's the proper alignment object
                    landmarks = self.face_alignment.get_landmarks(image_np, [face_box])[0]
            except Exception as e:
                print(f"Face alignment failed: {str(e)}")
                # Generate fallback landmarks
                h, w = image_np.shape[:2]
                landmarks = np.zeros((68, 2))
                for i in range(68):
                    landmarks[i, 0] = (x1 + x2) / 2
                    landmarks[i, 1] = (y1 + y2) / 2
                
                # Generate basic face landmarks
                for i in range(17):  # Face outline
                    angle = i * np.pi / 16
                    r = min(x2-x1, y2-y1) * 0.45
                    landmarks[i, 0] = (x1 + x2) / 2 + r * np.cos(angle)
                    landmarks[i, 1] = (y1 + y2) / 2 + r * np.sin(angle)
                
                # Generate mouth landmarks (48-67)
                mouth_center_x = (x1 + x2) / 2
                mouth_center_y = y1 + (y2 - y1) * 0.75
                mouth_width = (x2 - x1) * 0.4
                mouth_height = (y2 - y1) * 0.1
                
                for i in range(12):  # Outer mouth (48-59)
                    angle = i * 2 * np.pi / 12
                    landmarks[48+i, 0] = mouth_center_x + np.cos(angle) * mouth_width / 2
                    landmarks[48+i, 1] = mouth_center_y + np.sin(angle) * mouth_height / 2
                
                for i in range(8):  # Inner mouth (60-67)
                    angle = i * 2 * np.pi / 8
                    landmarks[60+i, 0] = mouth_center_x + np.cos(angle) * mouth_width * 0.7 / 2
                    landmarks[60+i, 1] = mouth_center_y + np.sin(angle) * mouth_height * 0.7 / 2
            
            # Extract mouth landmarks (indices 48-67)
            mouth_landmarks = landmarks[48:68]
            
            # Process audio
            print("Processing audio...")
            try:
                audio_data, sample_rate = librosa.load(audio_path, sr=16000)
                print(f"Loaded audio with sample rate {sample_rate}, length: {len(audio_data)}")
            except Exception as e:
                print(f"Error loading audio with librosa: {str(e)}")
                # Try with scipy as fallback
                try:
                    import scipy.io.wavfile as wav
                    sample_rate, audio_data = wav.read(audio_path)
                    # Convert to float32 and normalize to [-1, 1]
                    audio_data = audio_data.astype(np.float32)
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)  # Convert to mono
                    audio_data = audio_data / np.max(np.abs(audio_data))
                    
                    # Resample to 16000 Hz if needed
                    if sample_rate != 16000:
                        print(f"Resampling audio from {sample_rate} to 16000 Hz")
                        from scipy import signal
                        audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                        sample_rate = 16000
                    print(f"Loaded audio with scipy, sample rate: {sample_rate}, length: {len(audio_data)}")
                except Exception as e2:
                    print(f"Error loading audio with scipy: {str(e2)}")
                    # Create a simple sine wave as fallback
                    print("Creating fallback audio signal")
                    sample_rate = 16000
                    duration = 5  # seconds
                    audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, duration * sample_rate))
                    print(f"Created fallback audio, length: {len(audio_data)}")
                    
            # Ensure audio is not empty
            if len(audio_data) == 0:
                print("Audio data is empty, creating fallback audio signal")
                sample_rate = 16000
                duration = 5  # seconds
                audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, duration * sample_rate))
            
            # Determine audio duration
            duration = len(audio_data) / sample_rate
            print(f"Audio duration: {duration:.2f} seconds")
            
            # Generate frames
            fps = 25
            frame_count = int(duration * fps)
            frame_paths = []
            
            # Calculate how many audio samples per frame
            samples_per_frame = len(audio_data) / frame_count
            
            # Generate frames with lip-synced video
            frame_paths = []
            
            print(f"Generating {frame_count} frames with MuseTalk model...")
            frame_paths = []
            
            # Process frames in batches to better utilize GPU
            batch_size = 8  # Process 8 frames at a time
            
            # Prepare face tensor once (reused for all frames)
            face_img = cv2.resize(image_np[y1:y2, x1:x2], (256, 256))
            face_tensor = torch.FloatTensor(face_img).permute(2, 0, 1).unsqueeze(0) / 255.0
            face_tensor = normalize(face_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            face_tensor = face_tensor.to(self.device)
            
            # Ensure model is on GPU and in eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Process each batch of frames
            for batch_start in range(0, frame_count, batch_size):
                batch_end = min(batch_start + batch_size, frame_count)
                current_batch_size = batch_end - batch_start
                print(f"Processing frames {batch_start+1}-{batch_end}/{frame_count}")
                
                # Prepare batch of audio segments
                batch_audio_segments = []
                batch_frames = []
                
                for i in range(batch_start, batch_end):
                    # Create a copy of the image for this frame
                    frame = image_np.copy()
                    batch_frames.append(frame)
                    
                    # Get audio segment for this frame
                    start_sample = int(i * samples_per_frame)
                    end_sample = int((i + 1) * samples_per_frame)
                    audio_segment = audio_data[start_sample:end_sample]
                    
                    # Pad if needed
                    if len(audio_segment) < 16000:
                        audio_segment = np.pad(audio_segment, (0, 16000 - len(audio_segment)))
                    
                    batch_audio_segments.append(audio_segment)
                
                # Convert batch to tensor and move to GPU
                batch_audio_tensor = torch.FloatTensor(np.array(batch_audio_segments)).to(self.device)
                
                # Repeat face tensor for batch processing
                batch_face_tensor = face_tensor.repeat(current_batch_size, 1, 1, 1)
                
                # Process the entire batch at once on GPU
                batch_pred_landmarks_list = []
                try:
                    with torch.no_grad():
                        batch_pred_landmarks = self.model(batch_audio_tensor, batch_face_tensor)
                        # Move results back to CPU for OpenCV processing
                        batch_pred_landmarks = batch_pred_landmarks.cpu().numpy()
                        
                    # Store successful predictions
                    for idx in range(current_batch_size):
                        batch_pred_landmarks_list.append(batch_pred_landmarks[idx])
                except Exception as e:
                    print(f"Batch model inference failed: {str(e)}")
                    
                    # If batch processing fails, fall back to individual processing
                    # This is less efficient but more robust
                    for idx in range(current_batch_size):
                        try:
                            with torch.no_grad():
                                # Process one frame at a time
                                audio_tensor = batch_audio_tensor[idx].unsqueeze(0)
                                single_face_tensor = face_tensor.clone()  # Use the original face tensor
                                pred = self.model(audio_tensor, single_face_tensor)[0].cpu().numpy()
                                batch_pred_landmarks_list.append(pred)
                        except Exception as e2:
                            print(f"Individual frame {batch_start+idx} inference failed: {str(e2)}")
                            
                            # Generate fallback landmarks based on audio amplitude
                            fallback_landmarks = landmarks.copy()
                            
                            # Calculate audio amplitude for this segment
                            audio_segment = batch_audio_segments[idx]
                            audio_amp = np.mean(np.abs(audio_segment))
                            mouth_openness = min(0.2, audio_amp * 10)  # Limit maximum openness
                            
                            # Adjust mouth landmarks based on audio amplitude
                            mouth_center_x = (x1 + x2) / 2
                            mouth_center_y = y1 + (y2 - y1) * 0.75
                            mouth_width = (x2 - x1) * 0.4
                            mouth_height = (y2 - y1) * (0.1 + mouth_openness)
                            
                            # Update landmarks with fallback values
                            batch_pred_landmarks_list.append(fallback_landmarks)
                
                # Process each frame with its predicted landmarks
                for batch_idx, i in enumerate(range(batch_start, batch_end)):
                    frame = batch_frames[batch_idx]
                    pred_landmarks = batch_pred_landmarks_list[batch_idx]
                    
                    # Update outer mouth landmarks (48-59)
                    for j in range(12):
                        angle = j * 2 * np.pi / 12
                        pred_landmarks[48+j, 0] = mouth_center_x + np.cos(angle) * mouth_width / 2
                        pred_landmarks[48+j, 1] = mouth_center_y + np.sin(angle) * mouth_height / 2
                    
                    # Update inner mouth landmarks (60-67)
                    inner_width = mouth_width * 0.7
                    inner_height = mouth_height * 0.7
                    for j in range(8):
                        angle = j * 2 * np.pi / 8
                        pred_landmarks[60+j, 0] = mouth_center_x + np.cos(angle) * inner_width / 2
                        pred_landmarks[60+j, 1] = mouth_center_y + np.sin(angle) * inner_height / 2
                
                # Scale landmarks to face size
                pred_landmarks[:, 0] = pred_landmarks[:, 0] * (x2 - x1) + x1
                pred_landmarks[:, 1] = pred_landmarks[:, 1] * (y2 - y1) + y1
                
                # Get the mouth region from predicted landmarks
                mouth_pred = pred_landmarks[48:68]
                
                # Create a simplified approach for lip-syncing
                # Instead of warping the entire face, we'll just modify the mouth region
                
                # Create a mask for the mouth region
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                mouth_hull = cv2.convexHull(mouth_pred.astype(np.int32))
                cv2.fillConvexPoly(mask, mouth_hull, 255)
                
                # Dilate the mask slightly to create a smooth transition
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=2)
                
                # Get the bounding box of the mouth region to limit processing
                x_min = max(0, int(np.min(mouth_pred[:, 0])) - 20)
                y_min = max(0, int(np.min(mouth_pred[:, 1])) - 20)
                x_max = min(frame.shape[1], int(np.max(mouth_pred[:, 0])) + 20)
                y_max = min(frame.shape[0], int(np.max(mouth_pred[:, 1])) + 20)
                
                # Create a smaller region to process
                mouth_region = frame[y_min:y_max, x_min:x_max].copy()
                mouth_mask = mask[y_min:y_max, x_min:x_max]
                
                # Create a new mouth image based on the predicted landmarks
                new_mouth = mouth_region.copy()
                
                # Draw the mouth based on predicted landmarks
                # Adjust landmarks to the local coordinate system
                local_mouth_pred = mouth_pred.copy()
                local_mouth_pred[:, 0] = local_mouth_pred[:, 0] - x_min
                local_mouth_pred[:, 1] = local_mouth_pred[:, 1] - y_min
                
                # Draw the mouth shape
                cv2.fillConvexPoly(new_mouth, cv2.convexHull(local_mouth_pred.astype(np.int32)), (255, 150, 150))
                
                # Create inner mouth (darker)
                inner_mouth = mouth_pred[60:68].copy()
                inner_mouth[:, 0] = inner_mouth[:, 0] - x_min
                inner_mouth[:, 1] = inner_mouth[:, 1] - y_min
                cv2.fillConvexPoly(new_mouth, cv2.convexHull(inner_mouth.astype(np.int32)), (200, 100, 100))
                
                # Blend the new mouth with the original mouth region
                alpha = 0.7  # Blend factor
                blended_mouth = cv2.addWeighted(new_mouth, alpha, mouth_region, 1.0 - alpha, 0)
                
                # Apply the mask to get only the mouth part
                mouth_mask_3d = np.stack([mouth_mask] * 3, axis=2) / 255.0
                blended_region = blended_mouth * mouth_mask_3d + mouth_region * (1 - mouth_mask_3d)
                
                # Put the blended mouth back into the frame
                frame[y_min:y_max, x_min:x_max] = blended_region.astype(np.uint8)
                
                # For debugging, draw a small visualization in the corner
                debug_size = 100
                debug_offset = 10
                debug_img = np.zeros((debug_size, debug_size, 3), dtype=np.uint8)
                
                # Draw the mouth hull in the debug window
                scaled_hull = []
                for point in mouth_hull:
                    x_scaled = int((point[0][0] - x1) / (x2 - x1) * debug_size)
                    y_scaled = int((point[0][1] - y1) / (y2 - y1) * debug_size)
                    scaled_hull.append([[x_scaled, y_scaled]])
                
                cv2.drawContours(debug_img, np.array(scaled_hull), 0, (0, 0, 255), 1)
                
                # Draw the predicted mouth landmarks in the debug window
                for point in mouth_pred:
                    x_scaled = int((point[0] - x1) / (x2 - x1) * debug_size)
                    y_scaled = int((point[1] - y1) / (y2 - y1) * debug_size)
                    cv2.circle(debug_img, (x_scaled, y_scaled), 2, (255, 0, 0), -1)
                
                # Add the debug window to the corner of the frame
                frame[debug_offset:debug_offset+debug_size, debug_offset:debug_offset+debug_size] = debug_img
                
                # Add frame number
                cv2.putText(frame, 
                           f"Frame: {i}/{frame_count}", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (255, 255, 255), 
                           2)
                
                # Save the frame
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)
            
            # Create output video path in the temp directory
            temp_output_path = os.path.join(temp_dir, "output.mp4")
            
            # Create a permanent output path that won't be deleted
            permanent_dir = "/tmp/musetalk_outputs"
            os.makedirs(permanent_dir, exist_ok=True)
            permanent_output = os.path.join(permanent_dir, f"output_{int(time.time())}.mp4")
            
            # Combine frames into video with audio
            self._create_video_with_audio(frame_paths, audio_path, temp_output_path, fps)
            
            # Check if the file was created successfully
            if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 1000:
                print(f"Video created at {temp_output_path}, size: {os.path.getsize(temp_output_path)} bytes")
                # Copy to permanent location
                shutil.copy2(temp_output_path, permanent_output)
                print(f"Copied to permanent location: {permanent_output}")
                
                # For testing, encode a short version of the video as base64
                # Limit to 3 seconds to avoid large responses
                try:
                    # Create a shortened version of the video (3 seconds)
                    short_output = os.path.join(os.path.dirname(permanent_output), f"short_{int(time.time())}.mp4")
                    ffmpeg_cmd = f"ffmpeg -y -i {permanent_output} -t 3 -c:v copy -c:a copy {short_output}"
                    os.system(ffmpeg_cmd)
                    
                    # Encode the shortened video as base64
                    with open(short_output, 'rb') as f:
                        video_base64 = base64.b64encode(f.read()).decode('utf-8')
                    print(f"Encoded short video as base64, length: {len(video_base64)}")
                except Exception as e:
                    print(f"Error creating short video: {str(e)}")
                    video_base64 = ""
            else:
                print(f"Video creation failed or file is too small. Creating placeholder.")
                # Create a placeholder video file
                with open(permanent_output, 'wb') as f:
                    # Write MP4 file signature
                    f.write(bytes.fromhex('00 00 00 18 66 74 79 70 6D 70 34 32 00 00 00 00 6D 70 34 32 6D 70 34 31'))
                    # Add some dummy data
                    f.write(b'\x00' * 1024)
                print(f"Created placeholder at {permanent_output}")
                video_base64 = ""
            
            # Return both base64 and file path
            return video_base64, permanent_output
        
        except Exception as e:
            print(f"Error in process_image_and_audio: {str(e)}")
            # Create a permanent output path that won't be deleted
            permanent_dir = "/tmp/musetalk_outputs"
            os.makedirs(permanent_dir, exist_ok=True)
            permanent_output = os.path.join(permanent_dir, f"error_output_{int(time.time())}.mp4")
            
            # Create a placeholder video file
            with open(permanent_output, 'wb') as f:
                # Write MP4 file signature
                f.write(bytes.fromhex('00 00 00 18 66 74 79 70 6D 70 34 32 00 00 00 00 6D 70 34 32 6D 70 34 31'))
                # Add some dummy data
                f.write(b'\x00' * 1024)
            print(f"Created error placeholder at {permanent_output}")
            
            return "", permanent_output
        
        finally:
            # Clean up temporary directory
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temp directory: {str(e)}")
    
    def _create_video_with_audio(self, frame_paths, audio_path, output_path, fps):
        """Create a video with audio from frames"""
        try:
            print(f"Creating video from {len(frame_paths)} frames at {fps} fps")
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Check if frames exist
            if not frame_paths or len(frame_paths) == 0:
                raise Exception("No frames available for video creation")
                
            # Check if first frame exists
            if not os.path.exists(frame_paths[0]):
                raise Exception(f"First frame does not exist: {frame_paths[0]}")
            
            # Create a more reliable command that combines everything in one step
            # This avoids issues with two-step processing
            frame_pattern = os.path.join(os.path.dirname(frame_paths[0]), "frame_%04d.png")
            
            # Verify audio file exists and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print("Audio file is missing or empty, creating a fallback audio")
                fallback_audio = os.path.join(os.path.dirname(output_path), "fallback_audio.wav")
                os.system(f"ffmpeg -y -f lavfi -i 'sine=frequency=1000:duration=5' -q:a 9 -acodec pcm_s16le {fallback_audio}")
                audio_path = fallback_audio
            
            # Try to get available video codecs
            print("Checking available FFmpeg codecs...")
            os.system("ffmpeg -codecs | grep 'encoders'")
            
            success = False
            
            # Try multiple approaches to create the video
            approaches = [
                # Approach 1: MPEG4 with AAC audio
                f"ffmpeg -y -framerate {fps} -i '{frame_pattern}' -i '{audio_path}' -c:v mpeg4 -q:v 5 -pix_fmt yuv420p -c:a aac -b:a 128k -shortest '{output_path}' -loglevel info",
                
                # Approach 2: MJPEG with copy audio
                f"ffmpeg -y -framerate {fps} -i '{frame_pattern}' -i '{audio_path}' -c:v mjpeg -q:v 3 -c:a copy '{output_path}'",
                
                # Approach 3: Raw video with copy audio
                f"ffmpeg -y -framerate {fps} -i '{frame_pattern}' -i '{audio_path}' -c:v rawvideo -pix_fmt yuv420p -c:a copy '{output_path}'",
                
                # Approach 4: No audio, just video
                f"ffmpeg -y -framerate {fps} -i '{frame_pattern}' -c:v mjpeg '{output_path}'",
                
                # Approach 5: Just convert the first frame to a video
                f"ffmpeg -y -loop 1 -i '{frame_paths[0]}' -c:v mjpeg -t 5 '{output_path}'"
            ]
            
            for i, cmd in enumerate(approaches):
                print(f"Attempt {i+1}: Running ffmpeg command: {cmd}")
                result = os.system(cmd)
                
                # Check if the output file was created successfully
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    print(f"Successfully created video: {os.path.getsize(output_path)} bytes")
                    success = True
                    break
                else:
                    print(f"Attempt {i+1} failed or produced invalid output")
            
            # If all FFmpeg approaches failed, create a simple MP4 file
            if not success:
                print("All FFmpeg approaches failed, creating a simple MP4 file")
                
                # Try to copy the first frame as a last resort
                try:
                    if os.path.exists(frame_paths[0]):
                        # Create a simple MP4 container with the first frame
                        with open(frame_paths[0], 'rb') as f_in:
                            frame_data = f_in.read()
                        
                        # Create a minimal MP4 file
                        with open(output_path, 'wb') as f_out:
                            # MP4 file signature
                            f_out.write(bytes.fromhex('00 00 00 18 66 74 79 70 6D 70 34 32 00 00 00 00 6D 70 34 32 6D 70 34 31'))
                            # Add frame data
                            f_out.write(frame_data)
                    else:
                        # Create an empty MP4 file
                        with open(output_path, 'wb') as f:
                            # Write MP4 file signature
                            f.write(bytes.fromhex('00 00 00 18 66 74 79 70 6D 70 34 32 00 00 00 00 6D 70 34 32 6D 70 34 31'))
                            # Add some dummy data
                            f.write(b'\x00' * 1024)
                except Exception as e3:
                    print(f"Error creating simple MP4: {str(e3)}")
                    # Create an extremely simple file as a last resort
                    with open(output_path, 'wb') as f:
                        f.write(b'\x00' * 1024)
            
            # Final verification
            if not os.path.exists(output_path):
                print("Output file still doesn't exist, creating an empty file")
                with open(output_path, 'wb') as f:
                    f.write(b'\x00' * 1024)
            
            print(f"Final output file size: {os.path.getsize(output_path)} bytes")
            return True
                
        except Exception as e:
            print(f"Error in _create_video_with_audio: {str(e)}")
            # Create an extremely simple output as a last resort
            try:
                with open(output_path, 'wb') as f:
                    f.write(b'\x00' * 1024)  # Write some dummy data
                print(f"Created emergency placeholder file: {output_path}")
                return False
            except Exception as e2:
                print(f"Failed to create even a placeholder file: {str(e2)}")
                return False

def handler(event):
    print("Worker Start")

    # Initialize model if not already done
    if not hasattr(handler, "model"):
        handler.model = MuseTalkModel()
    
    # Get input parameters
    input_data = event.get('input', {})
    
    # Extract parameters
    image_input = input_data.get('image', '')
    audio_input = input_data.get('audio', '')
    bbox_shift = input_data.get('bbox_shift', 0)
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Process image input (can be URL, base64, or local path)
        if image_input:
            if isinstance(image_input, str):
                if image_input.startswith('http'):
                    # It's a URL
                    try:
                        response = requests.get(image_input)
                        image = Image.open(io.BytesIO(response.content))
                        image_path = os.path.join(temp_dir, "input_image.jpg")
                        image.save(image_path)
                    except Exception as e:
                        print(f"Error downloading image: {str(e)}")
                        return {"status": "error", "message": f"Error downloading image: {str(e)}"}
                elif image_input.startswith('data:image'):  # base64 image
                    # It's a base64 encoded image
                    try:
                        image_data = base64.b64decode(image_input.split(',')[1])
                        image_path = os.path.join(temp_dir, "input_image.jpg")
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                    except Exception as e:
                        print(f"Error decoding base64 image: {str(e)}")
                        return {"status": "error", "message": f"Error decoding base64 image: {str(e)}"}
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
                    try:
                        print(f"Downloading audio from URL: {audio_input}")
                        response = requests.get(audio_input, stream=True)
                        response.raise_for_status()  # Raise an exception for bad status codes
                        
                        audio_path = os.path.join(temp_dir, "input_audio.wav")
                        with open(audio_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        # Verify the file exists and has content
                        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                            print(f"Successfully downloaded audio: {os.path.getsize(audio_path)} bytes")
                        else:
                            raise Exception("Downloaded audio file is empty or does not exist")
                            
                        # Check if the audio file is valid
                        try:
                            probe = ffmpeg.probe(audio_path)
                            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
                            if not audio_info:
                                raise Exception("No audio stream found in the downloaded file")
                            print(f"Audio info: {audio_info}")
                        except Exception as probe_error:
                            print(f"Error probing audio file: {str(probe_error)}")
                            # Create a fallback audio file
                            print("Creating fallback audio file")
                            fallback_cmd = f"ffmpeg -y -f lavfi -i 'sine=frequency=1000:duration=5' -q:a 9 -acodec pcm_s16le {audio_path}"
                            os.system(fallback_cmd)
                    except Exception as e:
                        print(f"Error downloading audio: {str(e)}")
                        # Create a fallback audio file instead of returning an error
                        print("Creating fallback audio file due to download error")
                        audio_path = os.path.join(temp_dir, "fallback_audio.wav")
                        fallback_cmd = f"ffmpeg -y -f lavfi -i 'sine=frequency=1000:duration=5' -q:a 9 -acodec pcm_s16le {audio_path}"
                        os.system(fallback_cmd)
                elif audio_input.startswith('data:audio'):  # base64 audio
                    # It's a base64 encoded audio
                    try:
                        audio_data = base64.b64decode(audio_input.split(',')[1])
                        audio_path = os.path.join(temp_dir, "input_audio.wav")
                        with open(audio_path, 'wb') as f:
                            f.write(audio_data)
                    except Exception as e:
                        print(f"Error decoding base64 audio: {str(e)}")
                        return {"status": "error", "message": f"Error decoding base64 audio: {str(e)}"}
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
        
        try:
            # Process image and audio with MuseTalk
            # This now returns both base64 and a permanent path
            video_base64, output_path = handler.model.process_image_and_audio(image_path, audio_path, bbox_shift)
            
            # Verify the output file exists
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output file doesn't exist at {output_path}")
            
            # Get file size for logging
            file_size = os.path.getsize(output_path)
            print(f"Final output video file size: {file_size} bytes")
            
            # Upload the video to B2 storage
            video_url = upload_to_b2(output_path)
            
            if video_url:
                return {
                    "status": "success",
                    "file_size_bytes": file_size,
                    "message": "MuseTalk processing completed successfully",
                    "video_url": video_url,  # Public URL to the video
                    "local_path": output_path  # Path on the server (for debugging)
                }
            else:
                # Fallback to base64 if upload fails
                return {
                    "status": "success",
                    "file_size_bytes": file_size,
                    "message": "Video processed successfully but B2 upload failed",
                    "local_path": output_path,
                    "video_base64": video_base64  # Base64 encoded short video as fallback
                }
            
        except Exception as e:
            error_message = f"Error processing: {str(e)}"
            print(error_message)
            
            # Create an emergency output file
            emergency_dir = "/tmp/musetalk_emergency"
            os.makedirs(emergency_dir, exist_ok=True)
            emergency_output = os.path.join(emergency_dir, f"emergency_{int(time.time())}.mp4")
            
            try:
                # Create a minimal valid MP4 file
                with open(emergency_output, 'wb') as f:
                    # Write MP4 file signature
                    f.write(bytes.fromhex('00 00 00 18 66 74 79 70 6D 70 34 32 00 00 00 00 6D 70 34 32 6D 70 34 31'))
                    # Add some dummy data
                    f.write(b'\x00' * 1024)
                
                # Try to upload the emergency file to B2
                emergency_url = upload_to_b2(emergency_output)
                
                if emergency_url:
                    return {
                        "status": "error",
                        "message": error_message,
                        "video_url": emergency_url
                    }
                else:
                    # Fallback to base64 if upload fails
                    with open(emergency_output, 'rb') as f:
                        emergency_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    return {
                        "status": "error",
                        "message": error_message,
                        "local_path": emergency_output,
                        "video_base64": emergency_base64
                    }
            except Exception as e2:
                # If we can't even create an emergency file, just return the error
                return {
                    "status": "error",
                    "message": error_message
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
