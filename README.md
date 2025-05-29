# FantasyTalking RunPod Worker

This is a RunPod serverless worker for FantasyTalking - a realistic talking portrait generation system. The worker takes an image and audio input to generate a talking video where the person in the image appears to be speaking the provided audio.

## Model Information

This worker implements the FantasyTalking model from [acvlab/FantasyTalking](https://huggingface.co/acvlab/FantasyTalking), which generates realistic talking portraits via coherent motion synthesis.

### Required Models
- **Wan2.1-I2V-14B-720P**: Base video generation model (14B parameters)
- **Wav2Vec2-base-960h**: Audio feature extraction
- **FantasyTalking checkpoint**: Specialized weights for talking portrait generation

## Input Format

The worker expects a JSON input with the following structure:

```json
{
    "input": {
        "image": "base64_encoded_image",
        "audio": "base64_encoded_audio",
        "prompt": "Optional text prompt describing the behavior",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "prompt_cfg_scale": 5.0,
        "audio_cfg_scale": 5.0,
        "num_frames": 81,
        "fps": 8
    }
}
```

### Required Parameters
- `image`: Base64 encoded image (JPG, PNG, etc.)
- `audio`: Base64 encoded audio (WAV, MP3, etc.)

### Optional Parameters
- `prompt`: Text description of desired behavior (default: "")
- `num_inference_steps`: Number of denoising steps (default: 50)
- `guidance_scale`: Classifier-free guidance scale (default: 7.5)
- `prompt_cfg_scale`: Prompt conditioning strength (default: 5.0)
- `audio_cfg_scale`: Audio conditioning strength (default: 5.0)
- `num_frames`: Number of video frames to generate (default: 81)
- `fps`: Output video frame rate (default: 8)

## Output Format

The worker returns a JSON response:

```json
{
    "status": "success",
    "video_base64": "base64_encoded_mp4_video",
    "prompt": "input_prompt",
    "num_frames": 81,
    "fps": 8,
    "format": "mp4",
    "resolution": "512x512"
}
```

## Local Testing

### 1. Setup Environment

```bash
# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Download required models (this may take a while due to large model sizes)
python3 model_download.py
```

### 3. Test Locally

```bash
# Run the handler locally with test input
python3 rp_handler.py
```

The script will automatically read `test_input.json` and process it through the handler.

## Docker Deployment

### Build and Push Docker Image

```bash
# Build docker image for GPU deployment
docker build -t your-dockerhub-username/fantasytalking-worker:v1.0.0 --platform linux/amd64 .

# Push docker image to docker hub
docker push your-dockerhub-username/fantasytalking-worker:v1.0.0
```

### RunPod Deployment

1. Go to the [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Select "Custom Source" → "Docker Image"
4. Enter your Docker image URL
5. **Important**: Select GPU instances (minimum 16GB VRAM recommended)
6. Configure endpoint settings and deploy

## Hardware Requirements

- **GPU**: NVIDIA GPU with at least 16GB VRAM (24GB+ recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ for models and temporary files

## Performance Notes

- First request may take longer due to model loading
- Video generation time depends on number of frames and inference steps
- Memory optimization is enabled to reduce VRAM usage
- Consider using fewer frames or lower resolution for faster generation

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `num_frames`, enable CPU offloading in config
2. **Model Download Fails**: Check internet connection, try manual download
3. **Slow Generation**: Reduce `num_inference_steps`, use smaller frame count

### Model Download Issues

If automatic model download fails, you can manually download models:

```bash
# Using huggingface-cli
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./models/Wan2.1-I2V-14B-720P
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
huggingface-cli download acvlab/FantasyTalking --files fantasytalking_model.ckpt --local-dir ./models/fantasytalking_model
```

## License

This project uses the FantasyTalking model which is licensed under Apache 2.0. Please refer to the original model repository for detailed licensing information.
