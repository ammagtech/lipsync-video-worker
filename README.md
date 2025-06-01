# FantasyTalking RunPod Worker

This is a RunPod serverless worker implementation of FantasyTalking - a state-of-the-art lip-sync video generation model. The worker takes an input image and audio file to generate realistic talking portrait videos with accurate lip synchronization.

## Features

- **Realistic Lip-Sync**: Generate high-quality talking portraits with accurate lip movements
- **GPU Accelerated**: Utilizes CUDA for fast video generation
- **Flexible Input**: Accepts base64 encoded images and audio files
- **Customizable**: Control video parameters like resolution, frame rate, and generation quality
- **RunPod Ready**: Optimized for RunPod serverless deployment

## Model Information

This implementation is based on:
- **FantasyTalking**: Realistic Talking Portrait Generation via Coherent Motion Synthesis
- **Base Model**: Wan2.1-I2V-14B-720P for video generation
- **Audio Processing**: Wav2Vec2 for audio feature extraction
- **Paper**: [arXiv:2504.04842](https://arxiv.org/abs/2504.04842)
- **Original Repository**: [Fantasy-AMAP/fantasy-talking](https://github.com/Fantasy-AMAP/fantasy-talking)

## Input Format

### **Test Mode (Quick Testing)**

For quick testing without real image/audio data, simply use:

```json
{
    "input": {
        "prompt": "test"
    }
}
```

This automatically uses dummy data and optimized settings for fast testing.

### **Full Input Format**

For production use with real data:

```json
{
    "input": {
        "image": "base64_encoded_image_string",
        "audio": "base64_encoded_audio_string",
        "prompt": "A woman is talking enthusiastically",
        "image_size": 512,
        "max_num_frames": 81,
        "fps": 23,
        "prompt_cfg_scale": 5.0,
        "audio_cfg_scale": 5.0,
        "seed": 1111
    }
}
```

### Parameters

- **prompt** (required): Text description to guide video generation
  - **Special value**: `"test"` - Activates test mode with dummy data
- **image** (required*): Base64 encoded image of the person (*not required in test mode)
- **audio** (required*): Base64 encoded audio file for lip-sync (*not required in test mode)
- **image_size** (optional): Target image size for processing (default: 512, test mode: 384)
- **max_num_frames** (optional): Maximum number of frames to generate (default: 81, test mode: 49)
- **fps** (optional): Frames per second for output video (default: 23)
- **prompt_cfg_scale** (optional): Prompt guidance scale (default: 5.0)
- **audio_cfg_scale** (optional): Audio guidance scale (default: 5.0)
- **seed** (optional): Random seed for reproducible results (default: 1111)

## Output Format

The worker returns:

```json
{
    "status": "success",
    "video_base64": "base64_encoded_video_string",
    "num_frames": 81,
    "fps": 23,
    "duration": 3.5,
    "resolution": "512x512",
    "prompt": "A woman is talking enthusiastically"
}
```

## Requirements

- **GPU**: NVIDIA GPU with at least 20GB VRAM (recommended)
- **CUDA**: CUDA 12.1 or compatible
- **Storage**: ~50GB for model weights
- **Memory**: 32GB+ RAM recommended

## To test this code locally:

**Note**: Local testing requires significant GPU resources. Consider using a cloud instance with GPU support.

```bash
# 1. Create a Python virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Update test_input.json with actual base64 encoded image and audio

# 5. Run the worker locally
python3 rp_handler.py
```

## Build and Push Docker Image to a Container Registry (e.g., Docker Hub)

```bash
# Build docker image (requires NVIDIA Docker support)
docker build -t your-dockerhub-username/fantasytalking-runpod:v1.0.0 --platform linux/amd64 .

# Push docker image to docker hub
docker push your-dockerhub-username/fantasytalking-runpod:v1.0.0
```

## RunPod Deployment

1. **Create a new Serverless Endpoint** in RunPod Console
2. **Configure the endpoint**:
   - **Container Image**: `your-dockerhub-username/fantasytalking-runpod:v1.0.0`
   - **GPU Type**: RTX 4090, A100, or similar (minimum 20GB VRAM)
   - **Container Disk**: 50GB+ (for model storage)
   - **Memory**: 32GB+ recommended
3. **Set Environment Variables** (if needed):
   - `CUDA_VISIBLE_DEVICES=0`
   - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
4. **Deploy** and test with the provided input format

## Quick Testing

### **Test Your Deployment**

Once deployed, test immediately with this simple request:

```json
{
    "input": {
        "prompt": "test"
    }
}
```

This will:
- ✅ Use dummy image and audio data (no need to upload files)
- ✅ Use optimized settings for faster processing (384px, 49 frames)
- ✅ Verify the entire pipeline works
- ✅ Take ~2-3 minutes instead of 5-10 minutes

### **Test Examples**

See `test_examples.json` for various test configurations:
- **Fast test**: 256px, 25 frames (~1 minute)
- **Quality test**: 512px, 81 frames (~5 minutes)
- **Custom settings**: Override any parameters

## Performance Notes

- **First Request**: May take 5-10 minutes due to model downloading
- **Subsequent Requests**: ~30-60 seconds per video (depending on length and GPU)
- **VRAM Usage**: ~20GB for full quality, can be reduced with `num_persistent_param_in_dit` parameter
- **Model Caching**: Models are cached after first download for faster subsequent runs

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `max_num_frames` or `image_size`
2. **Model Download Fails**: Check internet connection and HuggingFace access
3. **CUDA Errors**: Ensure GPU drivers and CUDA are properly installed
4. **FFmpeg Errors**: Audio/video encoding issues - check input formats

### Error Responses

The worker returns error responses in this format:
```json
{
    "status": "error",
    "message": "Description of the error"
}
```

## License

This implementation follows the original FantasyTalking license (Apache-2.0). Please refer to the original repository for full license details.

## Citation

If you use this implementation, please cite the original FantasyTalking paper:

```bibtex
@article{wang2025fantasytalking,
   title={FantasyTalking: Realistic Talking Portrait Generation via Coherent Motion Synthesis},
   author={Wang, Mengchao and Wang, Qiang and Jiang, Fan and Fan, Yaqi and Zhang, Yunpeng and Qi, Yonggang and Zhao, Kun and Xu, Mu},
   journal={arXiv preprint arXiv:2504.04842},
   year={2025}
}
```
