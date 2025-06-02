# MuseTalk Lip-Sync Worker for RunPod

This repository contains a RunPod worker for lip-syncing videos using the MuseTalk model from TMElyralab.

## Features

- Lip-syncing using the actual MuseTalk model
- GPU acceleration for faster processing
- Face detection and landmark tracking
- Audio-driven mouth movement synthesis
- Support for adjustable face position via bbox_shift parameter
- Base64 encoded video output for easy integration
- Support for URL, base64, or local file paths for inputs

## Input Format

The worker accepts the following input format:

```json
{
  "input": {
    "image": "[URL, base64, or local path]",
    "audio": "[URL, base64, or local path]",
    "bbox_shift": 0
  }
}
```

- `image`: Can be a URL, base64 encoded string, or local file path. Must contain a clearly visible face.
- `audio`: Can be a URL, base64 encoded string, or local file path. Audio will drive the lip movements.
- `bbox_shift`: Optional parameter for adjusting the face bounding box position (-9 to 9). Useful if face detection is slightly off.

If `image` and `audio` are not provided, the worker will generate test files for demonstration.

## Output Format

The worker returns the following output:

```json
{
    "status": "success",
    "video_base64": "base64_encoded_video_data",
    "message": "MuseTalk processing completed successfully"
}
```

## To test this code locally:

```bash
# 1. Create a Python virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install the required dependencies
pip install -r requirements.txt

# 4. Run your script locally
# The script will automatically read test_input.json as input
python3 rp_handler.py
```

## Build and Push Docker Image

```bash
# Build docker image
docker build -t your-dockerhub-username/musetalk-worker:v1.0.0 --platform linux/amd64 .

# Push docker image to docker hub
docker push your-dockerhub-username/musetalk-worker:v1.0.0
```

## RunPod Deployment

1. Push your Docker image to a container registry (Docker Hub, etc.)
2. Create a new Serverless Endpoint on RunPod
3. Select the appropriate GPU type (recommended: at least 16GB VRAM)
4. Set the Docker image to your pushed image
5. Configure worker count and other settings as needed
6. Deploy the endpoint

## Model Information

This worker uses the MuseTalk model from TMElyralab. For more information about the model, visit the [MuseTalk Hugging Face page](https://huggingface.co/TMElyralab/MuseTalk).
