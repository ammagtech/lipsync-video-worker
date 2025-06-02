# MuseTalk RunPod Worker

This is a serverless worker implementation for the [MuseTalk](https://huggingface.co/TMElyralab/MuseTalk) model on RunPod. The worker processes image and audio inputs to create lip-synced videos where the subject's mouth movements match the provided audio.

## Features

- Lip-syncing using the MuseTalk model from TMElyralab
- GPU acceleration for faster processing
- Automatic model loading and initialization
- Support for adjustable mouth openness via bbox_shift parameter
- Base64 encoded video output for easy integration
- Support for URL, base64, or local file paths for inputs

## Input Format

The worker accepts the following input parameters:

```json
{
    "input": {
        "image": "path/to/image.jpg",      // Path, URL, or base64 of input image
        "audio": "path/to/audio.wav",      // Path, URL, or base64 of audio file
        "bbox_shift": 0                     // Optional: Adjust mouth openness (-9 to 9)
    }
}
```

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
