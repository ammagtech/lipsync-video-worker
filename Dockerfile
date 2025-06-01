# Use PyTorch base image with CUDA support (more reliable than nvidia/cuda)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip

# Install core dependencies first
RUN pip install --no-cache-dir runpod==1.7.7 transformers==4.46.2 safetensors einops

# Install audio/video processing dependencies
RUN pip install --no-cache-dir librosa soundfile imageio Pillow opencv-python

# Install ML dependencies
RUN pip install --no-cache-dir huggingface_hub accelerate diffusers

# Install remaining dependencies
RUN pip install --no-cache-dir sentencepiece protobuf ftfy modelscope

# Install DiffSynth-Studio (may fail, so we'll handle it in the code)
RUN pip install --no-cache-dir git+https://github.com/modelscope/DiffSynth-Studio.git || echo "DiffSynth-Studio installation failed, will install at runtime"

# Install controlnet-aux (optional, may cause issues)
RUN pip install --no-cache-dir controlnet-aux==0.0.7 || echo "controlnet-aux installation failed, continuing without it"

# Copy application files
COPY rp_handler.py /app/
COPY model.py /app/
COPY utils.py /app/

# Create models directory
RUN mkdir -p /app/models

# Optional: Pre-download models during build (uncomment to enable)
# This will make the Docker image ~32GB larger but eliminates first-request delay
# RUN python3 -c "from utils import download_models_if_needed; download_models_if_needed()"

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Start the container
CMD ["python3", "-u", "rp_handler.py"]