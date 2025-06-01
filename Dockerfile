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
RUN pip install --no-cache-dir -r requirements.txt

# Install huggingface-cli for model downloads
RUN pip install --no-cache-dir "huggingface_hub[cli]"

# Copy application files
COPY rp_handler.py /app/
COPY model.py /app/
COPY utils.py /app/

# Create models directory
RUN mkdir -p /app/models

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Start the container
CMD ["python3", "-u", "rp_handler.py"]