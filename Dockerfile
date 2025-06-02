FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create model directories
RUN mkdir -p /app/models/musetalk \
    /app/models/sd-vae-ft-mse

# Instead of downloading models in Dockerfile, we'll handle it in the runtime
# This avoids potential download issues during build

# Set environment variables
ENV FFMPEG_PATH=/usr/bin/ffmpeg

# Copy application code
COPY . /app/

# Start the container
CMD ["python3", "-u", "rp_handler.py"]