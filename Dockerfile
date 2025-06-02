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

# Download MuseTalk model (simplified for testing)
RUN wget -q https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk.json -O /app/models/musetalk/musetalk.json && \
    wget -q https://huggingface.co/TMElyralab/MuseTalk/resolve/main/pytorch_model.bin -O /app/models/musetalk/pytorch_model.bin

# Set environment variables
ENV FFMPEG_PATH=/usr/bin/ffmpeg

# Copy application code
COPY . /app/

# Start the container
CMD ["python3", "-u", "rp_handler.py"]