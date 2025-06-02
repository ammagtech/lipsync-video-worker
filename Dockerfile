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
    /app/models/whisper \
    /app/models/dwpose \
    /app/models/face-parse-bisent \
    /app/models/sd-vae-ft-mse

# Download MuseTalk model
RUN wget -q https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk.json -O /app/models/musetalk/musetalk.json && \
    wget -q https://huggingface.co/TMElyralab/MuseTalk/resolve/main/pytorch_model.bin -O /app/models/musetalk/pytorch_model.bin

# Download Whisper model
RUN wget -q https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt -O /app/models/whisper/tiny.pt

# Set environment variables
ENV FFMPEG_PATH=/usr/bin/ffmpeg

# Copy application code
COPY . /app/

# Start the container
CMD ["python3", "-u", "rp_handler.py"]