FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies with proper error handling
RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-opencv \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt

# Install dlib separately first with specific compiler flags
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir dlib==19.24.0

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Create model directories
RUN mkdir -p /app/models/musetalk \
    /app/models/sd-vae-ft-mse

# Download MuseTalk model and required files with better error handling
RUN mkdir -p /app/models/musetalk && \
    cd /app/models/musetalk && \
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 \
        https://huggingface.co/TMElyralab/MuseTalk/resolve/main/pytorch_model.bin || \
        echo "Warning: Could not download pytorch_model.bin" && \
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 \
        https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk.json || \
        echo "Warning: Could not download musetalk.json" && \
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 \
        https://huggingface.co/TMElyralab/MuseTalk/resolve/main/config.json || \
        echo "Warning: Could not download config.json" && \
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 \
        https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 || \
        echo "Warning: Could not download shape_predictor_68_face_landmarks.dat.bz2" && \
    if [ -f shape_predictor_68_face_landmarks.dat.bz2 ]; then \
        bunzip2 shape_predictor_68_face_landmarks.dat.bz2 || echo "Warning: Could not decompress file"; \
    fi

# Download sd-vae-ft-mse model with better error handling
RUN mkdir -p /app/models/sd-vae-ft-mse && \
    cd /app/models/sd-vae-ft-mse && \
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 \
        https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json || \
        echo "Warning: Could not download sd-vae config.json" && \
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 \
        https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin || \
        echo "Warning: Could not download diffusion_pytorch_model.bin"

# Download DWPose model with better error handling
RUN mkdir -p /app/models/dwpose && \
    cd /app/models/dwpose && \
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 \
        https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth || \
        echo "Warning: Could not download dw-ll_ucoco_384.pth"

# Download face-parse-bisent model with better error handling
RUN mkdir -p /app/models/face-parse-bisent && \
    cd /app/models/face-parse-bisent && \
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 \
        https://github.com/zllrunning/face-parsing.PyTorch/raw/master/res/cp/79999_iter.pth || \
        echo "Warning: Could not download 79999_iter.pth" && \
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 \
        https://download.pytorch.org/models/resnet18-5c106cde.pth || \
        echo "Warning: Could not download resnet18-5c106cde.pth"

# Download Whisper model for audio processing with better error handling
RUN mkdir -p /app/models/whisper && \
    cd /app/models/whisper && \
    wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 \
        https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt || \
        echo "Warning: Could not download Whisper tiny.pt model"

# Copy application code
COPY . /app/

# Start the container
CMD ["python3", "-u", "rp_handler.py"]