# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libfontconfig1 \
    libxrender1 \
    libgomp1

# Set working directory
WORKDIR /

# Copy requirements and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY rp_handler.py /
COPY config.py /
COPY model_download.py /

# Create models directory
RUN mkdir -p /models

# Set proper permissions
RUN chmod +x /rp_handler.py

# Start the container - modify to download models at runtime
CMD python3 model_download.py && python3 -u rp_handler.py