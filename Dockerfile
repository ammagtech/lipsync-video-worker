# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
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
        libgomp1 \
        cmake \
        ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /

# Copy requirements and install Python dependencies (excluding flash-attn)
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip

# Temporarily remove flash-attn to avoid pip install error
RUN grep -v "flash-attn" /requirements.txt > /temp_requirements.txt

# Install all other requirements
RUN pip3 install --no-cache-dir -r /temp_requirements.txt

# Manually install flash-attn from source
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    pip install packaging && \
    pip install . --no-build-isolation && \
    cd .. && rm -rf flash-attention

# Copy application files
COPY rp_handler.py /
COPY config.py /
COPY model_download.py /

# Create models directory
RUN mkdir -p /models

# Set proper permissions
RUN chmod +x /rp_handler.py

# Start the container
CMD ["python3", "-u", "rp_handler.py"]
