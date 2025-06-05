# Adaptive Speculative Decoding Docker Image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    vim \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install additional packages for H100 optimization
RUN pip3 install \
    flash-attn \
    xformers \
    triton

# Copy application code
COPY . .

# Install the package in development mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p logs checkpoints results data

# Expose ports
EXPOSE 8000 9090

# Set up entrypoint
ENTRYPOINT ["/bin/bash"]