FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

# Set non-interactive frontend to prevent apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# CRITICAL: Allow pip to install globally on Ubuntu 24.04 without a venv
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set the base directory environment variable
ENV VLLM_BASE_DIR=/workspace/vllm

# 1. Install System Dependencies
# Added 'git', 'wget', and 'python3-pip' as they are required for the script steps
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    ninja-build \
    python3-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Setup Workspace
WORKDIR $VLLM_BASE_DIR

# 2. Download Tiktoken files
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# 3. Set Environment Variables
# Note: TORCH_CUDA_ARCH_LIST=12.1a is very specific (Hopper/H100 usually).
# Ensure this matches your target hardware.
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings

# 4. Install Python Dependencies (Using pip instead of uv)
#RUN python3 -m pip install --upgrade pip

# Install PyTorch for CUDA 13.0
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install Helper libraries
RUN pip install xgrammar triton

# Install FlashInfer
# Note: Using the same index URLs as provided in your script
RUN pip install flashinfer-python --no-deps --index-url https://flashinfer.ai/whl && \
    pip install flashinfer-cubin --index-url https://flashinfer.ai/whl && \
    pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130

# Install fast safetensors to improve loading speeds
RUN pip install fastsafetensors>=0.1.10

# 5. Clone and Build vLLM
RUN git clone --recursive https://github.com/vllm-project/vllm.git
WORKDIR $VLLM_BASE_DIR/vllm

# Prepare build requirements
RUN python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    pip install -r requirements/build.txt

# Final Build
# Uses --no-build-isolation to respect the pre-installed Torch/FlashInfer
# Changed -e (editable) to . (standard install) for better Docker portability
RUN pip install --no-build-isolation . -v

WORKDIR $VLLM_BASE_DIR
# Default entrypoint (Optional: starts the server)
CMD ["/bin/bash"]
