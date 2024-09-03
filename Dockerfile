FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Set non-interactive installation mode and configure timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Environment variables for Llama library
ENV LLAMA_CUBLAS=1
ENV CMAKE_ARGS=-DLLAMA_CUBLAS=on
ENV FORCE_CMAKE=1

# Install Python, build tools, compilers, and git
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    libblas-dev \
    liblapack-dev \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update pip and install wheel
RUN python3 -m pip install --upgrade pip wheel

# Install Requirements
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Special installation for llama-cpp-python with GPU support
RUN pip install llama-cpp-python==0.2.55 --no-cache-dir --force-reinstall --verbose

# Force specific numpy version
RUN python3 -m pip install numpy==1.26.2 --no-cache-dir --force-reinstall

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]