FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ARG TENSORRT_VERSION=10.9.0.34
ARG CUDA_USER_VERSION=12.8
ARG OPERATING_SYSTEM=Linux

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# System & Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common wget gnupg2 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-distutils \
    && wget -qO get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && ln -sf /usr/bin/pip3.10   /usr/local/bin/pip3 \
    && python3 --version

# System libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool pkg-config \
      ca-certificates git curl libjpeg-dev libpng-dev \
      language-pack-en locales locales-all \
      python3-setuptools libprotobuf-dev protobuf-compiler \
      zlib1g-dev swig vim gdb valgrind libavcodec-dev  \
      libsm6 libxext6 libxrender-dev cmake libgtk2.0-dev  \
      libavformat-dev libswscale-dev ffmpeg  \

    && rm -rf /var/lib/apt/lists/*

# TensorRT install
# TODO: Unsure if this works, since the application does not use TensorRT currently
COPY ./libs/TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz /opt
RUN cd /opt \
    && tar -xzf TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz \
    && rm TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz \
    && PYTHON_TAG=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')") \
    && find TensorRT-${TENSORRT_VERSION}/python \
         -type f \
         -name "*${PYTHON_TAG}-none-linux_x86_64.whl" \
         -exec python3 -m pip install --no-cache-dir {} +

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/TensorRT-${TENSORRT_VERSION}/lib \
    PATH=$PATH:/opt/TensorRT-${TENSORRT_VERSION}/bin

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
  && echo "export PATH=/root/.local/bin:\$PATH" >> /etc/profile.d/poetry.sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy necessary files
WORKDIR /app
COPY pyproject.toml poetry.lock /app/
COPY README.md /app
COPY src /app/src
COPY checkpoints/FastSAM-x.pt \ 
     checkpoints/sam2.1_hiera_base_plus.pt \ 
     checkpoints/sam2.1_hiera_large.pt \
     checkpoints/sam2.1_hiera_small.pt \
     checkpoints/sam2.1_hiera_tiny.pt \
     /app/checkpoints/

# Install dependencies
RUN mkdir libs && cd libs && \
    git clone https://github.com/tobiipro/g3pylib.git && \
    git clone https://github.com/facebookresearch/dinov2.git && \
    git clone https://github.com/ilianbronchart/sam2.git && \
    poetry config virtualenvs.create false && poetry install && \
    poetry run pip install -e sam2