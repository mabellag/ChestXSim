# ============================
# STAGE 1: BUILDER
# ============================
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# System deps + Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 python3-pip \
        libglib2.0-0 libxext6 libsm6 libxrender1 libgl1-mesa-glx \
        libgfortran5 libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python / pip point to 3.10 and upgrade tooling
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip setuptools wheel

# Copy only what is needed to install chestxsim
COPY pyproject.toml ./
COPY src ./src

# 1) GPU PyTorch (cu121 wheels)
RUN pip install --no-cache-dir \
      torch==2.1.0 torchvision==0.16.0 \
      --index-url https://download.pytorch.org/whl/cu121

# 2) Install ALL remaining deps + chestxsim from pyproject.toml
#    (cupy-cuda12x, astra-toolbox, fastai, etc.)
RUN pip install --no-cache-dir . \
    && rm -rf /root/.cache

# ============================
# STAGE 2: FINAL RUNTIME
# ============================
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS final

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# Python runtime + basic libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 python3-pip \
        libglib2.0-0 libxext6 libsm6 libxrender1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# CRITICAL FIX: pip-installed packages live in *dist-packages*, not site-packages
COPY --from=builder /usr/local/lib/python3.10/dist-packages \
                    /usr/local/lib/python3.10/dist-packages
# console scripts (run_simulation, interpolate, etc.)
COPY --from=builder /usr/local/bin /usr/local/bin

# Use the installed console script as entrypoint
# ENTRYPOINT ["run_simulation"]
CMD ["--help"]
