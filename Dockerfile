FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ARG VERSION=0.0.0.dev0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /build

COPY pyproject.toml ./
COPY wyoming_onnx_stt ./wyoming_onnx_stt/

# Create venv and install deps
RUN uv venv /opt/venv --python python3.11 && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgomp1 \
    libnvinfer10 \
    libnvinfer-plugin10 \
    libnvonnxparsers10 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

VOLUME /data

ENV STT_DATA_DIR=/data
ENV HF_HOME=/data/cache

EXPOSE 10300

ENTRYPOINT ["python", "-m", "wyoming_onnx_stt"]
