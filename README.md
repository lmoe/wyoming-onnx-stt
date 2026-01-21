# wyoming-onnx-stt

A Wyoming protocol server for ONNX-based speech-to-text, built for Home Assistant.

NVIDIA's ASR models (Canary, Parakeet) deliver excellent accuracy. The onnx-asr library runs these models with NumPy and ONNX Runtime only, which reduces VRAM usage significantly and simplifies deployment.

## Features

- Wyoming protocol native (Zeroconf discovery included)
- Multiple ASR models via onnx-asr (Canary, Parakeet, GigaAM, Zipformer)
- INT8/FP16 quantization for reduced VRAM usage
- TensorRT execution provider support for faster inference
- Silero VAD for filtering silence and reducing unnecessary processing

## Supported Models

Any model supported by [onnx-asr](https://github.com/istupakov/onnx-asr#supported-model-architectures):

| Model | Languages | Notes |
|-------|-----------|-------|
| `nemo-canary-1b-v2` (default) | 25 EU languages | Best multilingual accuracy |
| `nemo-parakeet-tdt-0.6b-v3` | English | Fast, English-only |
| `nemo-parakeet-tdt-0.6b-v2` | English | Older Parakeet |

See [onnx-asr docs](https://github.com/istupakov/onnx-asr) for the full list.

## Quick Start

```bash
docker run -d \
  --gpus all \
  -p 10300:10300 \
  --name wyoming-onnx-stt \
  -v /path/to/data:/data \
  -e STT_LANGUAGES=en \
  lmo3/wyoming-onnx-stt:latest
```

Then add to Home Assistant:

1. Settings -> Devices & Services -> Add Integration -> Wyoming Protocol -> Enter IP and port (default 10300)
   - Or use the auto-detected wyoming-onnx-stt node if HA received the Zeroconf advertisement
2. Settings -> Voice assistants -> Select your assistant -> Speech-to-text -> wyoming-onnx-stt
3. Done

## Configuration

Environment variables only. No config files.

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_URI` | `tcp://0.0.0.0:10300` | Server address |
| `STT_MODEL` | `nemo-canary-1b-v2` | onnx-asr model name |
| `STT_LANGUAGES` | `en` | Supported languages (comma-separated) |
| `STT_DEFAULT_LANGUAGE` | `en` | Fallback when client does not specify |
| `STT_DEVICE` | `cuda` | `cuda` or `cpu` |
| `STT_QUANTIZATION` | (none) | `int8` or `fp16` for lower VRAM |
| `STT_PNC` | `true` | Punctuation and capitalization |
| `STT_ZEROCONF` | `wyoming-onnx-stt` | Zeroconf service name (empty to disable) |
| `STT_LOG_LEVEL` | `INFO` | Log level |
| `STT_DATA_DIR` | `/data` | Data directory for model cache |

Or use CLI arguments (`--model`, `--languages`, `--quantization`, etc.).

### Language Configuration

Home Assistant reads the language list from your Wyoming node and displays those languages in its configuration dropdown. The onnx-asr library does not expose which languages a model supports through its API. You must tell this server which languages to advertise. **You only need to advertise the languages you want to use.**

```bash
# English only (Parakeet)
docker run ... -e STT_MODEL=nemo-parakeet-tdt-0.6b-v3 -e STT_LANGUAGES=en ...

# Multilingual (Canary)
docker run ... -e STT_LANGUAGES=en,de,fr,es ...
```

### Quantization

Quantization reduces VRAM usage at the cost of some accuracy. Measurements for `nemo-canary-1b-v2`:

| Quantization | VRAM |
|--------------|------|
| None (fp32) | ~4-5 GB |
| `fp16` | ~3-4 GB |
| `int8` | ~500 MB |

```bash
# Low VRAM setup
docker run ... -e STT_QUANTIZATION=int8 ...
```

### TensorRT

TensorRT can accelerate inference on NVIDIA GPUs. The first run takes longer because TensorRT builds optimized engine files for your specific GPU.

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_TENSORRT` | `false` | Enable TensorRT execution provider |
| `STT_TRT_WORKSPACE_GB` | `6` | TensorRT workspace size in GB (VRAM) |
| `STT_TRT_FP16` | `true` | Enable TensorRT FP16 mode |

```bash
docker run ... -e STT_TENSORRT=true ...
```

### Voice Activity Detection (VAD)

Silero VAD filters out silence before sending audio to the ASR model. This reduces unnecessary processing and can improve transcription quality by removing leading/trailing silence.

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_VAD_ENABLED` | `false` | Enable Silero VAD |
| `STT_VAD_THRESHOLD` | `0.5` | Speech detection threshold (0.0-1.0) |
| `STT_VAD_MIN_SPEECH_MS` | `250` | Minimum speech duration in ms |
| `STT_VAD_MIN_SILENCE_MS` | `100` | Minimum silence to split segments in ms |
| `STT_VAD_SPEECH_PAD_MS` | `30` | Padding around detected speech in ms |

```bash
docker run ... -e STT_VAD_ENABLED=true ...
```

### Warmup

Enable warmup to run a dummy transcription at startup. This loads the model into memory so the first real request is faster.

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_WARMUP` | `false` | Warm up model at startup |

## Data

Mount a volume to `/data` for model persistence:

```
/data/
└── cache/    # Model cache (auto-downloaded)
```

Models download automatically on first run. Mount a volume to avoid re-downloading on container restart.

## Requirements (Docker)

- NVIDIA GPU with CUDA support (or CPU, but slow)
- Docker with nvidia-container-toolkit
- 2-5 GB disk space depending on model

## Local Development

Prerequisites: [uv](https://docs.astral.sh/uv/), NVIDIA GPU with CUDA (optional)

```bash
git clone https://github.com/lmoe/wyoming-onnx-stt
cd wyoming-onnx-stt

uv sync
make dev
```

Development commands:

```bash
make run        # run server
make dev        # run with debug logging
make lint       # ruff + mypy
make format     # format code
make test       # run tests
```

## License

MIT
