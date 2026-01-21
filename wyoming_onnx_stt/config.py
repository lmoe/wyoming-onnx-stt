from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import SERVICE_NAME


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="STT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    uri: str = Field(default="tcp://0.0.0.0:10300", description="Server URI")
    model: str = Field(default="nemo-canary-1b-v2", description="onnx-asr model name")
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device for inference")
    quantization: Literal["int8", "fp16"] | None = Field(
        default=None, description="Quantization level (int8 for lowest VRAM, fp16, or None for fp32)"
    )
    languages: str = Field(default="en", description="Comma-separated language codes the model supports")
    default_language: str = Field(default="en", description="Fallback language when client doesn't specify")
    data_dir: Path = Field(default=Path("./data"), description="Data directory for models")
    pnc: bool = Field(default=True, description="Enable punctuation and capitalization")
    zeroconf: str | None = Field(default=SERVICE_NAME, description="Zeroconf service name")
    log_level: str = Field(default="INFO", description="Log level")

    vad_enabled: bool = Field(default=False, description="Enable Silero VAD")
    vad_threshold: float = Field(default=0.5, description="Speech detection threshold (0.0-1.0)")
    vad_min_speech_ms: float = Field(default=250.0, description="Minimum speech duration in ms")
    vad_min_silence_ms: float = Field(default=100.0, description="Minimum silence duration to split segments in ms")
    vad_speech_pad_ms: float = Field(default=30.0, description="Padding added around speech segments in ms")

    tensorrt: bool = Field(default=False, description="Enable TensorRT execution provider")
    trt_workspace_gb: int = Field(default=6, description="TensorRT workspace size in GB")
    trt_fp16: bool = Field(default=True, description="Enable TensorRT FP16 mode")

    warmup: bool = Field(default=False, description="Warm up model at startup")
