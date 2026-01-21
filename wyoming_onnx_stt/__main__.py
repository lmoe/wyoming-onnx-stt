#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import logging
from functools import partial
from pathlib import Path

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .config import Settings
from .handler import SAMPLE_RATE, OnnxEventHandler, OnnxModel

_LOGGER = logging.getLogger(__name__)
_ONNX_ASR_ATTRIBUTION = Attribution(
    name="onnx-asr",
    url="https://github.com/istupakov/onnx-asr",
)


def parse_args(defaults: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wyoming ONNX STT server - speech-to-text using onnx-asr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python -m wyoming_onnx_stt

  # Low VRAM mode with int8 quantization (~500MB)
  python -m wyoming_onnx_stt --quantization int8

  # Use Parakeet model (English only, faster)
  python -m wyoming_onnx_stt --model nemo-parakeet-tdt-0.6b-v3 --languages en

  # Multilingual setup with Canary
  python -m wyoming_onnx_stt --languages en,de,fr,es --default-language en

Environment variables:
  All arguments can be set via STT_* env vars (e.g., STT_QUANTIZATION=int8)
""",
    )

    server = parser.add_argument_group("Server")
    server.add_argument("--uri", default=defaults.uri, help="Server URI (default: %(default)s)")
    server.add_argument(
        "--model",
        default=defaults.model,
        help="onnx-asr model name: nemo-canary-1b-v2, nemo-parakeet-tdt-0.6b-v3, gigaam-v2, etc. (default: %(default)s)",
    )
    server.add_argument(
        "--data-dir",
        type=Path,
        default=defaults.data_dir,
        help="Data directory for model cache (default: %(default)s)",
    )
    server.add_argument(
        "--zeroconf",
        default=defaults.zeroconf,
        help="Zeroconf service name for Home Assistant discovery, empty to disable (default: %(default)s)",
    )
    server.add_argument(
        "--log-level",
        default=defaults.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: %(default)s)",
    )
    server.add_argument("--version", action="version", version=__version__)

    lang = parser.add_argument_group("Language", "Configure supported languages for Home Assistant")
    lang.add_argument(
        "--languages",
        default=defaults.languages,
        help="Comma-separated language codes the model supports. Advertised to Home Assistant (default: %(default)s)",
    )
    lang.add_argument(
        "--default-language",
        default=defaults.default_language,
        help="Fallback language when client doesn't specify one (default: %(default)s)",
    )

    inference = parser.add_argument_group("Inference", "Model and hardware configuration")
    inference.add_argument(
        "--device",
        default=defaults.device,
        choices=["cuda", "cpu"],
        help="Inference device. CUDA recommended, CPU is slow (default: %(default)s)",
    )
    inference.add_argument(
        "--quantization",
        default=defaults.quantization,
        choices=["int8", "fp16"],
        help="Quantization for lower VRAM: int8 (~500MB), fp16 (~3GB), none (~4GB) (default: none)",
    )
    inference.add_argument(
        "--pnc",
        action=argparse.BooleanOptionalAction,
        default=defaults.pnc,
        help="Enable punctuation and capitalization in output (default: %(default)s)",
    )
    inference.add_argument(
        "--tensorrt",
        action=argparse.BooleanOptionalAction,
        default=defaults.tensorrt,
        help="Enable TensorRT execution provider for faster inference (default: %(default)s)",
    )
    inference.add_argument(
        "--trt-workspace-gb",
        type=int,
        default=defaults.trt_workspace_gb,
        help="TensorRT workspace size in GB (default: %(default)s)",
    )
    inference.add_argument(
        "--trt-fp16",
        action=argparse.BooleanOptionalAction,
        default=defaults.trt_fp16,
        help="Enable TensorRT FP16 mode (default: %(default)s)",
    )
    inference.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=defaults.warmup,
        help="Warm up model at startup (recommended with TensorRT) (default: %(default)s)",
    )

    vad = parser.add_argument_group("VAD", "Voice Activity Detection (segments audio before transcription)")
    vad.add_argument(
        "--vad-enabled",
        action=argparse.BooleanOptionalAction,
        default=defaults.vad_enabled,
        help="Enable Silero VAD (default: %(default)s)",
    )
    vad.add_argument(
        "--vad-threshold",
        type=float,
        default=defaults.vad_threshold,
        help="Speech detection threshold 0.0-1.0 (default: %(default)s)",
    )
    vad.add_argument(
        "--vad-min-speech-ms",
        type=float,
        default=defaults.vad_min_speech_ms,
        help="Minimum speech duration in ms (default: %(default)s)",
    )
    vad.add_argument(
        "--vad-min-silence-ms",
        type=float,
        default=defaults.vad_min_silence_ms,
        help="Minimum silence to split segments in ms (default: %(default)s)",
    )
    vad.add_argument(
        "--vad-speech-pad-ms",
        type=float,
        default=defaults.vad_speech_pad_ms,
        help="Padding around speech segments in ms (default: %(default)s)",
    )

    return parser.parse_args()


def parse_languages(languages_str: str) -> list[str]:
    return sorted(lang.strip().lower() for lang in languages_str.split(",") if lang.strip())


async def main() -> None:
    settings = Settings()
    args = parse_args(settings)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    languages = parse_languages(args.languages)
    if not languages:
        _LOGGER.warning("No languages specified, defaulting to 'en'")
        languages = ["en"]

    _LOGGER.info("Starting wyoming-onnx-stt")
    _LOGGER.info("Model: %s", args.model)
    _LOGGER.info("Languages: %s (default: %s)", ",".join(languages), args.default_language)
    _LOGGER.info("Device: %s, quantization: %s", args.device, args.quantization or "none")

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="onnx-asr",
                description="ONNX Speech-to-Text",
                attribution=_ONNX_ASR_ATTRIBUTION,
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=args.model,
                        description=args.model,
                        attribution=_ONNX_ASR_ATTRIBUTION,
                        installed=True,
                        languages=languages,
                        version=__version__,
                    )
                ],
            )
        ],
    )

    model = OnnxModel(
        model_name=args.model,
        device=args.device,
        quantization=args.quantization,
        pnc=args.pnc,
        tensorrt=args.tensorrt,
        trt_workspace_gb=args.trt_workspace_gb,
        trt_fp16=args.trt_fp16,
        vad_enabled=args.vad_enabled,
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
    )

    if args.warmup:
        import numpy as np

        _LOGGER.info("Warming up model...")
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence
        model.transcribe(dummy_audio, args.default_language)
        _LOGGER.info("Warmup complete")

    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// URI")

        from wyoming.zeroconf import HomeAssistantZeroconf

        zeroconf_host = None if server.host in ("0.0.0.0", "::") else server.host
        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf,
            port=server.port,
            host=zeroconf_host,
        )
        await hass_zeroconf.register_server()
        _LOGGER.info("Zeroconf discovery enabled: %s -> %s:%s", args.zeroconf, hass_zeroconf.host, server.port)

    _LOGGER.info("Server ready on %s", args.uri)
    model_lock = asyncio.Lock()

    await server.run(
        partial(
            OnnxEventHandler,
            wyoming_info,
            args.default_language,
            model,
            model_lock,
        )
    )


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        run()
