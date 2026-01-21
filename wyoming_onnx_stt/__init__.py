from importlib.metadata import PackageNotFoundError, version

SERVICE_NAME = "wyoming-onnx-stt"
try:
    __version__ = version(SERVICE_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

