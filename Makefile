.PHONY: install run dev test lint format check build clean docker help

VERSION ?= $(shell git describe --tags --dirty 2>/dev/null || echo "0.0.0.dev0")

install:
	uv sync

run:
	uv run wyoming-onnx-stt $(ARGS)

dev:
	uv run wyoming-onnx-stt --log-level DEBUG $(ARGS)

test:
	uv run pytest tests/

lint:
	uv run ruff check wyoming_onnx_stt/
	uv run mypy wyoming_onnx_stt/

format:
	uv run ruff format wyoming_onnx_stt/
	uv run ruff check --fix wyoming_onnx_stt/

check: lint format

build:
	uv build

docker:
	docker build --build-arg VERSION=$(VERSION) -t lmo3/wyoming-onnx-stt:$(VERSION) -t lmo3/wyoming-onnx-stt:latest .
