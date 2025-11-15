# Makefile for managing project dependencies and environment setup

# Default environment and extra
ENV ?= dev
EXTRA ?= cpu

# Default target
all: install

# Target for syncing dependencies using uv
sync:
	@echo "Syncing dependencies with uv for environment: $(ENV), extra: $(EXTRA)"
	@# Use system Python to avoid uv downloading incompatible versions
	uv sync --extra $(EXTRA) || { echo "uv sync failed"; exit 1; }

# Target for installing the project in editable mode
install: sync
	@echo "Installing the project in editable mode..."
	uv pip install -e .

# Target for running tests (optional)
# test:
#	@echo "Running tests..."
#	pytest

# Target for linting (optional)
# lint:
#	@echo "Running linting..."
#	flake8 .

# Clean virtual environment
clean:
	@echo "Cleaning up the environment..."
	@if [ -d ".venv" ]; then rm -rf .venv; fi

# Print help
help:
	@echo "Makefile commands:"
	@echo "  sync       Sync dependencies using uv for specified extra (cpu, rocm, cu124)"
	@echo "  install    Install project in editable mode"
	@echo "  clean      Remove virtual environment"
	@echo "  help       Display this help message"
	@echo "  test       (optional) Run tests"
	@echo "  lint       (optional) Run linting"
