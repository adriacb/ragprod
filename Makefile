# Makefile for managing project dependencies and environment setup

# Set default environment to dev
ENV ?= dev

# Define extras for uv sync
EXTRA ?= cpu

# Default target: install dependencies and setup environment
all: install

# Target for syncing dependencies using uv
sync:
	@echo "Syncing dependencies with uv for environment: $(ENV), extra: $(EXTRA)"
	uv sync --extra $(EXTRA)

# Target for installing the project in editable mode
install: sync
	@echo "Installing the project in editable mode..."
	uv pip install -e .

# Target for running tests (to be uncommented when needed)
# test:
#	@echo "Running tests..."
#	pytest

# Target for linting (to be uncommented when needed)
# lint:
#	@echo "Running linting..."
#	flake8 .

# Target for cleaning up the environment
clean:
	@echo "Cleaning up the environment..."
	rm -rf .venv

# Print help message
help:
	@echo "Makefile commands:"
	@echo "  sync       Syncs dependencies with uv for the specified extra (e.g., cpu, rocm, cu124)"
	@echo "  install    Installs the project in editable mode"
	@echo "  clean      Removes the virtual environment"
	@echo "  help       Displays this help message"
	@echo "  test       (Commented out) Run tests"
	@echo "  lint       (Commented out) Run linting"
