FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install the 'uv' CLI
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Make sure 'uv' is on PATH
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml /app/
COPY src/ /app/src
COPY Makefile /app/
COPY README.md /app/

# Build arguments for environment configuration
ARG EXTRA
ARG ENV_FILE=envs/api.env.example
ENV UV_SYSTEM_PYTHON=1

# Install dependencies
RUN make sync

# Expose the API port
EXPOSE 8000

# Run the FastAPI server
CMD ["uv", "run", "python", "-m", "ragprod.presentation.api.run", "--host", "0.0.0.0", "--port", "8000"]

# Build command:
# docker build -f docker/api.Dockerfile --build-arg EXTRA=cpu -t ragprod-api .
#
# Run command:
# docker run -p 8000:8000 ragprod-api
#
# Run with env file:
# docker run -p 8000:8000 -v $(pwd)/envs:/app/envs ragprod-api uv run python -m ragprod.presentation.api.run --env /app/envs/api.env
