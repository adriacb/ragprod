FROM python:3.12-slim

# Install system dependencies.
RUN apt-get update && \
    apt-get install -y curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install the 'uv' CLI.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Make sure 'uv' is on PATH.
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml /app/
COPY src/ /app/src
COPY Makefile /app/
COPY README.md /app/

ARG EXTRA
ARG FASTMCP_SERVER_AUTH_GITHUB_CLIENT_ID
ARG FASTMCP_SERVER_AUTH_GITHUB_CLIENT_SECRET
ARG FASTMCP_SERVER_AUTH_BASE_URL
ENV UV_SYSTEM_PYTHON=1

RUN make sync

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "src.ragprod.presentation.mcp.run"]

#  docker build -f docker/mcp.Dockerfile --build-arg EXTRA=cpu -t ragprod-mcp .