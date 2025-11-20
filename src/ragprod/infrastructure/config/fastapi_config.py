from typing import Optional
from pydantic import Field
from .base import BaseConfigModel


class FastAPIConfigModel(BaseConfigModel):
    """Pydantic model for FastAPI configuration."""

    host: str = Field(default="0.0.0.0", alias="FASTAPI_HOST")
    port: int = Field(default=8000, alias="FASTAPI_PORT")
    workers: int = Field(default=1, alias="FASTAPI_WORKERS")
    reload: bool = Field(default=True, alias="FASTAPI_RELOAD")
    access_log: bool = Field(default=True, alias="FASTAPI_ACCESS_LOG")
    api_prefix: str = Field(default="/api", alias="FASTAPI_API_PREFIX")
    api_title: str = Field(default="RAGProd API", alias="FASTAPI_API_TITLE")
    api_description: str = Field(
        default="API for RAGProd", alias="FASTAPI_API_DESCRIPTION"
    )
    api_version: str = Field(default="0.1.0", alias="FASTAPI_API_VERSION")


class FastAPIConfig(FastAPIConfigModel):
    """FastAPI configuration implementation."""

    def __init__(self, config: Optional[FastAPIConfigModel] = None):
        """Initialize FastAPI configuration.

        Args:
            config: Optional FastAPIConfigModel instance. If not provided, reads from environment.
        """
        self._config = config or FastAPIConfigModel()

    def get_host(self) -> str:
        return self._config.host

    def get_port(self) -> int:
        return self._config.port

    def get_workers(self) -> int:
        return self._config.workers

    def get_reload(self) -> bool:
        return self._config.reload

    def get_access_log(self) -> bool:
        return self._config.access_log

    def get_api_prefix(self) -> str:
        return self._config.api_prefix

    def get_api_title(self) -> str:
        return self._config.api_title

    def get_api_description(self) -> str:
        return self._config.api_description

    def get_api_version(self) -> str:
        return self._config.api_version