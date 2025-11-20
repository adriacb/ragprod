from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class BaseConfigModel(BaseSettings):
    """Base configuration model with environment loading capabilities."""

    model_config = {
        "env_file": None,
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
        "extra": "ignore",
        "populate_by_name": True,
        "validate_assignment": True,
        "env_prefix": "",
        "env_ignore_empty": True,
        "env_parse_file_values": True,
    }

    @classmethod
    def from_env_file(cls, env_file: Optional[str | Path] = None) -> "BaseConfigModel":
        """Create a configuration instance from an environment file.

        Args:
            env_file: Optional path to the environment file. If not provided, will look for .env in current directory.

        Returns:
            BaseConfigModel: Configuration instance with values loaded from environment.
        """
        if env_file:
            env_file = Path(env_file)
            if not env_file.exists():
                raise FileNotFoundError(f"Environment file not found: {env_file}")
            load_dotenv(env_file, override=True)

        return cls()