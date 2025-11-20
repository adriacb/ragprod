from pathlib import Path
import os
from types import SimpleNamespace
from dotenv import load_dotenv

from .fastapi_config import FastAPIConfigModel
from .langfuse_config import LangfuseConfigModel
from ragprod.infrastructure.logger import LoggerInitializer, LoggerConfig


def load_settings(env_path: str | None = None) -> SimpleNamespace:
    """
    Load environment file ('.env.dev' or '.env.pro') and return a simple settings container.
    """
    # decide profile early so we can choose sensible temp defaults
    if env_path:
        # Extract filename from path and remove .env prefix
        env_file_name = Path(env_path).name.lower()
        env_name = env_file_name.replace(".env", "").replace(".", "")
        if not env_name:
            env_name = "dev"
    else:
        env_name = "dev"
    # temporary logger to capture loader messages (will be reconfigured after env file is loaded)
    temp_level = "DEBUG" if env_name == "dev" else "INFO"
    logger = LoggerInitializer.initialize(
        LoggerConfig(level=temp_level, json_format=False)
    )

    if env_path:
        load_dotenv(env_path, override=True)
        logger.debug("Loading env file: %s", env_path)
    else:
        logger.debug(
            "No env file found for env=%s; using process environment", env_name
        )

    # Instantiate config models; be tolerant if required values are missing
    fastapi_conf = None
    langfuse_conf = None

    # Reconfigure global logger based on LOG_LEVEL from env file (or sensible defaults per profile)
    # If .env set LOG_LEVEL it will now be in os.environ because we loaded env_file above.
    final_level = (
        os.environ.get("LOG_LEVEL") or ("DEBUG" if env_name == "dev" else "WARNING")
    ).upper()
    # Avoid caching bound logger wrappers so reconfiguration in this loader
    # propagates to other modules that request loggers after settings are loaded.
    logger = LoggerInitializer.initialize(
        LoggerConfig(
            level=final_level, json_format=False, cache_logger_on_first_use=False
        )
    )

    try:
        fastapi_conf = (
            FastAPIConfigModel.from_env_file(env_path)
            if env_path
            else FastAPIConfigModel()
        )
    except Exception as ex:  # pragma: no cover - defensive for missing optional values
        logger.warning("Failed to build FastAPIConfigModel: %s", ex)
        try:
            fastapi_conf = FastAPIConfigModel()  # fallback to defaults
        except Exception:
            fastapi_conf = None

    try:
        langfuse_conf = (
            LangfuseConfigModel.from_env_file(env_path) if env_path else None
        )
    except (
        Exception
    ) as ex:  # pragma: no cover - Langfuse may require secrets in prod only
        logger.warning("Failed to build LangfuseConfigModel: %s", ex)
        langfuse_conf = None

    return SimpleNamespace(
        env=env_name,
        env_file=str(env_path) if env_path else None,
        fastapi=fastapi_conf,
        langfuse=langfuse_conf,
    )


# module-level SETTINGS for easy import across the app
# SETTINGS = load_settings()