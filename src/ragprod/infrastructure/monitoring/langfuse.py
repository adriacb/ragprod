"""Langfuse service for monitoring and tracing."""
from typing import Optional
from langfuse.callback import CallbackHandler
from fastagent.utils.logger import LoggerInitializer

logger = LoggerInitializer.get_default_logger()

def create_langfuse_callback(config) -> Optional[CallbackHandler]:
    """Create a Langfuse callback handler with the given settings.
    Returns None if credentials are not provided."""
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.warning("Langfuse client is disabled - missing credentials")
        return None

    try:
        return CallbackHandler(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
            timeout=config.langfuse_timeout,
            tags=config.langfuse_tags,
            version=config.langfuse_version,
            release=config.langfuse_release,
            environment=config.langfuse_environment
        )
    except Exception as e:
        logger.error(f"Failed to create Langfuse client: {e}")
        return None