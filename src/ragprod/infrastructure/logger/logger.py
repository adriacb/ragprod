import structlog
import logging
import sys
from typing import Optional, Any, List
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    """Configuration class for structlog logger settings."""

    level: str = "INFO"
    json_format: bool = False
    processors: Optional[List[Any]] = None
    context_class: Optional[Any] = None
    wrapper_class: Optional[Any] = None
    cache_logger_on_first_use: bool = True
    logger_factory: Optional[Any] = None
    additional_processors: Optional[List[Any]] = None
    log_file: Optional[str] = None  # <-- add this

    def get_processors(self) -> List[Any]:
        """Get the list of processors based on configuration."""
        processors = [
            structlog.contextvars.merge_contextvars,  # Add contextvars processor first
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        # Add additional processors before the renderer
        if self.additional_processors:
            processors.extend(self.additional_processors)

        # Add the renderer last
        if self.json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        return processors


class LoggerInitializer:
    """Class for initializing and configuring structlog logger."""

    @staticmethod
    def initialize(config: LoggerConfig) -> structlog.BoundLogger:
        level_name = str(config.level).upper()
        level = getattr(logging, level_name, logging.INFO)
        root = logging.getLogger()

        # Clear existing handlers (optional, ensures re-initialization is clean)
        if root.handlers:
            root.handlers.clear()

        # Add StreamHandler only if no log_file is set
        if config.log_file:
            handler = logging.FileHandler(config.log_file)
        else:
            handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
        root.setLevel(level)
        for h in root.handlers:
            h.setLevel(level)

        structlog.configure(
            processors=config.get_processors(),
            wrapper_class=config.wrapper_class or structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=config.cache_logger_on_first_use,
            logger_factory=config.logger_factory or structlog.stdlib.LoggerFactory(),
        )

        structlog.contextvars.clear_contextvars()
        return structlog.get_logger()

    @staticmethod
    def get_default_logger() -> structlog.BoundLogger:
        """Get a logger with default configuration."""
        default_config = LoggerConfig()
        # Return a proxy so callers that grabbed a logger early keep using
        # a forwarding object that always delegates to the current
        # structlog.get_logger() (which reflects the latest configuration).
        global _LOGGER_PROXY
        try:
            _LOGGER_PROXY
        except NameError:
            _LOGGER_PROXY = None

        if _LOGGER_PROXY is None:
            # Ensure structlog is initialized at least once with defaults
            LoggerInitializer.initialize(default_config)
            _LOGGER_PROXY = LoggerProxy()

        return _LOGGER_PROXY


class LoggerProxy:
    """A tiny proxy that forwards attribute access to the current structlog logger.

    This allows modules that capture a logger early (at import time) to continue
    calling the same object while forwarding calls to the most recent logger
    configured by `LoggerInitializer.initialize()`.
    """

    def __getattr__(self, name):
        return getattr(structlog.get_logger(), name)

    @property
    def wrapped(self):
        """Access the current underlying BoundLogger if needed for advanced use."""
        return structlog.get_logger()

def get_logger(config: dict | None):
    if not config:
        return LoggerInitializer.get_default_logger()
    
    return LoggerConfig.initialize(
            LoggerConfig(
                **config
            )
        )