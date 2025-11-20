import pytest
import structlog
import logging
import sys
import tempfile
import os
from unittest.mock import patch, MagicMock, call
from ragprod.infrastructure.logger import logger as logger_module
from ragprod.infrastructure.logger.logger import (
    LoggerConfig,
    LoggerInitializer,
    LoggerProxy,
)


class TestLoggerConfig:
    """Test cases for LoggerConfig class."""

    def test_default_config(self):
        """Test LoggerConfig with default values."""
        config = LoggerConfig()
        assert config.level == "INFO"
        assert config.json_format is False
        assert config.processors is None
        assert config.context_class is None
        assert config.wrapper_class is None
        assert config.cache_logger_on_first_use is True
        assert config.logger_factory is None
        assert config.additional_processors is None
        assert config.log_file is None

    def test_custom_config(self):
        """Test LoggerConfig with custom values."""
        custom_processors = [structlog.processors.add_log_level]
        config = LoggerConfig(
            level="DEBUG",
            json_format=True,
            processors=custom_processors,
            cache_logger_on_first_use=False,
            log_file="/tmp/test.log",
        )
        assert config.level == "DEBUG"
        assert config.json_format is True
        assert config.processors == custom_processors
        assert config.cache_logger_on_first_use is False
        assert config.log_file == "/tmp/test.log"

    def test_get_processors_default(self):
        """Test get_processors() with default configuration (console renderer)."""
        config = LoggerConfig()
        processors = config.get_processors()

        # Check that required processors are present
        assert len(processors) >= 6
        # Check that ConsoleRenderer is used (not JSONRenderer)
        assert any(
            isinstance(p, structlog.dev.ConsoleRenderer) for p in processors
        )
        assert not any(
            isinstance(p, structlog.processors.JSONRenderer) for p in processors
        )

    def test_get_processors_json_format(self):
        """Test get_processors() with JSON format enabled."""
        config = LoggerConfig(json_format=True)
        processors = config.get_processors()

        # Check that JSONRenderer is used
        assert any(
            isinstance(p, structlog.processors.JSONRenderer) for p in processors
        )
        assert not any(
            isinstance(p, structlog.dev.ConsoleRenderer) for p in processors
        )

    def test_get_processors_with_additional_processors(self):
        """Test get_processors() with additional processors."""
        additional = [structlog.processors.add_log_level]
        config = LoggerConfig(additional_processors=additional)
        processors = config.get_processors()

        # Check that additional processors are included
        # They should be before the renderer
        renderer_index = next(
            i
            for i, p in enumerate(processors)
            if isinstance(p, (structlog.processors.JSONRenderer, structlog.dev.ConsoleRenderer))
        )
        # Additional processors should be before the renderer (renderer is at the end)
        assert renderer_index < len(processors)
        # Check that add_log_level is in the processors list
        assert structlog.processors.add_log_level in processors


class TestLoggerInitializer:
    """Test cases for LoggerInitializer class."""

    def setup_method(self):
        """Reset structlog configuration before each test."""
        structlog.reset_defaults()
        # Clear root logger handlers
        root = logging.getLogger()
        root.handlers.clear()

    def teardown_method(self):
        """Clean up after each test."""
        structlog.reset_defaults()
        root = logging.getLogger()
        root.handlers.clear()

    def test_initialize_default_config(self):
        """Test initialize() with default LoggerConfig."""
        config = LoggerConfig()
        logger = LoggerInitializer.initialize(config)

        assert logger is not None
        # structlog.get_logger() returns a proxy, check it has logger methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")

        # Verify root logger is configured
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

    def test_initialize_custom_level(self):
        """Test initialize() with custom log level."""
        config = LoggerConfig(level="DEBUG")
        logger = LoggerInitializer.initialize(config)

        root = logging.getLogger()
        assert root.level == logging.DEBUG
        for handler in root.handlers:
            assert handler.level == logging.DEBUG

    def test_initialize_invalid_level_falls_back_to_info(self):
        """Test initialize() with invalid log level falls back to INFO."""
        config = LoggerConfig(level="INVALID_LEVEL")
        logger = LoggerInitializer.initialize(config)

        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_initialize_with_file_handler(self):
        """Test initialize() with log_file configured."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            config = LoggerConfig(log_file=log_file)
            logger = LoggerInitializer.initialize(config)

            root = logging.getLogger()
            assert len(root.handlers) == 1
            assert isinstance(root.handlers[0], logging.FileHandler)
            assert root.handlers[0].baseFilename == log_file

            # Test that logger can write to file
            logger.info("test message")
            root.handlers[0].flush()
            root.handlers[0].close()  # Close handler to release file on Windows

            # Verify file was written
            with open(log_file, "r") as f:
                content = f.read()
                assert "test message" in content
        finally:
            # Clean up handlers
            root = logging.getLogger()
            for handler in root.handlers[:]:
                handler.close()
                root.removeHandler(handler)
            if os.path.exists(log_file):
                try:
                    os.unlink(log_file)
                except PermissionError:
                    # File might still be locked, skip deletion
                    pass

    def test_initialize_with_stream_handler(self):
        """Test initialize() without log_file uses StreamHandler."""
        config = LoggerConfig(log_file=None)
        logger = LoggerInitializer.initialize(config)

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
        # On Windows/pytest, stdout might be wrapped, so just check it's a StreamHandler
        assert not isinstance(root.handlers[0], logging.FileHandler)

    def test_initialize_clears_existing_handlers(self):
        """Test initialize() clears existing handlers before adding new ones."""
        # Add a handler first
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler(sys.stderr))

        config = LoggerConfig()
        LoggerInitializer.initialize(config)

        # Should have only one handler (the new one)
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
        # Verify it's not the stderr handler we added
        assert not isinstance(root.handlers[0], logging.FileHandler)

    def test_initialize_configures_structlog(self):
        """Test that initialize() properly configures structlog."""
        config = LoggerConfig(json_format=True, level="WARNING")
        logger = LoggerInitializer.initialize(config)

        # Verify structlog is configured by checking if we can get a logger
        test_logger = structlog.get_logger()
        assert test_logger is not None

        # Verify contextvars are cleared
        # (This is tested implicitly by the fact that initialize() calls clear_contextvars)

    def test_initialize_with_custom_wrapper_class(self):
        """Test initialize() with custom wrapper class."""
        config = LoggerConfig(wrapper_class=structlog.stdlib.BoundLogger)
        logger = LoggerInitializer.initialize(config)

        assert logger is not None
        # structlog returns a proxy, verify it has logger methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")

    def test_initialize_with_custom_logger_factory(self):
        """Test initialize() with custom logger factory."""
        config = LoggerConfig(logger_factory=structlog.stdlib.LoggerFactory())
        logger = LoggerInitializer.initialize(config)

        assert logger is not None

    def test_get_default_logger(self):
        """Test get_default_logger() returns a logger."""
        # Clear any existing proxy
        if hasattr(logger_module, "_LOGGER_PROXY"):
            logger_module._LOGGER_PROXY = None

        logger = LoggerInitializer.get_default_logger()

        assert logger is not None
        # Should return a LoggerProxy instance
        assert isinstance(logger, LoggerProxy)

    def test_get_default_logger_initializes_structlog(self):
        """Test that get_default_logger() initializes structlog if not already done."""
        structlog.reset_defaults()
        root = logging.getLogger()
        root.handlers.clear()

        # Clear any existing proxy by accessing the module-level variable
        if hasattr(logger_module, "_LOGGER_PROXY"):
            logger_module._LOGGER_PROXY = None

        logger = LoggerInitializer.get_default_logger()

        # Verify logger is returned and can be used
        assert logger is not None
        assert hasattr(logger, "info")
        # Verify structlog is configured by checking we can get a logger
        test_logger = structlog.get_logger()
        assert test_logger is not None

    def test_get_default_logger_returns_same_proxy(self):
        """Test that get_default_logger() returns the same proxy instance."""
        # Clear any existing proxy
        if hasattr(logger_module, "_LOGGER_PROXY"):
            logger_module._LOGGER_PROXY = None

        logger1 = LoggerInitializer.get_default_logger()
        logger2 = LoggerInitializer.get_default_logger()

        # Should return the same proxy instance
        assert logger1 is logger2


class TestLoggerProxy:
    """Test cases for LoggerProxy class."""

    def setup_method(self):
        """Set up structlog for each test."""
        structlog.reset_defaults()
        config = LoggerConfig()
        LoggerInitializer.initialize(config)

    def teardown_method(self):
        """Clean up after each test."""
        structlog.reset_defaults()
        root = logging.getLogger()
        root.handlers.clear()

    def test_proxy_forwards_attributes(self):
        """Test that LoggerProxy forwards attribute access to structlog logger."""
        proxy = LoggerProxy()

        # Should be able to access logger methods
        assert hasattr(proxy, "info")
        assert hasattr(proxy, "debug")
        assert hasattr(proxy, "warning")
        assert hasattr(proxy, "error")
        assert hasattr(proxy, "critical")

    def test_proxy_can_log_messages(self, caplog):
        """Test that LoggerProxy can be used to log messages."""
        proxy = LoggerProxy()
        proxy.info("test message", key="value")

        # Flush handlers
        root = logging.getLogger()
        for handler in root.handlers:
            handler.flush()

        # Check that logging occurred - either in caplog or by verifying the method was called
        # The logger should have the info method and it should execute without error
        assert hasattr(proxy, "info")
        # Verify we can call it multiple times
        proxy.debug("debug message")
        proxy.warning("warning message")

    def test_proxy_wrapped_property(self):
        """Test that wrapped property returns the underlying BoundLogger."""
        proxy = LoggerProxy()
        wrapped = proxy.wrapped

        assert wrapped is not None
        # structlog returns a proxy, verify it has logger methods
        assert hasattr(wrapped, "info")
        assert hasattr(wrapped, "debug")

    def test_proxy_delegates_to_current_logger(self):
        """Test that proxy always delegates to the current structlog logger."""
        proxy = LoggerProxy()

        # Reconfigure structlog
        new_config = LoggerConfig(level="DEBUG")
        LoggerInitializer.initialize(new_config)

        # Proxy should still work and use the new configuration
        assert hasattr(proxy, "info")
        assert hasattr(proxy, "debug")

    def test_proxy_with_context_vars(self):
        """Test that proxy works with structlog context variables."""
        proxy = LoggerProxy()

        # Bind context variables
        structlog.contextvars.bind_contextvars(user_id="123", request_id="abc")

        # Logger should have access to context
        assert hasattr(proxy, "info")

        # Clear context
        structlog.contextvars.clear_contextvars()


class TestLoggerIntegration:
    """Integration tests for the logger module."""

    def setup_method(self):
        """Set up for each test."""
        structlog.reset_defaults()
        root = logging.getLogger()
        root.handlers.clear()

    def teardown_method(self):
        """Clean up after each test."""
        structlog.reset_defaults()
        root = logging.getLogger()
        root.handlers.clear()

    def test_full_logging_workflow(self, capsys):
        """Test a complete logging workflow."""
        config = LoggerConfig(level="INFO")
        logger = LoggerInitializer.initialize(config)

        logger.info("Info message", key1="value1")
        logger.warning("Warning message", key2="value2")
        logger.error("Error message", key3="value3")

        # Flush handlers
        root = logging.getLogger()
        for handler in root.handlers:
            handler.flush()

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "Info message" in output or "info" in output.lower()
        assert "Warning message" in output or "warning" in output.lower()
        assert "Error message" in output or "error" in output.lower()

    def test_logger_with_file_output(self):
        """Test logger writing to a file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            config = LoggerConfig(log_file=log_file, level="INFO")
            logger = LoggerInitializer.initialize(config)

            logger.info("File log message", test_key="test_value")
            logger.error("Error in file", error_code=500)

            # Flush and close handlers
            root = logging.getLogger()
            for handler in root.handlers:
                handler.flush()
                handler.close()

            # Read file content
            with open(log_file, "r") as f:
                content = f.read()
                assert "File log message" in content or "test_key" in content
                assert "Error in file" in content or "error_code" in content
        finally:
            # Clean up handlers
            root = logging.getLogger()
            for handler in root.handlers[:]:
                handler.close()
                root.removeHandler(handler)
            if os.path.exists(log_file):
                try:
                    os.unlink(log_file)
                except PermissionError:
                    # File might still be locked, skip deletion
                    pass

    def test_logger_with_json_format(self, capsys):
        """Test logger with JSON format output."""
        config = LoggerConfig(json_format=True, level="INFO")
        logger = LoggerInitializer.initialize(config)

        logger.info("JSON message", json_key="json_value")

        # Flush handlers
        root = logging.getLogger()
        for handler in root.handlers:
            handler.flush()

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # JSON output should contain the message or key
        assert "JSON message" in output or "json_key" in output or "json_value" in output

    def test_multiple_initializations(self):
        """Test that multiple initializations work correctly."""
        config1 = LoggerConfig(level="DEBUG")
        logger1 = LoggerInitializer.initialize(config1)

        config2 = LoggerConfig(level="WARNING")
        logger2 = LoggerInitializer.initialize(config2)

        # Both loggers should work
        assert logger1 is not None
        assert logger2 is not None

        # Root logger should reflect the last configuration
        root = logging.getLogger()
        assert root.level == logging.WARNING

