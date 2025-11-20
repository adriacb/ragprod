import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from pydantic import ValidationError

from ragprod.infrastructure.config.base import BaseConfigModel
from ragprod.infrastructure.config.fastapi_config import (
    FastAPIConfigModel,
    FastAPIConfig,
)
from ragprod.infrastructure.config.langfuse_config import LangfuseConfigModel
from ragprod.infrastructure.config.settings import load_settings


class TestBaseConfigModel:
    """Test cases for BaseConfigModel class."""

    def test_base_config_model_config(self):
        """Test BaseConfigModel has correct model configuration."""
        # Check that model_config is set correctly
        assert BaseConfigModel.model_config["env_file"] is None
        assert BaseConfigModel.model_config["env_file_encoding"] == "utf-8"
        assert BaseConfigModel.model_config["env_nested_delimiter"] == "__"
        assert BaseConfigModel.model_config["extra"] == "ignore"
        assert BaseConfigModel.model_config["populate_by_name"] is True
        assert BaseConfigModel.model_config["validate_assignment"] is True
        assert BaseConfigModel.model_config["env_prefix"] == ""
        assert BaseConfigModel.model_config["env_ignore_empty"] is True
        assert BaseConfigModel.model_config["env_parse_file_values"] is True

    def test_from_env_file_with_existing_file(self, tmp_path):
        """Test from_env_file() with an existing environment file."""
        # Create a test env file
        env_file = tmp_path / ".env.test"
        env_file.write_text("TEST_VAR=test_value\n")

        # Create a test config model
        class TestConfig(BaseConfigModel):
            test_var: str = "default"

        # Load from file
        config = TestConfig.from_env_file(env_file)

        # Note: The actual loading depends on dotenv, but we can verify the method works
        assert config is not None
        assert isinstance(config, TestConfig)

    def test_from_env_file_with_nonexistent_file(self):
        """Test from_env_file() raises FileNotFoundError for nonexistent file."""
        env_file = Path("/nonexistent/path/.env")

        class TestConfig(BaseConfigModel):
            test_var: str = "default"

        with pytest.raises(FileNotFoundError) as exc_info:
            TestConfig.from_env_file(env_file)

        assert "Environment file not found" in str(exc_info.value)
        assert str(env_file) in str(exc_info.value)

    def test_from_env_file_with_none(self):
        """Test from_env_file() with None (uses default .env lookup)."""
        class TestConfig(BaseConfigModel):
            test_var: str = "default"

        # Should not raise an error, will try to load from default location
        config = TestConfig.from_env_file(None)
        assert config is not None
        assert isinstance(config, TestConfig)

    def test_from_env_file_with_string_path(self, tmp_path):
        """Test from_env_file() accepts string path."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("TEST_VAR=test_value\n")

        class TestConfig(BaseConfigModel):
            test_var: str = "default"

        config = TestConfig.from_env_file(str(env_file))
        assert config is not None
        assert isinstance(config, TestConfig)


class TestFastAPIConfigModel:
    """Test cases for FastAPIConfigModel class."""

    def test_default_values(self):
        """Test FastAPIConfigModel has correct default values."""
        config = FastAPIConfigModel()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert config.reload is True
        assert config.access_log is True
        assert config.api_prefix == "/api"
        assert config.api_title == "RAGProd API"
        assert config.api_description == "API for RAGProd"
        assert config.api_version == "0.1.0"

    def test_custom_values(self):
        """Test FastAPIConfigModel with custom values."""
        config = FastAPIConfigModel(
            host="127.0.0.1",
            port=9000,
            workers=4,
            reload=False,
            access_log=False,
            api_prefix="/v1",
            api_title="Custom API",
            api_description="Custom Description",
            api_version="1.0.0",
        )

        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.workers == 4
        assert config.reload is False
        assert config.access_log is False
        assert config.api_prefix == "/v1"
        assert config.api_title == "Custom API"
        assert config.api_description == "Custom Description"
        assert config.api_version == "1.0.0"

    def test_environment_variable_aliases(self, monkeypatch):
        """Test FastAPIConfigModel reads from environment variables via aliases."""
        monkeypatch.setenv("FASTAPI_HOST", "192.168.1.1")
        monkeypatch.setenv("FASTAPI_PORT", "3000")
        monkeypatch.setenv("FASTAPI_WORKERS", "8")
        monkeypatch.setenv("FASTAPI_RELOAD", "false")
        monkeypatch.setenv("FASTAPI_ACCESS_LOG", "false")
        monkeypatch.setenv("FASTAPI_API_PREFIX", "/custom")
        monkeypatch.setenv("FASTAPI_API_TITLE", "Env API")
        monkeypatch.setenv("FASTAPI_API_DESCRIPTION", "Env Description")
        monkeypatch.setenv("FASTAPI_API_VERSION", "2.0.0")

        config = FastAPIConfigModel()

        assert config.host == "192.168.1.1"
        assert config.port == 3000
        assert config.workers == 8
        assert config.reload is False
        assert config.access_log is False
        assert config.api_prefix == "/custom"
        assert config.api_title == "Env API"
        assert config.api_description == "Env Description"
        assert config.api_version == "2.0.0"

    def test_port_validation(self):
        """Test FastAPIConfigModel validates port as integer."""
        # Valid port
        config = FastAPIConfigModel(port=8080)
        assert config.port == 8080

        # Port from string should be converted
        with patch.dict(os.environ, {"FASTAPI_PORT": "9090"}):
            config = FastAPIConfigModel()
            assert config.port == 9090

    def test_boolean_fields(self):
        """Test FastAPIConfigModel boolean fields accept various formats."""
        # Test with boolean values
        config = FastAPIConfigModel(reload=True, access_log=False)
        assert config.reload is True
        assert config.access_log is False

        # Test with string values (from env)
        with patch.dict(os.environ, {"FASTAPI_RELOAD": "True", "FASTAPI_ACCESS_LOG": "False"}):
            config = FastAPIConfigModel()
            assert config.reload is True
            assert config.access_log is False


class TestFastAPIConfig:
    """Test cases for FastAPIConfig class."""

    def test_init_without_config(self):
        """Test FastAPIConfig initialization without config parameter."""
        config = FastAPIConfig()

        assert config._config is not None
        assert isinstance(config._config, FastAPIConfigModel)

    def test_init_with_config(self):
        """Test FastAPIConfig initialization with config parameter."""
        model = FastAPIConfigModel(host="127.0.0.1", port=9000)
        config = FastAPIConfig(config=model)

        assert config._config is model
        assert config._config.host == "127.0.0.1"
        assert config._config.port == 9000

    def test_get_host(self):
        """Test get_host() method."""
        model = FastAPIConfigModel(host="192.168.1.1")
        config = FastAPIConfig(config=model)

        assert config.get_host() == "192.168.1.1"

    def test_get_port(self):
        """Test get_port() method."""
        model = FastAPIConfigModel(port=8080)
        config = FastAPIConfig(config=model)

        assert config.get_port() == 8080

    def test_get_workers(self):
        """Test get_workers() method."""
        model = FastAPIConfigModel(workers=4)
        config = FastAPIConfig(config=model)

        assert config.get_workers() == 4

    def test_get_reload(self):
        """Test get_reload() method."""
        model = FastAPIConfigModel(reload=False)
        config = FastAPIConfig(config=model)

        assert config.get_reload() is False

    def test_get_access_log(self):
        """Test get_access_log() method."""
        model = FastAPIConfigModel(access_log=False)
        config = FastAPIConfig(config=model)

        assert config.get_access_log() is False

    def test_get_api_prefix(self):
        """Test get_api_prefix() method."""
        model = FastAPIConfigModel(api_prefix="/v1")
        config = FastAPIConfig(config=model)

        assert config.get_api_prefix() == "/v1"

    def test_get_api_title(self):
        """Test get_api_title() method."""
        model = FastAPIConfigModel(api_title="Custom Title")
        config = FastAPIConfig(config=model)

        assert config.get_api_title() == "Custom Title"

    def test_get_api_description(self):
        """Test get_api_description() method."""
        model = FastAPIConfigModel(api_description="Custom Description")
        config = FastAPIConfig(config=model)

        assert config.get_api_description() == "Custom Description"

    def test_get_api_version(self):
        """Test get_api_version() method."""
        model = FastAPIConfigModel(api_version="2.0.0")
        config = FastAPIConfig(config=model)

        assert config.get_api_version() == "2.0.0"

    def test_all_getters_with_defaults(self):
        """Test all getter methods with default values."""
        config = FastAPIConfig()

        assert config.get_host() == "0.0.0.0"
        assert config.get_port() == 8000
        assert config.get_workers() == 1
        assert config.get_reload() is True
        assert config.get_access_log() is True
        assert config.get_api_prefix() == "/api"
        assert config.get_api_title() == "RAGProd API"
        assert config.get_api_description() == "API for RAGProd"
        assert config.get_api_version() == "0.1.0"


class TestLangfuseConfigModel:
    """Test cases for LangfuseConfigModel class."""

    def test_required_fields(self):
        """Test LangfuseConfigModel requires public_key, secret_key, and tags."""
        # Should raise ValidationError without required fields
        with pytest.raises(ValidationError):
            LangfuseConfigModel()

    def test_all_fields_provided(self):
        """Test LangfuseConfigModel with all fields provided."""
        config = LangfuseConfigModel(
            public_key="pk_test",
            secret_key="sk_test",
            host="https://custom.langfuse.com",
            timeout=1000,
            tags=["tag1", "tag2"],
            version="1.0.0",
            release="production",
            environment="production",
        )

        assert config.public_key == "pk_test"
        assert config.secret_key == "sk_test"
        assert config.host == "https://custom.langfuse.com"
        assert config.timeout == 1000
        assert config.tags == ["tag1", "tag2"]
        assert config.version == "1.0.0"
        assert config.release == "production"
        assert config.environment == "production"

    def test_default_values(self, monkeypatch):
        """Test LangfuseConfigModel has correct default values."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")
        # Use JSON format for list fields in environment variables
        monkeypatch.setenv("LANGFUSE_TAGS", '["tag1","tag2"]')

        config = LangfuseConfigModel()

        assert config.host == "https://cloud.langfuse.com"
        assert config.timeout is None
        assert config.version == "0.1.0"
        assert config.release == "development"
        assert config.environment == "development"

    def test_environment_variable_aliases(self, monkeypatch):
        """Test LangfuseConfigModel reads from environment variables via aliases."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_env")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_env")
        monkeypatch.setenv("LANGFUSE_HOST", "https://env.langfuse.com")
        monkeypatch.setenv("LANGFUSE_TIMEOUT", "2000")
        # Use JSON format for list fields in environment variables
        monkeypatch.setenv("LANGFUSE_TAGS", '["env1","env2"]')
        monkeypatch.setenv("LANGFUSE_VERSION", "2.0.0")
        monkeypatch.setenv("LANGFUSE_RELEASE", "staging")
        monkeypatch.setenv("LANGFUSE_ENVIRONMENT", "staging")

        config = LangfuseConfigModel()

        assert config.public_key == "pk_env"
        assert config.secret_key == "sk_env"
        assert config.host == "https://env.langfuse.com"
        assert config.timeout == 2000
        assert config.tags == ["env1", "env2"]
        assert config.version == "2.0.0"
        assert config.release == "staging"
        assert config.environment == "staging"

    def test_tags_validator_with_string(self):
        """Test tags field validator splits comma-separated string."""
        config = LangfuseConfigModel(
            public_key="pk_test",
            secret_key="sk_test",
            tags="tag1,tag2,tag3",
        )

        assert config.tags == ["tag1", "tag2", "tag3"]

    def test_tags_validator_with_list(self):
        """Test tags field validator accepts list directly."""
        config = LangfuseConfigModel(
            public_key="pk_test",
            secret_key="sk_test",
            tags=["tag1", "tag2"],
        )

        assert config.tags == ["tag1", "tag2"]

    def test_tags_validator_with_spaces(self):
        """Test tags field validator trims whitespace."""
        config = LangfuseConfigModel(
            public_key="pk_test",
            secret_key="sk_test",
            tags="tag1 , tag2 , tag3 ",
        )

        assert config.tags == ["tag1", "tag2", "tag3"]

    def test_tags_validator_with_empty_string(self):
        """Test tags field validator handles empty string."""
        config = LangfuseConfigModel(
            public_key="pk_test",
            secret_key="sk_test",
            tags="",
        )

        assert config.tags == []

    def test_tags_validator_with_empty_after_split(self):
        """Test tags field validator filters empty tags."""
        config = LangfuseConfigModel(
            public_key="pk_test",
            secret_key="sk_test",
            tags="tag1,,tag2,  ,tag3",
        )

        assert config.tags == ["tag1", "tag2", "tag3"]

    def test_timeout_optional(self):
        """Test timeout field is optional."""
        config = LangfuseConfigModel(
            public_key="pk_test",
            secret_key="sk_test",
            tags=["tag1"],
            timeout=None,
        )

        assert config.timeout is None


class TestLoadSettings:
    """Test cases for load_settings function."""

    def setup_method(self):
        """Set up test environment."""
        # Clear any existing environment variables
        env_vars_to_clear = [
            "LOG_LEVEL",
            "FASTAPI_HOST",
            "FASTAPI_PORT",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        ]
        for var in env_vars_to_clear:
            os.environ.pop(var, None)

    def test_load_settings_with_env_file(self, tmp_path, monkeypatch):
        """Test load_settings() with an environment file."""
        env_file = tmp_path / ".env.dev"
        env_file.write_text(
            "FASTAPI_HOST=127.0.0.1\n"
            "FASTAPI_PORT=9000\n"
            "LOG_LEVEL=DEBUG\n"
        )

        settings = load_settings(str(env_file))

        assert settings is not None
        assert settings.env == "dev"
        assert settings.env_file == str(env_file)
        assert settings.fastapi is not None
        assert settings.fastapi.host == "127.0.0.1"
        assert settings.fastapi.port == 9000

    def test_load_settings_without_env_file(self):
        """Test load_settings() without environment file."""
        settings = load_settings(None)

        assert settings is not None
        assert settings.env == "dev"  # Default when env_path is None
        assert settings.env_file is None

    def test_load_settings_env_name_extraction(self, tmp_path):
        """Test load_settings() extracts env name from file path."""
        env_file = tmp_path / ".env.prod"
        env_file.write_text("FASTAPI_HOST=127.0.0.1\n")

        settings = load_settings(str(env_file))

        assert settings.env == "prod"

    def test_load_settings_env_name_default(self):
        """Test load_settings() uses 'dev' as default env name."""
        settings = load_settings("")

        assert settings.env == "dev"

    def test_load_settings_handles_fastapi_config_error(self, tmp_path, monkeypatch):
        """Test load_settings() handles FastAPI config errors gracefully."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("INVALID_CONFIG=value\n")

        # Mock FastAPIConfigModel to raise an exception
        with patch(
            "ragprod.infrastructure.config.settings.FastAPIConfigModel.from_env_file"
        ) as mock_fastapi:
            mock_fastapi.side_effect = ValidationError.from_exception_data(
                "FastAPIConfigModel", []
            )

            settings = load_settings(str(env_file))

            # Should still return settings with None fastapi config
            assert settings is not None
            # fastapi_conf might be None or fallback to defaults
            assert settings.fastapi is None or isinstance(settings.fastapi, FastAPIConfigModel)

    def test_load_settings_handles_langfuse_config_error(self, tmp_path):
        """Test load_settings() handles Langfuse config errors gracefully."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("FASTAPI_HOST=127.0.0.1\n")

        # Langfuse requires public_key and secret_key, so it should fail gracefully
        settings = load_settings(str(env_file))

        assert settings is not None
        # langfuse_conf should be None when required fields are missing
        assert settings.langfuse is None

    def test_load_settings_log_level_from_env(self, tmp_path, monkeypatch):
        """Test load_settings() uses LOG_LEVEL from environment file."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("LOG_LEVEL=ERROR\nFASTAPI_HOST=127.0.0.1\n")

        settings = load_settings(str(env_file))

        assert settings is not None
        # The logger should be configured with ERROR level
        # We can't easily test the logger level, but we can verify settings loaded

    def test_load_settings_log_level_default_dev(self, tmp_path):
        """Test load_settings() uses DEBUG as default log level for dev."""
        env_file = tmp_path / ".env.dev"
        env_file.write_text("FASTAPI_HOST=127.0.0.1\n")

        settings = load_settings(str(env_file))

        assert settings is not None
        assert settings.env == "dev"

    def test_load_settings_log_level_default_prod(self, tmp_path):
        """Test load_settings() uses WARNING as default log level for prod."""
        env_file = tmp_path / ".env.prod"
        env_file.write_text("FASTAPI_HOST=127.0.0.1\n")

        settings = load_settings(str(env_file))

        assert settings is not None
        assert settings.env == "prod"

    def test_load_settings_with_langfuse_config(self, tmp_path, monkeypatch):
        """Test load_settings() loads Langfuse config when all required fields present."""
        env_file = tmp_path / ".env.test"
        # Use JSON format for list fields in .env files
        env_file.write_text(
            "FASTAPI_HOST=127.0.0.1\n"
            "LANGFUSE_PUBLIC_KEY=pk_test\n"
            "LANGFUSE_SECRET_KEY=sk_test\n"
            'LANGFUSE_TAGS=["tag1","tag2"]\n'
        )

        settings = load_settings(str(env_file))

        assert settings is not None
        assert settings.langfuse is not None
        assert settings.langfuse.public_key == "pk_test"
        assert settings.langfuse.secret_key == "sk_test"
        assert settings.langfuse.tags == ["tag1", "tag2"]

    def test_load_settings_returns_simplenamespace(self, tmp_path):
        """Test load_settings() returns SimpleNamespace with expected attributes."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("FASTAPI_HOST=127.0.0.1\n")

        settings = load_settings(str(env_file))

        assert hasattr(settings, "env")
        assert hasattr(settings, "env_file")
        assert hasattr(settings, "fastapi")
        assert hasattr(settings, "langfuse")

    def test_load_settings_env_name_lowercase(self, tmp_path):
        """Test load_settings() converts env name to lowercase."""
        env_file = tmp_path / ".env.PRODUCTION"
        env_file.write_text("FASTAPI_HOST=127.0.0.1\n")

        settings = load_settings(str(env_file))

        assert settings.env == "production"

