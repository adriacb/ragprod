from typing import Optional, List
from pydantic import Field, field_validator
from .base import BaseConfigModel


class LangfuseConfigModel(BaseConfigModel):
    """Pydantic model for Langfuse configuration."""

    public_key: str = Field(..., alias="LANGFUSE_PUBLIC_KEY")
    secret_key: str = Field(..., alias="LANGFUSE_SECRET_KEY")
    host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")
    timeout: Optional[int] = Field(default=None, alias="LANGFUSE_TIMEOUT")
    tags: List[str] = Field(..., alias="LANGFUSE_TAGS")
    version: str = Field(default="0.1.0", alias="LANGFUSE_VERSION")
    release: str = Field(default="development", alias="LANGFUSE_RELEASE")
    environment: str = Field(default="development", alias="LANGFUSE_ENVIRONMENT")

    @field_validator("tags", mode="before")
    def split_tags(cls, v):
        if isinstance(v, str):
            return [tag.strip() for tag in v.split(",") if tag.strip()]
        return v