from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from .service import init_services
from ragprod.infrastructure.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Handles initialization and cleanup of resources.
    """
    # Startup
    logger.info("Starting FastAPI application lifespan")
    
    # Get env path from environment variable if set
    env_path = os.environ.get("RAGPROD_ENV_PATH")
    if env_path:
        logger.info(f"Using env file: {env_path}")
    
    init_services(env_path)
    
    logger.info("FastAPI application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application")

