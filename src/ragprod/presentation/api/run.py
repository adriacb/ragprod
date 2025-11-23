import uvicorn
import argparse
from pathlib import Path


def main():
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(description="Run RAGProd FastAPI server")
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Path to .env file (default: None, uses process environment)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Set env_path in app state if provided
    if args.env:
        env_path = Path(args.env)
        if not env_path.exists():
            print(f"Warning: Env file not found: {env_path}")
        else:
            # We'll pass this via environment variable since we can't easily set app.state before lifespan
            import os
            os.environ["RAGPROD_ENV_PATH"] = str(env_path.absolute())
    
    # Run the server
    uvicorn.run(
        "ragprod.presentation.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
