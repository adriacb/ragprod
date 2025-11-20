from ragprod.infrastructure.monitoring.langfuse import (
    create_langfuse_callback, 
    CallbackHandler, 
    Optional
)

def get_langfuse_callback(config) -> Optional[CallbackHandler]:
    """Get a Langfuse callback handler with the given settings.
    Returns None if credentials are not provided."""
    return create_langfuse_callback(config)