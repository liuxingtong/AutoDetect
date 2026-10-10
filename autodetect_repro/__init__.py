"""Small-scale, API-only AutoDetect reproduction."""

__version__ = "0.1.0"

# Re-export embedding-related components for easy access
from .embedding_service import (
    get_embedding_function,
    get_default_embedding,
    create_embedding_service_from_config,
    Qwen3EmbeddingModel,
    EmbeddingModel,
)
from .embedding_client import (
    RemoteEmbeddingClient,
    create_remote_client,
    get_remote_embedding_function,
)
