"""Embedding service for semantic similarity computations.

This module provides a pluggable embedding interface that supports:
- Local models (Qwen3-Embedding-0.6B, Qwen3-Embedding-2B, etc.)
- Remote embedding API endpoints
- Sentence transformers for lightweight use cases

The embedding callable returned by get_embedding_function() conforms to the
NoveltyFilter interface: Callable[[str], Sequence[float]]
"""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts
            **kwargs: Additional model-specific arguments
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    def __len__(self) -> int:
        """Return the embedding dimension (alias for get_dimension)."""
        return self.get_dimension()


class Qwen3EmbeddingModel(EmbeddingModel):
    """Qwen3 Embedding model wrapper for local inference.
    
    Supports Qwen3-Embedding-0.6B, Qwen3-Embedding-2B, and other Qwen embedding variants.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
    ):
        """
        Args:
            model_path: Path to the local model directory or HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            normalize: Whether to normalize embeddings (L2)
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self._model = None
        self._tokenizer = None
        self._dimension = None
    
    def _ensure_model_loaded(self):
        """Lazy load the model on first use."""
        if self._model is not None:
            return
            
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError as e:
            raise RuntimeError(
                "transformers and torch are required for Qwen3 embedding. "
                "Install with: pip install transformers torch"
            ) from e
        
        print(f"Loading Qwen3 embedding model from {self.model_path}...")
        start = time.time()
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        ).to(self.device)
        self._model.eval()
        
        # Get dimension from model config
        self._dimension = self._model.config.hidden_size
        print(f"Model loaded in {time.time() - start:.1f}s, dimension={self._dimension}")
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts into embeddings using mean pooling."""
        self._ensure_model_loaded()
        
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        batch_size = kwargs.get('batch_size', self.batch_size)
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to float32 for numpy compatibility
                embeddings = embeddings.float().cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        self._ensure_model_loaded()
        return self._dimension


class RemoteEmbeddingModel(EmbeddingModel):
    """Remote embedding API wrapper (e.g., OpenAI, SiliconFlow, custom endpoints)."""
    
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        batch_size: int = 32,
    ):
        """
        Args:
            base_url: Base URL for the embedding API
            model_name: Name of the embedding model
            api_key: API key for authentication
            timeout: Request timeout in seconds
            batch_size: Batch size for encoding
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self._dimension = None
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts using remote API."""
        import requests
        
        if isinstance(texts, str):
            texts = [texts]
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        all_embeddings = []
        batch_size = kwargs.get('batch_size', self.batch_size)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "model": self.model_name,
                "input": batch,
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)
    
    def get_dimension(self) -> int:
        """Return the embedding dimension (call encode first or set manually)."""
        if self._dimension is None:
            # Probe with a single text
            emb = self.encode("probe")
            self._dimension = len(emb[0])
        return self._dimension


def get_embedding_function(
    model_type: str = "local",
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 512,
    normalize: bool = True,
    cache_embeddings: bool = True,
) -> Callable[[str], Sequence[float]]:
    """
    Get an embedding function for use with NoveltyFilter and diversity metrics.
    
    Args:
        model_type: Type of embedding model ("local", "remote", "huggingface")
        model_path: Path to local model (for "local" type)
        model_name: Model name (for "remote" or "huggingface" type)
        base_url: API base URL (for "remote" type)
        api_key: API key for remote service
        device: Device for local inference ("cuda" or "cpu")
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        normalize: Whether to normalize embeddings
        cache_embeddings: Whether to cache computed embeddings
        
    Returns:
        A callable that takes a string and returns a list of floats (embedding vector)
        
    Example:
        >>> # Local Qwen3 embedding model
        >>> embed = get_embedding_function(
        ...     model_type="local",
        ...     model_path="/data/liuxingtong/models/Qwen3-Embedding-0.6B"
        ... )
        >>> vec = embed("Hello, world!")
        
        >>> # Remote API
        >>> embed = get_embedding_function(
        ...     model_type="remote",
        ...     base_url="https://api.openai.com/v1",
        ...     model_name="text-embedding-3-small"
        ... )
    """
    cache: dict = {} if cache_embeddings else None
    
    if model_type == "local":
        if not model_path:
            raise ValueError("model_path is required for local embedding model")
        model = Qwen3EmbeddingModel(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
        )
        
        def embed_local(text: str) -> Sequence[float]:
            if cache is not None and text in cache:
                return cache[text]
            vec = model.encode(text)
            result = vec[0].tolist()
            if cache is not None:
                cache[text] = result
            return result
        
        return embed_local
    
    elif model_type == "remote":
        if not base_url or not model_name:
            raise ValueError("base_url and model_name are required for remote embedding")
        model = RemoteEmbeddingModel(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            batch_size=batch_size,
        )
        
        def embed_remote(text: str) -> Sequence[float]:
            if cache is not None and text in cache:
                return cache[text]
            vec = model.encode(text)
            result = vec[0].tolist()
            if cache is not None:
                cache[text] = result
            return result
        
        return embed_remote
    
    elif model_type == "huggingface":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required for huggingface type. "
                "Install with: pip install sentence-transformers"
            )
        
        if not model_name:
            raise ValueError("model_name is required for huggingface type")
        model = SentenceTransformer(model_name, device=device)
        
        def embed_hf(text: str) -> Sequence[float]:
            if cache is not None and text in cache:
                return cache[text]
            vec = model.encode([text], normalize_embeddings=normalize)
            result = vec[0].tolist()
            if cache is not None:
                cache[text] = result
            return result
        
        return embed_hf
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'local', 'remote', or 'huggingface'")


def create_embedding_service_from_config(config: dict) -> Callable[[str], Sequence[float]]:
    """
    Create embedding service from a configuration dictionary.
    
    Configuration format:
        embedding:
            type: local  # local, remote, or huggingface
            model_path: /path/to/model  # for local
            model_name: model-name  # for remote/huggingface
            base_url: https://api.example.com  # for remote
            api_key_env: EMBEDDING_API_KEY  # env var for API key
            device: cuda  # for local/huggingface
            batch_size: 32
            max_length: 512
            normalize: true
    """
    emb_config = config.get("embedding", {})
    
    model_type = emb_config.get("type", "local")
    model_path = emb_config.get("model_path")
    model_name = emb_config.get("model_name")
    base_url = emb_config.get("base_url")
    api_key_env = emb_config.get("api_key_env")
    
    # Get API key from environment if specified
    api_key = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
    
    return get_embedding_function(
        model_type=model_type,
        model_path=model_path,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        device=emb_config.get("device", "cuda"),
        batch_size=emb_config.get("batch_size", 32),
        max_length=emb_config.get("max_length", 512),
        normalize=emb_config.get("normalize", True),
    )


# Default configuration for Qwen3-Embedding-0.6B on the remote server
DEFAULT_MODEL_PATH = "/data/liuxingtong/models/Qwen3-Embedding-0.6B"

def get_default_embedding() -> Callable[[str], Sequence[float]]:
    """Get the default embedding function using Qwen3-Embedding-0.6B."""
    return get_embedding_function(
        model_type="local",
        model_path=DEFAULT_MODEL_PATH,
        device="cuda",
        normalize=True,
    )
