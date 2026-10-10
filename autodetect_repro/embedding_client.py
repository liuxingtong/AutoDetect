"""Remote embedding client for calling embedding service over HTTP.

This module provides a client that connects to a remote embedding service,
enabling the AutoDetect framework to use GPU-accelerated embedding models
running on a remote server.

Usage:
    # Create client
    client = RemoteEmbeddingClient("http://115.182.62.246:8080")
    
    # Get embedding for a single text
    vec = client.encode("Hello, world!")
    
    # Get embeddings for multiple texts
    vecs = client.encode_batch(["text1", "text2", "text3"])
    
    # Use with NoveltyFilter
    from autodetect_repro.novelty import NoveldingFilter
    filterer = NoveltyFilter(embedding=client.encode)
"""

import os
from typing import Dict, List, Optional, Sequence, Union

import requests


class RemoteEmbeddingClient:
    """Client for remote embedding service."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        batch_size: int = 32,
        cache_embeddings: bool = True,
    ):
        """
        Args:
            base_url: Base URL of the embedding service (e.g., "http://115.182.62.246:8080")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            batch_size: Batch size for encoding
            cache_embeddings: Whether to cache computed embeddings
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY")
        self.timeout = timeout
        self.batch_size = batch_size
        self._cache: Dict[str, List[float]] = {} if cache_embeddings else None
        self._dimension: Optional[int] = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def encode(self, text: str) -> Sequence[float]:
        """
        Encode a single text into an embedding vector.
        
        Args:
            text: Input text string
            
        Returns:
            List of floats representing the embedding vector
        """
        if self._cache is not None and text in self._cache:
            return self._cache[text]
        
        vecs = self.encode_batch([text])
        result = vecs[0]
        
        if self._cache is not None:
            self._cache[text] = result
        
        return result
    
    def encode_batch(self, texts: List[str]) -> List[Sequence[float]]:
        """
        Encode multiple texts into embedding vectors.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []
        
        # Filter out cached texts
        uncached_texts = []
        uncached_indices = []
        
        if self._cache is not None:
            for i, text in enumerate(texts):
                if text in self._cache:
                    continue
                uncached_texts.append(text)
                uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Build result list
        result = [None] * len(texts)
        
        # Fill cached results
        if self._cache is not None:
            for i, text in enumerate(texts):
                if text in self._cache:
                    result[i] = self._cache[text]
        
        # Fetch uncached texts in batches
        for batch_start in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[batch_start:batch_start + self.batch_size]
            
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=self._get_headers(),
                    json={"texts": batch},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                data = response.json()
                embeddings = data.get("embeddings", [])
                
                # Store dimension if not set
                if self._dimension is None and embeddings:
                    self._dimension = len(embeddings[0])
                
                # Update cache and result
                for j, text in enumerate(batch):
                    vec = embeddings[j] if j < len(embeddings) else None
                    if vec is not None:
                        idx = uncached_indices[batch_start + j]
                        result[idx] = vec
                        if self._cache is not None:
                            self._cache[text] = vec
                            
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to encode texts: {e}")
        
        # Handle any remaining None values (shouldn't happen)
        result = [r if r is not None else [0.0] * (self._dimension or 1024) for r in result]
        
        return result
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            The embedding dimension (e.g., 1024 for Qwen3-Embedding-0.6B)
        """
        if self._dimension is not None:
            return self._dimension
        
        try:
            response = requests.get(
                f"{self.base_url}/dimension",
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            self._dimension = data.get("dimension", 1024)
            return self._dimension
            
        except requests.exceptions.RequestException:
            # Fallback: probe with a single text
            vec = self.encode("probe")
            self._dimension = len(vec)
            return self._dimension
    
    def health_check(self) -> bool:
        """
        Check if the embedding service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._get_headers(),
                timeout=5,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache is not None:
            self._cache.clear()
    
    def __len__(self) -> int:
        """Return the embedding dimension."""
        return self.get_dimension()
    
    def __repr__(self) -> str:
        return f"RemoteEmbeddingClient(base_url={self.base_url})"


def create_remote_client(
    base_url: str,
    api_key: Optional[str] = None,
) -> RemoteEmbeddingClient:
    """
    Factory function to create a remote embedding client.
    
    Args:
        base_url: Base URL of the embedding service
        api_key: Optional API key
        
    Returns:
        Configured RemoteEmbeddingClient instance
    """
    return RemoteEmbeddingClient(
        base_url=base_url,
        api_key=api_key,
        batch_size=32,
        cache_embeddings=True,
    )


# Default client pointing to the remote server
DEFAULT_REMOTE_URL = "http://115.182.62.246:8080"

def get_remote_embedding_function(
    base_url: str = DEFAULT_REMOTE_URL,
    api_key: Optional[str] = None,
) -> callable:
    """
    Get an embedding function that calls a remote service.
    
    The returned function conforms to the NoveltyFilter interface:
    Callable[[str], Sequence[float]]
    
    Args:
        base_url: URL of the remote embedding service
        api_key: Optional API key
        
    Returns:
        A callable embedding function
    """
    client = create_remote_client(base_url, api_key)
    return client.encode
