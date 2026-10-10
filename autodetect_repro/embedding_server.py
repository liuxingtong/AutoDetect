#!/usr/bin/env python
"""Remote embedding service with REST API.

This service wraps embedding models (Qwen3-Embedding-0.6B, etc.) behind a simple
HTTP API that can be called from any machine, including the local AutoDetect framework.

Usage:
    # Start server on remote machine (default model path is the
    # reproduction host's /data/liuxingtong/models/Qwen3-Embedding-0.6B;
    # override with AUTODETECT_EMBEDDING_MODEL or --model).
    python -m autodetect_repro.embedding_server --model /path/to/Qwen3-Embedding-0.6B

    # Or with custom settings
    python -m autodetect_repro.embedding_server \
        --model /path/to/Qwen3-Embedding-0.6B \
        --host 0.0.0.0 --port 8080 --device cuda

API Endpoints:
    POST /embeddings  - Encode texts into embeddings
    GET  /health      - Health check
    GET  /dimension   - Get embedding dimension
"""

import argparse
import json
import os
import time
from typing import List, Optional, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .embedding_service import (
    EmbeddingModel,
    Qwen3EmbeddingModel,
    get_embedding_function,
)


# Pydantic models for request/response
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to encode", min_length=1)
    batch_size: Optional[int] = Field(32, description="Batch size for encoding")
    normalize: Optional[bool] = Field(True, description="Whether to normalize embeddings")


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    dimension: int


class DimensionResponse(BaseModel):
    dimension: int
    model_path: str


# Global model instance
_model: Optional[EmbeddingModel] = None
_model_path: str = ""


def create_app(model_path: str, device: str = "cuda", max_length: int = 512) -> FastAPI:
    """Create FastAPI application with embedding model."""
    
    app = FastAPI(
        title="Embedding Service",
        description="REST API for embedding model inference",
        version="1.0.0",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    global _model, _model_path
    
    @app.on_event("startup")
    async def load_model():
        global _model, _model_path
        print(f"Loading embedding model from {model_path}...")
        start = time.time()
        
        _model = Qwen3EmbeddingModel(
            model_path=model_path,
            device=device,
            max_length=max_length,
            normalize=True,
        )
        _model_path = model_path
        
        # Force model loading
        _model._ensure_model_loaded()
        
        print(f"Model loaded in {time.time() - start:.1f}s, dimension={_model.get_dimension()}")
    
    @app.post("/embeddings", response_model=EmbeddingResponse)
    async def encode_embeddings(request: EmbeddingRequest):
        """Encode texts into embeddings."""
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start = time.time()
        try:
            embeddings = _model.encode(
                request.texts,
                batch_size=request.batch_size,
            )
            
            # Convert to list
            embedding_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
            processing_time = (time.time() - start) * 1000
            
            return EmbeddingResponse(
                embeddings=embedding_list,
                dimension=_model.get_dimension(),
                count=len(request.texts),
                processing_time_ms=processing_time,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        model_loaded = _model is not None
        dimension = _model.get_dimension() if model_loaded else 0
        
        return HealthResponse(
            status="healthy" if model_loaded else "loading",
            model_loaded=model_loaded,
            model_path=_model_path,
            dimension=dimension,
        )
    
    @app.get("/dimension", response_model=DimensionResponse)
    async def get_dimension():
        """Get embedding dimension."""
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return DimensionResponse(
            dimension=_model.get_dimension(),
            model_path=_model_path,
        )
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Embedding Service")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.environ.get(
            "AUTODETECT_EMBEDDING_MODEL",
            "/data/liuxingtong/models/Qwen3-Embedding-0.6B",
        ),
        help="Path to embedding model (or set AUTODETECT_EMBEDDING_MODEL)"
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port to bind to"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    
    args = parser.parse_args()
    
    app = create_app(
        model_path=args.model,
        device=args.device,
        max_length=args.max_length,
    )
    
    print(f"Starting embedding service...")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Endpoint: http://{args.host}:{args.port}")
    print(f"  API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
