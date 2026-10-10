#!/bin/bash
# Start script for embedding service

MODEL_PATH="${1:-/data/liuxingtong/models/Qwen3-Embedding-0.6B}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8080}"
DEVICE="${4:-cuda}"

echo "Starting embedding service..."
echo "  Model: $MODEL_PATH"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Device: $DEVICE"

cd /data/liuxingtong

# Use the llm environment which has all dependencies
/data/liuxingtong/envs/llm/bin/python -m autodetect_embedding.embedding_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    2>&1 | tee /data/liuxingtong/logs/embedding_server.log
