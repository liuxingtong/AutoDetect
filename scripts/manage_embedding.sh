#!/bin/bash
# Script to manage embedding service and test integration
# Usage: ./manage_embedding.sh [start|stop|status|test]

set -e

REMOTE_HOST="liuxingtong@115.182.62.246"
REMOTE_DIR="/data/liuxingtong"
MODEL_PATH="/data/liuxingtong/models/Qwen3-Embedding-0.6B"
PORT=8080

start_service() {
    echo "Starting embedding service on remote server..."
    ssh -o StrictHostKeyChecking=no $REMOTE_HOST << 'ENDSSH'
        cd /data/liuxingtong
        
        # Check if already running
        if pgrep -f "embedding_server" > /dev/null; then
            echo "Embedding service is already running"
            exit 0
        fi
        
        # Start the service
        nohup /data/liuxingtong/envs/llm/bin/python -m autodetect_embedding.embedding_server \
            --model /data/liuxingtong/models/Qwen3-Embedding-0.6B \
            --host 0.0.0.0 \
            --port 8080 \
            --device cuda \
            > /data/liuxingtong/logs/embedding_server.log 2>&1 &
        
        echo "Waiting for service to start..."
        sleep 5
        
        # Check if started successfully
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            echo "✓ Embedding service started successfully"
            echo "  Health: $(curl -s http://localhost:8080/health | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')"
            echo "  Dimension: $(curl -s http://localhost:8080/dimension | python3 -c 'import sys,json; print(json.load(sys.stdin)["dimension"])')"
        else
            echo "✗ Failed to start embedding service"
            tail -20 /data/liuxingtong/logs/embedding_server.log
            exit 1
        fi
ENDSSH
}

stop_service() {
    echo "Stopping embedding service on remote server..."
    ssh -o StrictHostKeyChecking=no $REMOTE_HOST "pkill -f 'embedding_server' || echo 'No service running'"
    echo "✓ Service stopped"
}

check_status() {
    echo "Checking embedding service status..."
    ssh -o StrictHostKeyChecking=no $REMOTE_HOST << 'ENDSSH'
        if pgrep -f "embedding_server" > /dev/null; then
            echo "Service: RUNNING"
            curl -s http://localhost:8080/health | python3 -c '
import sys,json
d=json.load(sys.stdin)
print(f"  Status: {d[\"status\"]}")
print(f"  Model: {d[\"model_path\"]}")
print(f"  Dimension: {d[\"dimension\"]}")
print(f"  Model Loaded: {d[\"model_loaded\"]}")
'
        else
            echo "Service: STOPPED"
        fi
ENDSSH
}

test_service() {
    echo "Testing embedding service..."
    ssh -o StrictHostKeyChecking=no $REMOTE_HOST << 'ENDSSH'
        echo "Testing /health endpoint..."
        curl -s http://localhost:8080/health
        
        echo -e "\n\nTesting /dimension endpoint..."
        curl -s http://localhost:8080/dimension
        
        echo -e "\n\nTesting /embeddings endpoint..."
        curl -s -X POST http://localhost:8080/embeddings \
            -H 'Content-Type: application/json' \
            -d '{"texts": ["Hello, world!", "你好世界"]}' | python3 -c '
import sys,json
d=json.load(sys.stdin)
print(f"Count: {d[\"count\"]}")
print(f"Dimension: {d[\"dimension\"]}")
print(f"Processing time: {d[\"processing_time_ms\"]:.1f}ms")
print(f"First embedding (5 values): {d[\"embeddings\"][0][:5]}")
'
ENDSSH
}

run_tests() {
    echo "Running integration tests on remote server..."
    scp -o StrictHostKeyChecking=no /tmp/test_embedding_remote.py $REMOTE_HOST:/data/liuxingtong/autodetect_embedding/test_embedding_remote.py
    ssh -o StrictHostKeyChecking=no $REMOTE_HOST "/data/liuxingtong/envs/llm/bin/python /data/liuxingtong/autodetect_embedding/test_embedding_remote.py"
}

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start   - Start embedding service on remote server"
    echo "  stop    - Stop embedding service on remote server"
    echo "  status  - Check service status"
    echo "  test    - Quick API test"
    echo "  tests   - Run full integration tests"
    echo "  help    - Show this help message"
}

case "${1:-help}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    status)
        check_status
        ;;
    test)
        test_service
        ;;
    tests)
        run_tests
        ;;
    help|*)
        show_help
        ;;
esac
