#!/bin/bash
set -euo pipefail

echo "=== DIndex P2P Cluster Integration Tests ==="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

# Check GPU (optional)
COMPOSE_PROFILES=""
if nvidia-smi &> /dev/null; then
    echo "GPU detected, will use local embedding server"
    export COMPOSE_PROFILES="with-embeddings"
else
    echo "No GPU detected, checking DINDEX_EMBEDDING_ENDPOINT..."
    if [ -z "${DINDEX_EMBEDDING_ENDPOINT:-}" ]; then
        echo "ERROR: No GPU and DINDEX_EMBEDDING_ENDPOINT not set"
        echo ""
        echo "Either:"
        echo "  1. Run on a machine with a GPU (embedding server will start automatically)"
        echo "  2. Set DINDEX_EMBEDDING_ENDPOINT to an external embedding server"
        echo "     Example: export DINDEX_EMBEDDING_ENDPOINT=http://host:8000/v1/embeddings"
        exit 1
    fi
    echo "Using external embedding server: $DINDEX_EMBEDDING_ENDPOINT"
fi

echo ""
echo "Running P2P cluster tests..."
echo ""

# Run tests
RUST_LOG=info cargo test --test p2p_cluster -- --ignored --nocapture --test-threads=1 "$@"
