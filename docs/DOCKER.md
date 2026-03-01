# Docker

## Quick Start with Docker Compose

The default `docker-compose.yml` runs DIndex with a vLLM embedding server (requires NVIDIA GPU):

```bash
# Start the full stack (vLLM embeddings + DIndex)
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

This starts:
- **vLLM** serving BGE-M3 embeddings on port 8002 (GPU-accelerated)
- **DIndex** with HTTP API on port 8081 (mapped from container port 8080) and P2P on port 4001

## Building the Image

```bash
docker build -t dindex .
```

## Standalone Docker Usage

```bash
# Initialize configuration
docker run --rm -v dindex-data:/data dindex init --data-dir /data

# Start P2P node
docker run -d --name dindex \
  -p 4001:4001/udp \
  -p 4001:4001/tcp \
  -p 8081:8080 \
  -v dindex-data:/data \
  dindex start --listen /ip4/0.0.0.0/udp/4001/quic-v1

# Search
docker run --rm -v dindex-data:/data dindex search "your query" --top-k 10

# Index local documents
docker run --rm \
  -v dindex-data:/data \
  -v ./documents:/documents:ro \
  dindex index /documents --title "My Documents"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (trace, debug, info, warn, error) |
| `DINDEX_DATA_DIR` | `/data` | Data directory inside container |
| `HF_TOKEN` | — | HuggingFace token (for gated models) |

## Exposed Ports

| Port | Protocol | Description |
|------|----------|-------------|
| 4001 | UDP | P2P QUIC transport (primary) |
| 4001 | TCP | P2P TCP fallback |
| 8081 | TCP | HTTP API (mapped from container 8080) |
