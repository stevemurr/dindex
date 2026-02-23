# DIndex Dockerfile
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM rust:1.93-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Cargo files first for dependency caching
COPY Cargo.toml Cargo.lock* ./

# Fetch dependencies (this layer will be cached)
RUN cargo fetch

# Copy actual source code
COPY src ./src

# Build the actual binary
ARG FEATURES=""
RUN if [ -z "$FEATURES" ]; then \
    cargo build --release; \
    else \
    cargo build --release --features "$FEATURES"; \
    fi

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash dindex

# Create data directory
RUN mkdir -p /data && chown dindex:dindex /data

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/dindex /usr/local/bin/dindex

# Set ownership
RUN chown dindex:dindex /usr/local/bin/dindex

# Switch to non-root user
USER dindex

# Set environment variables
ENV RUST_LOG=info
ENV DINDEX_DATA_DIR=/data

# Expose P2P port (QUIC/UDP)
EXPOSE 4001/udp
# Expose TCP fallback port
EXPOSE 4001/tcp
# Expose HTTP API port
EXPOSE 8080/tcp

# Data volume for persistence
VOLUME ["/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Default entrypoint
ENTRYPOINT ["dindex"]

# Default command (show help)
CMD ["--help"]
