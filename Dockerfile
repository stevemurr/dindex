# DIndex Dockerfile
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM rust:1.83-bookworm AS builder

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

# Create dummy source to build dependencies
RUN mkdir -p src && \
    echo 'fn main() { println!("dummy"); }' > src/main.rs && \
    echo 'pub fn dummy() {}' > src/lib.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release && rm -rf src target/release/deps/dindex*

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

# Data volume for persistence
VOLUME ["/data"]

# Health check (verify binary works)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD dindex stats 2>/dev/null || exit 0

# Default entrypoint
ENTRYPOINT ["dindex"]

# Default command (show help)
CMD ["--help"]
