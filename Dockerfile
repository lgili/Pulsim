# SpiceLab Multi-stage Dockerfile
# Build: docker build -t spicelab:latest .
# Run:   docker run -p 50051:50051 -p 9090:9090 spicelab:latest

# =============================================================================
# Stage 1: Build dependencies
# =============================================================================
FROM ubuntu:22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Build SpiceLab
# =============================================================================
FROM deps AS builder

WORKDIR /src

# Copy source code
COPY . .

# Configure and build
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPICELAB_BUILD_GRPC=ON \
        -DSPICELAB_BUILD_TESTS=OFF \
        -DSPICELAB_BUILD_EXAMPLES=OFF && \
    cmake --build . --parallel $(nproc) && \
    cmake --install . --prefix /opt/spicelab

# =============================================================================
# Stage 3: Runtime image
# =============================================================================
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    zlib1g \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 1000 -s /bin/false spicelab

# Copy built binaries
COPY --from=builder /opt/spicelab /opt/spicelab

# Add to PATH
ENV PATH="/opt/spicelab/bin:${PATH}"

# Create directories
RUN mkdir -p /data /config && chown -R spicelab:spicelab /data /config

# Switch to non-root user
USER spicelab

# Default configuration
ENV SPICELAB_LISTEN_ADDRESS="0.0.0.0:50051"
ENV SPICELAB_METRICS_PORT="9090"
ENV SPICELAB_WORKERS="0"
ENV SPICELAB_MAX_SESSIONS="64"
ENV SPICELAB_LOG_LEVEL="info"

# Expose ports
EXPOSE 50051/tcp
EXPOSE 9090/tcp

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Default command
ENTRYPOINT ["spicelab_grpc_server"]
CMD ["--listen", "0.0.0.0:50051", "--metrics-port", "9090"]
