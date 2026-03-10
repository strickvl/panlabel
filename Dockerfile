# ---- Build stage ----
FROM rust:slim AS builder

# Install build dependencies needed by parquet/arrow C libs (zstd, lz4, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        cmake \
        make \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for better layer caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy main/lib so Cargo can fetch and cache dependencies
RUN mkdir src && \
    echo 'fn main() {}' > src/main.rs && \
    echo '' > src/lib.rs && \
    cargo build --release --features hf 2>/dev/null || true && \
    rm -rf src

# Copy the real source code
COPY src/ src/

# Build the actual binary
RUN cargo build --release --features hf && \
    strip target/release/panlabel

# ---- Runtime stage ----
FROM debian:bookworm-slim AS runtime

# Install minimal runtime dependencies (CA certs for HTTPS/HF downloads)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/panlabel /usr/local/bin/panlabel

ENTRYPOINT ["panlabel"]

# OCI labels
LABEL org.opencontainers.image.source="https://github.com/strickvl/panlabel" \
      org.opencontainers.image.title="panlabel" \
      org.opencontainers.image.description="The universal annotation converter" \
      org.opencontainers.image.version="0.3.0"
