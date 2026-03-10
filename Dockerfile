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

# Create dummy source files so Cargo can validate the manifest and cache dependencies.
# The bench stub satisfies the [[bench]] entry in Cargo.toml (benches/ is dockerignored).
RUN mkdir -p src benches && \
    echo 'fn main() {}' > src/main.rs && \
    echo '' > src/lib.rs && \
    echo 'fn main() {}' > benches/microbenches.rs && \
    cargo build --release --features hf --bin panlabel 2>/dev/null || true && \
    rm -rf src benches

# Copy the real source code and create bench stub for manifest validation
COPY src/ src/
RUN mkdir -p benches && echo 'fn main() {}' > benches/microbenches.rs

# Build the actual binary
RUN cargo build --release --features hf --bin panlabel && \
    strip target/release/panlabel

# ---- Runtime stage ----
FROM debian:bookworm-slim AS runtime

# Install minimal runtime dependencies (CA certs for HTTPS/HF downloads)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/panlabel /usr/local/bin/panlabel

ENTRYPOINT ["panlabel"]

ARG VERSION=dev
LABEL org.opencontainers.image.source="https://github.com/strickvl/panlabel" \
      org.opencontainers.image.title="panlabel" \
      org.opencontainers.image.description="The universal annotation converter" \
      org.opencontainers.image.version="${VERSION}"
