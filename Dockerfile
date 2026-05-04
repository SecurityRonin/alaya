FROM rust:1.85-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY examples/ examples/
COPY tests/ tests/

RUN cargo build --release --features "mcp llm"

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/alaya-mcp /usr/local/bin/alaya-mcp

ENTRYPOINT ["alaya-mcp"]
