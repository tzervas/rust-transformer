# Multi-stage build for security and minimal attack surface
FROM rust:1.82-slim as builder

# Create app directory
WORKDIR /app

# Copy dependency files first for better caching
COPY Cargo.toml ./

# Create a dummy main.rs to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --target x86_64-unknown-linux-gnu
RUN rm -rf src

# Copy actual source code
COPY src ./src

# Build the application
RUN cargo build --release --target x86_64-unknown-linux-gnu

# Runtime stage - minimal distroless image for security
FROM gcr.io/distroless/cc-debian12

# Create non-root user
USER nonroot:nonroot

# Copy the binary from builder stage
COPY --from=builder /app/target/x86_64-unknown-linux-gnu/release/rust-transformer /usr/local/bin/transformer

# Set working directory
WORKDIR /app

# Run as non-root user
ENTRYPOINT ["/usr/local/bin/transformer"]