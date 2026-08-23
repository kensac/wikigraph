FROM rust:1-slim AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --bin serve

FROM gcr.io/distroless/cc-debian12
COPY --from=build /app/target/release/serve /serve
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD ["/serve", "--check"]
ENTRYPOINT ["/serve", "--graph", "/data/graph.bin"]
