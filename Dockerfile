# Multi-stage Dockerfile for running embedding provider.

# Rust build software. This stage changes only when the base image or installed
# system packages change, so source edits do not invalidate it.
FROM rust:bookworm AS rust-system

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && \
    apt-get install -yqq --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN mkdir -p .cargo && \
    { echo '[unstable]'; echo 'cargo-lints = true'; } > .cargo/config.toml

# The pinned nightly is isolated from source and dependency changes.
FROM rust-system AS rust-toolchain

COPY rust-toolchain.toml ./
RUN NIGHTLY="$(sed -nE 's/^channel[[:space:]]*=[[:space:]]*"([^"]+)"/\1/p' rust-toolchain.toml)" \
    && rustup install "$NIGHTLY" && rustup default "$NIGHTLY"
RUN cargo +nightly-2026-07-30 install cargo-chef --version 0.1.77 --locked

# The recipe contains only Cargo manifests and the lockfile. Source changes may
# rerun the planner, but they do not change the dependency-layer input.
FROM rust-toolchain AS cargo-planner

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo +nightly-2026-07-30 chef prepare --recipe-path recipe.json

# Download locked registry and Git dependencies before application sources are
# copied into the build stage.
FROM rust-toolchain AS cargo-dependencies

COPY --from=cargo-planner /build/recipe.json ./recipe.json
RUN cargo +nightly-2026-07-30 chef cook --recipe-path recipe.json --no-build && \
    cargo +nightly-2026-07-30 fetch --locked

FROM cargo-dependencies AS rust-builder

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

#VOLUME ["/cargo-target"]
# RUN CARGO_TARGET_DIR=/cargo-target CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 MAKEFLAGS=-j1 CMAKE_BUILD_PARALLEL_LEVEL=1 \
#     cargo +nightly-2026-07-30 build \
#     -p chelix-embedding-service \
#     --release \
#     -j 1

RUN cargo +nightly-2026-07-30 build \
    -p chelix-embedding-service \
    --release

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl procps \
  && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /build/target/release/chelix-embedding-service /usr/local/bin/chelix-embedding-service
