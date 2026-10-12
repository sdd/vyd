#!/usr/bin/env just --justfile

default:
    just --list

fmt:
    cargo fmt --all
    taplo format Cargo.toml release-plz.toml crates/*/Cargo.toml examples/*.toml
    npx --yes prettier@3.6.2 --write README.md DESIGN_PRINCIPLES.md ".github/**/*.{yml,yaml}" .commitlintrc.yaml .pre-commit-config.yaml codecov.yml

check-format:
    cargo fmt --all --check
    taplo format --check Cargo.toml release-plz.toml crates/*/Cargo.toml examples/*.toml
    npx --yes prettier@3.6.2 --check README.md DESIGN_PRINCIPLES.md ".github/**/*.{yml,yaml}" .commitlintrc.yaml .pre-commit-config.yaml codecov.yml

lint:
    cargo clippy --workspace --all-targets --all-features --locked -- -D clippy::correctness -D clippy::suspicious
    RUSTDOCFLAGS='-D rustdoc::broken_intra_doc_links' cargo doc --workspace --all-features --no-deps --locked

test:
    cargo test --workspace --locked
    cargo test --workspace --all-features --locked

coverage:
    cargo llvm-cov --workspace --all-features --locked --lcov --output-path lcov.info
