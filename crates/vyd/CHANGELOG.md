# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2](https://github.com/sdd/vyd/compare/vyd/0.2.1...vyd/0.2.2) - 2026-10-28

### Added

- add Vortex stage recording and pipeline cancellation

### Other

- simplify pipeline configuration and error handling

## [0.2.1](https://github.com/sdd/vyd/compare/vyd/0.2.0...vyd/0.2.1) - 2026-10-25

### Added

- accept byte-size units for CPU pool limits
- add ability to config cpu pool limits from toml

## [0.2.0](https://github.com/sdd/vyd/compare/vyd/0.1.0...vyd/0.2.0) - 2026-10-25

### Added

- [**breaking**] refactor core into own crate

## [0.1.0](https://github.com/sdd/vyd/releases/tag/vyd/0.1.0) - 2026-10-12

### Added

- *(vyd)* add managed frames and GPU pipeline runtime
- introduce reusable video pipeline

### Other

- separate telemetry and finalize pipeline lifecycle
- generalize frame storage and video integrations
