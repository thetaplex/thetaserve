# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Built-in device, model and provider catalogues, overridable from the user config directory.
- Resolver that turns provider x device x model into a launch spec, with a memory-fit check.
- `thetaserve list models|devices|providers`.
- `thetaserve up --dry-run`, printing the resolved profile and the `vllm serve` command.
