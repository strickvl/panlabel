# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-01

Improves Hugging Face dataset support and release/distribution readiness.

### Added

- HF zip-style remote split import fallback (for repos shipping `train.zip`/`valid.zip`/`test.zip` layouts)
- Remote payload auto-routing to existing COCO/YOLO/VOC/HF readers after extraction
- Safer zip extraction guards (path traversal protection plus extraction limits)
- Broader remote HF support for split parquet shard layouts in acquisition flow

### Changed

- COCO reader now accepts `info.year` as either integer or string
- Improved remote HF provenance propagation across routed payload formats
- CI/clippy compatibility fixes for feature-gated builds

## [0.2.0] - 2026-03-01

First feature-complete release of panlabel with a full CLI and multi-format support.

### Added

- **CLI commands**: `convert`, `validate`, `stats`, `diff`, `sample`, `list-formats`
- **Format adapters** (read + write):
  - COCO JSON
  - CVAT XML (directory-based)
  - Label Studio JSON
  - Pascal VOC XML (directory-based)
  - TensorFlow Object Detection (TFOD) CSV
  - Ultralytics YOLO (directory-based)
  - IR JSON (canonical intermediate representation)
- **Format auto-detection** from file extension, content structure, and directory layout
- **Lossiness tracking**: explicit conversion reports with stable machine-readable issue codes
- **Conversion reports**: `--report json|text` output with `--allow-lossy` gating
- **Stats command**: annotation counts, label histograms, bounding box quality metrics
- **Diff command**: compare two datasets and report structural differences
- **Sample command**: randomly sample a subset of images from a dataset
- **Validation**: duplicate ID detection, missing reference checks, invalid bbox detection
- **Type-safe IR**: phantom types for coordinate spaces, newtypes for IDs
- **Property tests** (proptest) for all format adapters and cross-format roundtrips
- **Fuzz targets** (cargo-fuzz) for all parsers
- **Criterion benchmarks** for parsing and writing performance
- **CI**: GitHub Actions with fmt, clippy, test across Linux/macOS/Windows
- **Documentation hub**: `docs/` with CLI reference, format behavior, conversion semantics

## [0.1.0] - 2025-01-01

Placeholder initial release on crates.io.

[0.3.0]: https://github.com/strickvl/panlabel/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/strickvl/panlabel/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/strickvl/panlabel/releases/tag/v0.1.0
