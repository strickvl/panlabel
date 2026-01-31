# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Panlabel is a Rust library and CLI tool for converting between different object detection annotation formats (COCO, TensorFlow Object Detection, etc.). The project is structured as both a library (`src/lib.rs`) and a binary (`src/main.rs`), allowing use as a dependency or standalone CLI.

**Status:** Early development (v0.1.0) - core scaffolding complete, conversion logic pending implementation.

## Common Commands

### Build & Run
```bash
cargo build              # Build debug version
cargo build --release    # Build optimized release version
cargo run                # Run the CLI
cargo run -- -V          # Run with arguments (e.g., version flag)
```

### Testing
```bash
cargo test               # Run all tests (unit + integration)
cargo test --test cli    # Run only CLI integration tests
cargo test runs          # Run a single test by name
```

### Development
```bash
cargo check              # Fast syntax/type checking without building
cargo fmt                # Format code
cargo clippy             # Run linter
cargo doc --open         # Generate and view documentation
```

### Generate Test Data
```bash
# Requires numpy: pip install numpy (or uv pip install numpy)
python scripts/dataset_generator.py --num_images 1000 --annotations_per_image 10 --output_dir ./assets
```

## Architecture

```
src/
├── lib.rs    # Library entry point - exports run() and future conversion functions
└── main.rs   # CLI binary - thin wrapper calling lib.rs

tests/
└── cli.rs    # Integration tests using assert_cmd

scripts/
└── dataset_generator.py  # Generates COCO and TFOD synthetic datasets for testing
```

**Key design:** The CLI binary (`main.rs`) is intentionally minimal—it calls `panlabel::run()` from the library and handles errors. All business logic belongs in `lib.rs` (or modules it imports).

## Annotation Format Reference

The project converts between these formats:

**COCO format** (JSON): `[x, y, width, height]` - absolute pixel coordinates from top-left

**TFOD format** (CSV): `[ymin, xmin, ymax, xmax]` - normalized coordinates (0.0-1.0)

See `scripts/dataset_generator.py` for conversion logic between formats.

## Testing Notes

- Integration tests use `assert_cmd` crate for CLI testing
- Test data goes in `assets/` directory (gitignored)
- Use deterministic seed (42) for reproducible test data generation
