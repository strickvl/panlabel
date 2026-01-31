# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Panlabel is a Rust library and CLI tool for converting between different object detection annotation formats (COCO, TensorFlow Object Detection, etc.). The project is structured as both a library (`src/lib.rs`) and a binary (`src/main.rs`), allowing use as a dependency or standalone CLI.

**Status:** Early development (v0.1.0) - IR module, COCO JSON, and TFOD CSV converters implemented. Validation and benchmarking in place.

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

### Benchmarking
```bash
cargo bench              # Run all Criterion benchmarks
cargo bench -- --test    # Smoke test benchmarks (no timing)
```

### Fuzzing (requires nightly)
```bash
cargo +nightly fuzz run coco_json_parse    # Fuzz COCO JSON parser
```

### Generate Test Data
```bash
# Requires numpy: pip install numpy (or uv pip install numpy)
python scripts/dataset_generator.py --num_images 1000 --annotations_per_image 10 --output_dir ./assets
```

## Architecture

```
src/
├── lib.rs              # Library entry point - CLI parsing and command dispatch
├── main.rs             # CLI binary - thin wrapper calling lib.rs
├── error.rs            # Error types (PanlabelError)
├── ir/                 # Intermediate Representation module
│   ├── mod.rs          # IR module exports
│   ├── model.rs        # Core types: Dataset, Image, Annotation, Category
│   ├── bbox.rs         # BBoxXYXY with coordinate space type safety
│   ├── coord.rs        # Coord type for 2D points
│   ├── space.rs        # Pixel/Normalized coordinate space markers
│   ├── ids.rs          # Strongly-typed IDs (ImageId, AnnotationId, etc.)
│   ├── io_coco_json.rs # COCO JSON reader/writer
│   ├── io_tfod_csv.rs  # TFOD CSV reader/writer
│   └── io_json.rs      # IR JSON format (for debugging/testing)
└── validation/         # Dataset validation
    ├── mod.rs          # validate_dataset() function
    └── report.rs       # ValidationReport formatting

tests/
├── cli.rs              # CLI integration tests using assert_cmd
├── tfod_csv_roundtrip.rs  # TFOD format roundtrip tests
└── fixtures/           # Test fixture files

benches/
└── microbenches.rs     # Criterion benchmarks for parsing/writing

fuzz/
└── fuzz_targets/       # cargo-fuzz targets for parser fuzzing

scripts/
└── dataset_generator.py  # Generates COCO and TFOD synthetic datasets
```

**Key design:** The CLI binary (`main.rs`) is intentionally minimal—it calls `panlabel::run()` from the library and handles errors. All business logic belongs in `lib.rs` (or modules it imports). The IR module uses Rust's type system (phantom types for coordinate spaces, newtypes for IDs) to prevent common annotation bugs at compile time.

## Annotation Format Reference

The project converts between these formats:

**COCO format** (JSON): `[x, y, width, height]` - absolute pixel coordinates from top-left

**TFOD format** (CSV): `xmin, ymin, xmax, ymax` columns - normalized coordinates (0.0-1.0)

See `scripts/dataset_generator.py` for conversion logic between formats.

## Testing Notes

- Integration tests use `assert_cmd` crate for CLI testing
- Test data goes in `assets/` directory (gitignored)
- Use deterministic seed (42) for reproducible test data generation
