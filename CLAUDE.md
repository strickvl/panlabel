# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Panlabel is a Rust library and CLI tool for converting between different object detection annotation formats (COCO, TensorFlow Object Detection, etc.). The project is structured as both a library (`src/lib.rs`) and a binary (`src/main.rs`), allowing use as a dependency or standalone CLI.

**Status:** Early development (v0.1.0) - Full CLI with convert, validate, inspect, and list-formats commands. Supports COCO JSON, TFOD CSV, and IR JSON formats with lossiness tracking.

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
│   └── io_json.rs      # IR JSON format (canonical serialization)
├── validation/         # Dataset validation
│   ├── mod.rs          # validate_dataset() function
│   └── report.rs       # ValidationReport formatting
├── conversion/         # Format conversion reporting
│   ├── mod.rs          # build_conversion_report(), Format enum, IrLossiness
│   └── report.rs       # ConversionReport with lossiness warnings
└── inspect/            # Dataset inspection and statistics
    ├── mod.rs          # inspect_dataset() function
    └── report.rs       # InspectReport with terminal formatting

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

## CLI Commands

| Command | Description |
|---------|-------------|
| `validate` | Check dataset for errors (duplicate IDs, missing refs, invalid bboxes) |
| `convert` | Convert between formats with lossiness tracking |
| `inspect` | Display statistics (counts, label histogram, bbox quality metrics) |
| `list-formats` | Show supported formats with read/write and lossiness info |

### Convert with Auto-Detection

The `--from auto` flag detects format from file extension and content:
- `.csv` → TFOD
- `.json` → Peek at `annotations[0].bbox`: array = COCO, object = IR JSON

**Key design:** The CLI binary (`main.rs`) is intentionally minimal—it calls `panlabel::run()` from the library and handles errors. All business logic belongs in `lib.rs` (or modules it imports). The IR module uses Rust's type system (phantom types for coordinate spaces, newtypes for IDs) to prevent common annotation bugs at compile time.

## Annotation Format Reference

The project converts between these formats:

**IR JSON** (Panlabel's canonical format):
- Bbox: `{"min": {"x": f64, "y": f64}, "max": {"x": f64, "y": f64}}` - absolute pixel coords (xyxy)
- Lossless: preserves all metadata, licenses, attributes

**COCO format** (JSON):
- Bbox: `[x, y, width, height]` - absolute pixel coordinates from top-left
- Conditional lossiness: loses `info.name`, some attributes may not roundtrip

**TFOD format** (CSV):
- Columns: `filename, width, height, class, xmin, ymin, xmax, ymax`
- Coords: normalized (0.0-1.0)
- Lossy: no metadata, licenses, confidence scores, or images without annotations

See `scripts/dataset_generator.py` for synthetic data generation.

## Testing Notes

- Integration tests use `assert_cmd` crate for CLI testing
- Test data goes in `assets/` directory (gitignored)
- Use deterministic seed (42) for reproducible test data generation
