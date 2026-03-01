# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Panlabel is a Rust library and CLI tool for converting between different object detection annotation formats (COCO, TensorFlow Object Detection, etc.). The project is structured as both a library (`src/lib.rs`) and a binary (`src/main.rs`), allowing use as a dependency or standalone CLI.

**Status:** Early development (v0.1.0) - Full CLI with convert, validate, inspect, and list-formats commands. Supports COCO JSON, Label Studio JSON, TFOD CSV, YOLO directory format, Pascal VOC XML directory format, and IR JSON with lossiness tracking.

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
cargo test               # Run all tests (unit + integration + proptests)
cargo test --test cli    # Run only CLI integration tests
cargo test --test proptest_ir_json
cargo test --test proptest_coco
cargo test --test proptest_tfod
cargo test --test proptest_label_studio
cargo test --test proptest_voc
cargo test --test proptest_yolo
cargo test --test proptest_cross_format
cargo test runs          # Run a single test by name

PROPTEST_CASES=1000 cargo test --test proptest_ir_json   # deeper local run
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
cargo +nightly fuzz run coco_json_parse          # Fuzz COCO JSON parser
cargo +nightly fuzz run voc_xml_parse            # Fuzz VOC XML parser
cargo +nightly fuzz run tfod_csv_parse           # Fuzz TFOD CSV parser
cargo +nightly fuzz run label_studio_json_parse  # Fuzz Label Studio parser
cargo +nightly fuzz run ir_json_parse            # Fuzz IR JSON parser
cargo +nightly fuzz run yolo_label_line_parse    # Fuzz YOLO line parser
```

`fuzz/Cargo.toml` enables panlabel's `fuzzing` feature so the fuzz-only YOLO parser wrapper is available from the fuzz crate.

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
│   ├── io_label_studio_json.rs # Label Studio JSON reader/writer
│   ├── io_tfod_csv.rs  # TFOD CSV reader/writer
│   ├── io_yolo.rs      # Ultralytics YOLO reader/writer (directory-based)
│   ├── io_voc_xml.rs   # Pascal VOC XML reader/writer (directory-based)
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
├── common/mod.rs       # Shared BMP helpers for YOLO-related tests
├── proptest_helpers/mod.rs # Shared proptest strategies + semantic assertions
├── proptest_*.rs       # Property tests per adapter + cross-format subset checks
├── tfod_csv_roundtrip.rs  # TFOD format roundtrip tests
├── yolo_roundtrip.rs      # YOLO format roundtrip tests
├── voc_roundtrip.rs       # VOC format roundtrip tests
├── label_studio_roundtrip.rs # Label Studio format roundtrip tests
└── fixtures/           # Test fixture files

proptest-regressions/
└── ...                 # Committed proptest shrinking regressions

benches/
└── microbenches.rs     # Criterion benchmarks for parsing/writing

fuzz/
├── corpus/             # Seed corpora per fuzz target
└── fuzz_targets/       # cargo-fuzz targets for parser fuzzing

scripts/
└── dataset_generator.py  # Generates COCO and TFOD synthetic datasets

docs/
├── README.md            # Documentation hub
├── cli.md               # CLI contract and examples
├── formats.md           # Supported format behavior reference
├── tasks.md             # Task/use-case support matrix
└── conversion.md        # Lossiness + report schema and issue codes
```

## Documentation Topology

- Root `README.md` is a quick project gateway.
- `CONTRIBUTING.md` lives at repo root (GitHub auto-links it in issues/PRs).
- User-facing reference docs live in `docs/`.
- Forward-looking priorities live in `ROADMAP.md`.
- `design/` documents are historical context only and may be stale after implementation.
- Source of truth for docs accuracy:
  - CLI and auto-detection: `src/lib.rs`
  - Format adapters: `src/ir/io_*.rs`
  - Lossiness/report codes: `src/conversion/*`
  - User-visible behavior checks: `tests/cli.rs`, `tests/yolo_roundtrip.rs`, `tests/voc_roundtrip.rs`, `tests/label_studio_roundtrip.rs`

If command behavior, format semantics, or conversion issue codes change, update `docs/` in the same change.

## CLI Commands

| Command | Description |
|---------|-------------|
| `validate` | Check dataset for errors (duplicate IDs, missing refs, invalid bboxes) |
| `convert` | Convert between formats with lossiness tracking |
| `inspect` | Display statistics (counts, label histogram, bbox quality metrics) |
| `list-formats` | Show supported formats with read/write and lossiness info |

### Convert with Auto-Detection

The `--from auto` flag detects format from file extension/content for files and layout markers for directories:
- `.csv` → TFOD
- `.json`:
  - empty array-root JSON (`[]`) → Label Studio
  - non-empty array-root: check only first task for `data.image` string → Label Studio
  - object-root: requires a non-empty `annotations` array, then peek at `annotations[0].bbox`: array = COCO, object = IR JSON (empty datasets cannot be auto-detected)
- directory with `labels/` containing `.txt` files (or direct `labels/` dir with `.txt`) → YOLO
- directory with `Annotations/` containing `.xml` and sibling `JPEGImages/` (or direct `Annotations/` with sibling `JPEGImages/`) → VOC

**Key design:** The CLI binary (`main.rs`) is intentionally minimal—it calls `panlabel::run()` from the library and handles errors. All business logic belongs in `lib.rs` (or modules it imports). The IR module uses Rust's type system (phantom types for coordinate spaces, newtypes for IDs) to prevent common annotation bugs at compile time.

## Annotation Format Reference

Use `docs/formats.md` as the primary format reference.

Quick links:
- [`docs/formats.md`](docs/formats.md)
- [`docs/tasks.md`](docs/tasks.md)
- [`docs/conversion.md`](docs/conversion.md)
- [`docs/cli.md`](docs/cli.md)

Scope reminder: current support is object detection bboxes (not segmentation, keypoints/pose, OBB, or classification-only pipelines).

See `scripts/dataset_generator.py` for synthetic data generation.

## Testing Notes

- Integration tests use `assert_cmd` crate for CLI testing
- Test data goes in `assets/` directory (gitignored)
- Use deterministic seed (42) for reproducible test data generation
