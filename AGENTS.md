# Repository Guidelines

## Project Structure & Module Organization
- `src/lib.rs` is the library entry point with CLI command dispatch.
- `src/main.rs` is a thin CLI wrapper that calls into the library.
- `src/ir/` contains the Intermediate Representation module (model, bbox, converters), including `src/ir/io_yolo.rs` for Ultralytics-style YOLO, `src/ir/io_voc_xml.rs` for Pascal VOC XML, and `src/ir/io_label_studio_json.rs` for Label Studio JSON.
- `src/conversion/` contains conversion lossiness analysis and stable report issue codes.
- `src/inspect/` contains dataset inspection/statistics logic.
- `src/validation/` contains dataset validation logic.
- `tests/cli.rs` contains CLI integration tests using `assert_cmd`.
- `tests/tfod_csv_roundtrip.rs`, `tests/yolo_roundtrip.rs`, `tests/voc_roundtrip.rs`, and `tests/label_studio_roundtrip.rs` cover format-specific integration behavior.
- `tests/proptest_*.rs` add property-based roundtrip/idempotency/subset checks; shared helpers live in `tests/proptest_helpers/mod.rs` and `tests/common/mod.rs`.
- `docs/` is the durable documentation home for users and contributors.
- `benches/` contains Criterion benchmarks.
- `fuzz/` contains cargo-fuzz targets for parser fuzzing.
- `scripts/dataset_generator.py` generates synthetic datasets for testing.
- `assets/` is for generated test data and is gitignored.

## Build, Test, and Development Commands
```bash
cargo build              # Build debug version
cargo build --release    # Build optimized release version
cargo run                # Run the CLI
cargo run -- -V          # Run with CLI args
cargo check              # Fast type checking
cargo fmt                # Format code
cargo clippy             # Lint
cargo test               # All tests (unit + integration + proptests)
cargo test --test cli    # CLI integration tests only
cargo test --test proptest_ir_json
cargo test --test proptest_coco
cargo test --test proptest_tfod
cargo test --test proptest_label_studio
cargo test --test proptest_voc
cargo test --test proptest_yolo
cargo test --test proptest_cross_format
cargo test runs          # Run a single test by name

PROPTEST_CASES=1000 cargo test --test proptest_ir_json  # deeper local run
cargo doc --open         # Build and view docs
cargo bench              # Run Criterion benchmarks
cargo bench -- --test    # Smoke test benchmarks
cargo +nightly fuzz run coco_json_parse          # Fuzz COCO parser (requires nightly)
cargo +nightly fuzz run voc_xml_parse            # Fuzz Pascal VOC XML parser (requires nightly)
cargo +nightly fuzz run tfod_csv_parse           # Fuzz TFOD CSV parser (requires nightly)
cargo +nightly fuzz run label_studio_json_parse  # Fuzz Label Studio parser (requires nightly)
cargo +nightly fuzz run ir_json_parse            # Fuzz IR JSON parser (requires nightly)
cargo +nightly fuzz run yolo_label_line_parse    # Fuzz YOLO line parser (requires nightly)

python scripts/dataset_generator.py --num_images 1000 --annotations_per_image 10 --output_dir ./assets
```

## Coding Style & Naming Conventions
- Follow `rustfmt` defaults (4-space indentation, standard Rust formatting).
- Run `cargo fmt` and `cargo clippy` before opening a PR.
- Naming: `snake_case` for functions/modules/tests, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- Keep CLI glue in `src/main.rs`; put core behavior in `src/lib.rs` or its modules.

## Testing Guidelines
- Integration tests live in `tests/` and use `assert_cmd`; property tests use `proptest` in `tests/proptest_*.rs`.
- Add unit tests in `src/` with `#[cfg(test)]` when appropriate.
- Generated datasets belong in `assets/` (gitignored). Use the deterministic seed (42) when modifying the generator to keep data reproducible.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short and imperative (e.g., “Add basic CLI test”, “Update README”). Avoid prefixes unless needed.
- PRs should include: a clear description, test commands run, and any user-facing changes (help text, README) noted. Link an issue for larger changes.

## Docs Workflow
- If you change CLI behavior, update `docs/cli.md` and relevant README examples in the same change.
- If you change format behavior (COCO/Label Studio/TFOD/YOLO/VOC/IR), update `docs/formats.md`.
- If you change task/use-case support (detection vs segmentation/classification/etc.), update `docs/tasks.md`.
- If you change conversion/lossiness/report codes, update `docs/conversion.md`.
- Keep docs aligned with tests (`tests/cli.rs`, `tests/yolo_roundtrip.rs`, `tests/voc_roundtrip.rs`, `tests/label_studio_roundtrip.rs`, and `tests/proptest_*.rs`), since user-visible behavior is asserted there.
- If you change auto-detection heuristics, update `docs/cli.md` and keep examples aligned with `tests/cli.rs`.
- Keep forward-looking priorities in `ROADMAP.md` (separate from current-behavior docs).

## Configuration & Data Tips
- For the dataset generator, use a fresh Python virtual environment and install `numpy`.
- Keep generated assets out of git; only commit code and fixtures that are meant to be versioned.
- Never add anything under `design/` to git history (it is gitignored for a reason).
- If local design notes exist, treat them as historical background; implemented behavior belongs in `docs/` and tests.
- `fuzz/Cargo.toml` enables the crate `fuzzing` feature so fuzz-only parser entrypoints are available to fuzz targets.
- Keep `proptest-regressions/` in git to retain minimized repro cases from previous failures.
