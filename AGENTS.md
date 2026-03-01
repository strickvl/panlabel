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
cargo test               # All tests (unit + integration)
cargo test --test cli    # CLI integration tests only
cargo test runs          # Run a single test by name
cargo doc --open         # Build and view docs
cargo bench              # Run Criterion benchmarks
cargo bench -- --test    # Smoke test benchmarks
cargo +nightly fuzz run coco_json_parse  # Fuzz COCO parser (requires nightly)
cargo +nightly fuzz run voc_xml_parse    # Fuzz Pascal VOC XML parser (requires nightly)

python scripts/dataset_generator.py --num_images 1000 --annotations_per_image 10 --output_dir ./assets
```

## Coding Style & Naming Conventions
- Follow `rustfmt` defaults (4-space indentation, standard Rust formatting).
- Run `cargo fmt` and `cargo clippy` before opening a PR.
- Naming: `snake_case` for functions/modules/tests, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- Keep CLI glue in `src/main.rs`; put core behavior in `src/lib.rs` or its modules.

## Testing Guidelines
- Integration tests live in `tests/` and use `assert_cmd`.
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
- Keep docs aligned with tests (`tests/cli.rs`, `tests/yolo_roundtrip.rs`, `tests/voc_roundtrip.rs`, `tests/label_studio_roundtrip.rs`), since user-visible behavior is asserted there.
- If you change auto-detection heuristics, update `docs/cli.md` and keep examples aligned with `tests/cli.rs`.
- Keep forward-looking priorities in `ROADMAP.md` (separate from current-behavior docs).

## Configuration & Data Tips
- For the dataset generator, use a fresh Python virtual environment and install `numpy`.
- Keep generated assets out of git; only commit code and fixtures that are meant to be versioned.
- Never add anything under `design/` to git history (it is gitignored for a reason).
- If local design notes exist, treat them as historical background; implemented behavior belongs in `docs/` and tests.
