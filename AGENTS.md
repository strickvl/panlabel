# Repository Guidelines

## Project Structure & Module Organization
- `src/lib.rs` is the library entry point with CLI command dispatch.
- `src/main.rs` is a thin CLI wrapper that calls into the library.
- `src/ir/` contains the Intermediate Representation module (model, bbox, converters), including `src/ir/io_yolo.rs` for YOLO TXT directory format (flat Darknet-style and split-aware layouts, with optional confidence token), `src/ir/io_yolo_keras_txt.rs` for shared YOLO Keras / YOLOv4 PyTorch absolute-coordinate TXT, `src/ir/io_voc_xml.rs` for Pascal VOC XML, `src/ir/io_label_studio_json.rs` for Label Studio JSON, `src/ir/io_labelbox_json.rs` for Labelbox JSON/NDJSON, `src/ir/io_scale_ai_json.rs` for Scale AI JSON, `src/ir/io_unity_perception_json.rs` for Unity Perception/SOLO JSON, `src/ir/io_labelme_json.rs` for LabelMe JSON (per-image, file + directory), `src/ir/io_createml_json.rs` for Apple CreateML JSON, `src/ir/io_cloud_annotations_json.rs` for IBM Cloud Annotations JSON, `src/ir/io_vott_csv.rs` for VoTT CSV, `src/ir/io_vott_json.rs` for VoTT JSON, `src/ir/io_kitti.rs` for KITTI object detection labels, `src/ir/io_via_json.rs` for VGG Image Annotator (VIA) JSON, `src/ir/io_retinanet_csv.rs` for RetinaNet Keras CSV, `src/ir/io_tfrecord.rs` for TFRecord (single-file uncompressed TFOD-style Example records), `src/ir/io_sagemaker_manifest.rs` for AWS SageMaker Ground Truth manifests, `src/ir/io_superannotate_json.rs` for SuperAnnotate JSON, `src/ir/io_supervisely_json.rs` for Supervisely JSON, `src/ir/io_cityscapes_json.rs` for Cityscapes polygon JSON, `src/ir/io_marmot_xml.rs` for Marmot XML, and `src/ir/io_super_json_common.rs` shared helpers for the SuperAnnotate/Supervisely adapters.
- `src/conversion/` contains conversion lossiness analysis and stable report issue codes.
- `src/stats/` contains dataset statistics logic and HTML/text/JSON reporting.
- `src/validation/` contains dataset validation logic.
- `tests/cli.rs` contains CLI integration tests using `assert_cmd`.
- `tests/tfod_csv_roundtrip.rs`, `tests/tfrecord_roundtrip.rs`, `tests/yolo_roundtrip.rs`, `tests/yolo_keras_roundtrip.rs`, `tests/voc_roundtrip.rs`, `tests/label_studio_roundtrip.rs`, `tests/labelbox_roundtrip.rs`, `tests/scale_ai_roundtrip.rs`, `tests/unity_perception_roundtrip.rs`, `tests/labelme_roundtrip.rs`, `tests/createml_roundtrip.rs`, `tests/cloud_annotations_roundtrip.rs`, `tests/vott_csv_roundtrip.rs`, `tests/vott_json_roundtrip.rs`, `tests/kitti_roundtrip.rs`, `tests/via_roundtrip.rs`, `tests/retinanet_csv_roundtrip.rs`, `tests/sagemaker_manifest_roundtrip.rs`, `tests/superannotate_roundtrip.rs`, `tests/supervisely_roundtrip.rs`, `tests/cityscapes_roundtrip.rs`, and `tests/marmot_roundtrip.rs` cover format-specific integration behavior.
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
cargo test --test proptest_labelme
cargo test --test proptest_createml
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
cargo +nightly fuzz run labelme_json_parse       # Fuzz LabelMe JSON parser (requires nightly)
cargo +nightly fuzz run createml_json_parse      # Fuzz CreateML JSON parser (requires nightly)
cargo +nightly fuzz run kitti_txt_parse          # Fuzz KITTI parser (requires nightly)
cargo +nightly fuzz run via_json_parse           # Fuzz VIA JSON parser (requires nightly)
cargo +nightly fuzz run retinanet_csv_parse      # Fuzz RetinaNet CSV parser (requires nightly)

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

## Agent Usage Guidance
- When automating the CLI, prefer machine-readable stdout:
  - `--output-format json` is the consistent cross-command spelling.
  - Read-only commands (`validate`, `stats`, `diff`, `list-formats`) also accept `--output json`.
  - `convert` and `sample` still accept `--report json` as a compatibility alias.
- Prefer `--from auto` unless the source format is already known.
- Review the conversion/sample report before adding `--allow-lossy`; the stable issue codes explain exactly what will be dropped or normalized.
- JSON/report payloads are written to stdout. Fatal errors go to stderr.
- If you change CLI behavior, update `tests/cli.rs`, `docs/cli.md`, and relevant README examples in the same change.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short and imperative (e.g., “Add basic CLI test”, “Update README”). Avoid prefixes unless needed.
- PRs should include: a clear description, test commands run, and any user-facing changes (help text, README) noted. Link an issue for larger changes.

## Docs Workflow
- If you change CLI behavior, update `docs/cli.md` and relevant README examples in the same change.
- If you change format behavior (COCO/Label Studio/TFOD/TFRecord/YOLO/VOC/IR), update `docs/formats.md`.
- If you change task/use-case support (detection vs segmentation/classification/etc.), update `docs/tasks.md`.
- If you change conversion/lossiness/report codes, update `docs/conversion.md`.
- Keep docs aligned with tests (`tests/cli.rs`, `tests/yolo_roundtrip.rs`, `tests/voc_roundtrip.rs`, `tests/label_studio_roundtrip.rs`, `tests/labelme_roundtrip.rs`, `tests/createml_roundtrip.rs`, `tests/kitti_roundtrip.rs`, `tests/via_roundtrip.rs`, `tests/retinanet_csv_roundtrip.rs`, and `tests/proptest_*.rs`), since user-visible behavior is asserted there.
- If you change auto-detection heuristics, update `docs/cli.md` and keep examples aligned with `tests/cli.rs`.
- **When adding a new format adapter** (any new `src/ir/io_*.rs`), update **all** of: `README.md` (Supported formats table; add a Quick-start example only if it's a name-recognizable platform with a meaningfully different invocation), `CLAUDE.md` (project-status `Supports …` line plus the `src/ir/` and `tests/` tree comments), `AGENTS.md` (the `src/ir/` description line and the test list line), `docs/README.md` (both the **What does panlabel support today?** list and the **source of truth map**), and the relevant per-topic `docs/*.md` files (`formats.md`, `cli.md`, `tasks.md`, `conversion.md`). The repo-root README is the storefront and goes stale fastest, so it must always be in the change.
- Keep forward-looking priorities in `ROADMAP.md` (separate from current-behavior docs).

## Configuration & Data Tips
- For the dataset generator, use a fresh Python virtual environment and install `numpy`.
- Keep generated assets out of git; only commit code and fixtures that are meant to be versioned.
- Do not edit or commit files under `design/` unless a task explicitly asks for a specific design-file update.
- If local design notes exist, treat them as historical background; implemented behavior belongs in `docs/` and tests.
- `fuzz/Cargo.toml` enables the crate `fuzzing` feature so fuzz-only parser entrypoints are available to fuzz targets.
- Keep `proptest-regressions/` in git to retain minimized repro cases from previous failures.
