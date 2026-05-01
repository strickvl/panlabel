# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Panlabel is a Rust library and CLI tool for converting between different object detection annotation formats (COCO, TensorFlow Object Detection, etc.). The project is structured as both a library (`src/lib.rs`) and a binary (`src/main.rs`), allowing use as a dependency or standalone CLI.

Scope guardrail: panlabel currently covers mainstream/static-image 2D axis-aligned object-detection bbox conversion. It does not provide first-class segmentation, keypoints/pose, oriented boxes, video tracking IDs, or 3D/multisensor labels; richer source structures are skipped/reported or treated as lossy.

**Status:** Active development (v0.6.0) - Full CLI with convert, validate, stats, diff, sample, and list-formats commands. Supports COCO JSON, CVAT XML, Label Studio JSON, Labelbox JSON/NDJSON, Scale AI JSON, Unity Perception JSON, LabelMe JSON, CreateML JSON, IBM Cloud Annotations JSON, VoTT CSV, VoTT JSON, KITTI, VIA JSON, VIA CSV, RetinaNet Keras CSV, OpenImages CSV, Kaggle Wheat CSV, Google Cloud AutoML Vision CSV, Udacity Self-Driving Car CSV, TFOD CSV, TFRecord (single-file uncompressed TensorFlow Object Detection API-style `tf.train.Example` bbox records), YOLO directory format (flat Darknet-style and split-aware layouts, with optional confidence token), YOLO Keras / YOLOv4 PyTorch absolute-coordinate TXT, Pascal VOC XML directory format, HF ImageFolder, AWS SageMaker Ground Truth manifest, SuperAnnotate JSON, Supervisely JSON, Cityscapes JSON, Marmot XML, Datumaro JSON, WIDER Face TXT, OIDv4 TXT, BDD100K/Scalabel JSON, V7 Darwin JSON, Edge Impulse `bounding_boxes.labels`, ASAM OpenLABEL JSON (2D bbox subset), and IR JSON with lossiness tracking.

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
cargo test --test proptest_labelme
cargo test --test proptest_createml
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
cargo +nightly fuzz run labelme_json_parse       # Fuzz LabelMe JSON parser
cargo +nightly fuzz run createml_json_parse      # Fuzz CreateML JSON parser
cargo +nightly fuzz run kitti_txt_parse          # Fuzz KITTI parser
cargo +nightly fuzz run via_json_parse           # Fuzz VIA JSON parser
cargo +nightly fuzz run retinanet_csv_parse      # Fuzz RetinaNet CSV parser
cargo +nightly fuzz run openimages_csv_parse     # Fuzz OpenImages CSV parser
cargo +nightly fuzz run kaggle_wheat_csv_parse   # Fuzz Kaggle Wheat CSV parser
cargo +nightly fuzz run automl_vision_csv_parse  # Fuzz AutoML Vision CSV parser
cargo +nightly fuzz run udacity_csv_parse        # Fuzz Udacity CSV parser
```

`fuzz/Cargo.toml` enables panlabel's `fuzzing` feature so the fuzz-only YOLO parser wrapper is available from the fuzz crate.

### Releasing
```bash
# Check what dist will build (dry run)
dist plan

# Regenerate release workflow after config changes
dist generate

# To release:
# 1. Bump version in Cargo.toml, update CHANGELOG.md
# 2. IMPORTANT: run `dist generate` to ensure release.yml is in sync
# 3. Commit, tag, push:
git tag vX.Y.Z
git push && git push --tags
# The .github/workflows/release.yml workflow handles the rest:
# - Builds binaries for 5 platforms
# - Creates GitHub Release with archives + checksums
# - Publishes Homebrew formula to strickvl/homebrew-tap
```

Release infrastructure:
- **cargo-dist** (`dist-workspace.toml`) manages cross-platform binary builds and GitHub Releases
- **Homebrew tap**: `strickvl/homebrew-tap` — auto-updated by release CI (requires `HOMEBREW_TAP_TOKEN` secret)
- **Tag convention**: `vX.Y.Z` (e.g., `v0.6.0`) — annotated tags trigger the release workflow

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
│   ├── io_cvat_xml.rs  # CVAT XML reader/writer
│   ├── io_label_studio_json.rs # Label Studio JSON reader/writer
│   ├── io_labelbox_json.rs    # Labelbox JSON/NDJSON reader/writer
│   ├── io_scale_ai_json.rs    # Scale AI task/response JSON reader/writer
│   ├── io_unity_perception_json.rs # Unity Perception/SOLO JSON reader/writer
│   ├── io_labelme_json.rs     # LabelMe JSON reader/writer (file + directory)
│   ├── io_createml_json.rs    # Apple CreateML JSON reader/writer
│   ├── io_cloud_annotations_json.rs # IBM Cloud Annotations JSON reader/writer
│   ├── io_vott_csv.rs  # Microsoft VoTT CSV reader/writer
│   ├── io_vott_json.rs # Microsoft VoTT JSON reader/writer
│   ├── io_kitti.rs      # KITTI object detection reader/writer (directory-based)
│   ├── io_via_json.rs   # VGG Image Annotator (VIA) JSON reader/writer
│   ├── io_openimages_csv.rs  # OpenImages CSV reader/writer
│   ├── io_kaggle_wheat_csv.rs # Kaggle Wheat CSV reader/writer
│   ├── io_automl_vision_csv.rs # Google Cloud AutoML Vision CSV reader/writer
│   ├── io_udacity_csv.rs     # Udacity Self-Driving Car CSV reader/writer
│   ├── io_retinanet_csv.rs # RetinaNet Keras CSV reader/writer
│   ├── io_tfod_csv.rs  # TFOD CSV reader/writer
│   ├── io_tfrecord.rs # TFRecord reader/writer (single-file uncompressed TFOD-style Example records)
│   ├── io_yolo.rs      # Ultralytics YOLO reader/writer (directory-based)
│   ├── io_yolo_keras_txt.rs # YOLO Keras / YOLOv4 PyTorch TXT reader/writer
│   ├── io_voc_xml.rs   # Pascal VOC XML reader/writer (directory-based)
│   ├── io_hf_imagefolder.rs   # Hugging Face ImageFolder metadata reader/writer
│   ├── io_hf_parquet.rs       # Hugging Face parquet metadata support (feature-gated)
│   ├── io_sagemaker_manifest.rs # AWS SageMaker Ground Truth manifest reader/writer
│   ├── io_superannotate_json.rs # SuperAnnotate JSON reader/writer
│   ├── io_supervisely_json.rs   # Supervisely JSON reader/writer
│   ├── io_cityscapes_json.rs    # Cityscapes polygon JSON reader/writer
│   ├── io_marmot_xml.rs         # Marmot XML reader/writer
│   ├── io_datumaro_json.rs      # Datumaro JSON reader/writer
│   ├── io_wider_face_txt.rs     # WIDER Face aggregate TXT reader/writer
│   ├── io_oidv4_txt.rs          # OIDv4 Toolkit TXT reader/writer (`Label/` directories)
│   ├── io_bdd100k_json.rs       # BDD100K / Scalabel JSON bbox-subset reader/writer
│   ├── io_v7_darwin_json.rs     # V7 Darwin JSON bbox-subset reader/writer
│   ├── io_edge_impulse_labels.rs # Edge Impulse bounding_boxes.labels reader/writer
│   ├── io_openlabel_json.rs     # ASAM OpenLABEL JSON static-image 2D bbox-subset reader/writer
│   ├── io_via_csv.rs            # VIA CSV reader/writer (separate from VIA JSON)
│   ├── io_super_json_common.rs  # Shared helpers for SuperAnnotate/Supervisely adapters
│   └── io_json.rs      # IR JSON format (canonical serialization)
├── validation/         # Dataset validation
│   ├── mod.rs          # validate_dataset() function
│   └── report.rs       # ValidationReport formatting
├── conversion/         # Format conversion reporting
│   ├── mod.rs          # build_conversion_report(), Format enum, IrLossiness
│   └── report.rs       # ConversionReport with lossiness warnings
└── stats/              # Dataset statistics + HTML/text/JSON reporting
    ├── mod.rs          # stats_dataset() function
    ├── report.rs       # StatsReport with terminal formatting
    └── html.rs         # Self-contained HTML report renderer

tests/
├── cli.rs              # CLI integration tests using assert_cmd
├── common/mod.rs       # Shared BMP helpers for YOLO-related tests
├── proptest_helpers/mod.rs # Shared proptest strategies + semantic assertions
├── proptest_*.rs       # Property tests per adapter + cross-format subset checks
├── tfod_csv_roundtrip.rs  # TFOD format roundtrip tests
├── tfrecord_roundtrip.rs  # TFRecord format roundtrip tests
├── yolo_roundtrip.rs      # YOLO format roundtrip tests
├── yolo_keras_roundtrip.rs # YOLO Keras / YOLOv4 PyTorch TXT roundtrip tests
├── voc_roundtrip.rs       # VOC format roundtrip tests
├── cvat_roundtrip.rs      # CVAT XML format roundtrip tests
├── label_studio_roundtrip.rs # Label Studio format roundtrip tests
├── labelbox_roundtrip.rs  # Labelbox format roundtrip tests
├── scale_ai_roundtrip.rs  # Scale AI format roundtrip tests
├── unity_perception_roundtrip.rs # Unity Perception format roundtrip tests
├── labelme_roundtrip.rs   # LabelMe format roundtrip tests
├── createml_roundtrip.rs  # CreateML format roundtrip tests
├── kitti_roundtrip.rs     # KITTI format roundtrip tests
├── via_roundtrip.rs       # VIA JSON format roundtrip tests
├── retinanet_csv_roundtrip.rs # RetinaNet CSV format roundtrip tests
├── hf_imagefolder_roundtrip.rs   # HF ImageFolder format roundtrip tests
├── sagemaker_manifest_roundtrip.rs # SageMaker Ground Truth manifest roundtrip tests
├── superannotate_roundtrip.rs # SuperAnnotate JSON format roundtrip tests
├── supervisely_roundtrip.rs   # Supervisely JSON format roundtrip tests
├── cityscapes_roundtrip.rs    # Cityscapes JSON format roundtrip tests
├── marmot_roundtrip.rs        # Marmot XML format roundtrip tests
├── datumaro_roundtrip.rs      # Datumaro JSON roundtrip tests
├── wider_face_roundtrip.rs    # WIDER Face TXT roundtrip tests
├── oidv4_roundtrip.rs         # OIDv4 TXT roundtrip tests
├── bdd100k_roundtrip.rs       # BDD100K JSON roundtrip tests
├── v7_darwin_roundtrip.rs     # V7 Darwin JSON roundtrip tests
├── edge_impulse_roundtrip.rs  # Edge Impulse labels roundtrip tests
├── openlabel_roundtrip.rs     # OpenLABEL JSON roundtrip tests
├── via_csv_roundtrip.rs       # VIA CSV roundtrip tests
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
  - User-visible behavior checks: `tests/cli.rs`, `tests/*_roundtrip.rs`, and `tests/proptest_*.rs`

If command behavior, format semantics, or conversion issue codes change, update `docs/` in the same change.

### Adding a new format adapter

When adding a new format adapter (any new `src/ir/io_*.rs` reader/writer), update **all** of the following in the same change so the docs do not drift:

- `README.md` — add a row to the **Supported formats** table. Only add a Quick-start example if the format is a name-recognizable platform (e.g. SuperAnnotate, Supervisely, SageMaker) or has a meaningfully different invocation. Don't add a Quick-start line for every format.
- `CLAUDE.md` — append the format to the project-status `Supports …` line, the `src/ir/` tree comment, the `tests/` tree comment (new roundtrip test), and the auto-detection rules under "Convert with Auto-Detection" if applicable.
- `AGENTS.md` — append the new `io_*.rs` to the `src/ir/` description line, and any new `tests/*_roundtrip.rs` to the test list line.
- `docs/README.md` — both the **What does panlabel support today?** list and the **source of truth map** must include the new format.
- `docs/formats.md`, `docs/cli.md`, `docs/tasks.md`, `docs/conversion.md` — these are the source-of-truth docs. The repo-root README is the storefront and goes stale fastest, which is why it's listed first.

## CLI Commands

| Command | Description |
|---------|-------------|
| `validate` | Check dataset for errors (duplicate IDs, missing refs, invalid bboxes) |
| `convert` | Convert between formats with lossiness tracking |
| `stats` | Display statistics (counts, label histogram, bbox quality metrics) |
| `diff` | Compare two datasets semantically |
| `sample` | Create subset datasets (random or stratified), with JSON report output available |
| `list-formats` | Show supported formats with read/write and lossiness info, including JSON discovery output |

### Machine-readable output

- `--output-format json` is the consistent cross-command spelling for structured stdout.
- Read-only commands (`validate`, `stats`, `diff`, `list-formats`) also accept `--output json`.
- `convert` and `sample` keep `--report json` as a compatibility alias.
- JSON/report payloads go to stdout; fatal errors go to stderr.

### Convert with Auto-Detection

The `--from auto` flag detects format from file extension/content for files and layout markers for directories:
- `.csv` → content-based: 8 columns → TFOD or Udacity by coordinate range/header, 6 columns → RetinaNet, or other recognized CSV headers
- `.tfrecord` → TFRecord framing + first-record probe for TFOD-style `tf.train.Example` payloads (v1 scope)
- `.txt` → specifically named YOLO Keras / YOLOv4 PyTorch absolute-coordinate TXT files can be detected; shared names such as `train.txt` and `train_annotations.txt` are ambiguous and require explicit `--from`
- `.xml` → root `<annotations>` = CVAT; root `<Page CropBox="...">` = Marmot
- `.jsonl` / `.ndjson` / `.manifest`: Labelbox export-row shape is checked before SageMaker manifest rows
- `.json`:
  - empty array-root JSON (`[]`) → ambiguous (Label Studio or CreateML); requires explicit `--from`
  - non-empty array-root: Labelbox, Scale AI, Unity Perception, Label Studio, or CreateML by row/task shape
  - object-root: Labelbox, Scale AI, Unity Perception, LabelMe, VoTT JSON, SuperAnnotate, Cityscapes, Supervisely, VIA, COCO, or IR JSON by schema markers
- directory with `labels/` containing `.txt` files AND sibling `images/`, or `data.yaml` split keys pointing to image dirs/list files → YOLO (labels without images is reported as an incomplete layout)
- directory with matching YOLO Keras / YOLOv4 PyTorch TXT annotation files → YOLO Keras or YOLOv4 PyTorch; shared names may be ambiguous
- directory with `gtFine/<split>/<city>/*_gtFine_polygons.json`, a `gtFine/` root, or matching Cityscapes polygon JSON files → Cityscapes
- directory with Marmot `<Page CropBox="...">` XML files plus same-stem companion images → Marmot
- directory markers also cover VOC, CVAT, IBM Cloud Annotations, VoTT JSON, Scale AI, Unity Perception, LabelMe, SuperAnnotate, Supervisely, KITTI, and HF layouts
- detection uses evidence-based probing (`FormatProbe` + `probe_dir_formats()`) that reports what was found/missing
- `stats` falls back to `ir-json` for parseable JSON files but surfaces malformed JSON errors directly

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
