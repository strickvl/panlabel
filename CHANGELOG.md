# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `list-formats --output json` machine-readable format discovery, including aliases and file/directory layout hints
- `sample --output-format json` structured conversion/sample reports (with `--report` alias support)

### Changed

- `validate` now uses typed enum parsing for `--format` and report output mode values
- `--output-format` is now the consistent cross-command spelling for structured stdout (`validate`, `stats`, `diff`, and `list-formats` also keep `--output`; `convert` and `sample` keep `--report` as an alias)
- Contributor guidance (`AGENTS.md`, `CLAUDE.md`) now reflects the current `stats` module and agent-oriented CLI usage guidance

## v0.5.0

Split-aware YOLO reading, conversion report explainability, and documentation improvements.

### Added

- **Split-aware YOLO reading**: datasets with `data.yaml` containing `train:`, `val:`, `test:` path keys (Roboflow/Ultralytics Hub exports) are now supported. Three path patterns recognized: `images/<split>`, `<split>/images`, and `<split>` with sub-directories.
- `--split <name>` flag now works for YOLO imports (previously HF-only), selecting a single split from a split-aware dataset
- Default behavior merges all splits with split-prefixed file names (`train/img.jpg`, `val/img.jpg`) for collision avoidance
- Split provenance stored in `Dataset.info.attributes` (`yolo_layout_mode`, `yolo_splits_found`, `yolo_splits_read`)
- New `yolo_reader_split_handling` info code in conversion reports
- Auto-detection recognizes split-aware YOLO layouts via `data.yaml` with split keys
- **Conversion report explainability**: all format adapters now emit policy notes (info-level issue codes) explaining deterministic behavior, attribute mapping, and writer conventions
- `ConversionStage` field on report issues (`source_reader`, `analysis`, `target_writer`) for clearer provenance
- Drift-prevention tests ensure all issue codes are documented in `docs/conversion.md`

### Changed

- `--split` flag help text updated from HF-specific to shared (HF + YOLO)
- YOLO reader eagerly resolves `data.yaml` names at discovery time (avoids double file read)

## v0.4.0

Auto-detection UX overhaul, CVAT reporting parity, Docker support, and dependency updates.

### Added

- Docker support: multi-stage Dockerfile and GitHub Actions workflow for Docker Hub images
- CVAT conversion reporting: 4 new policy notes (deterministic ordering, image ID reassignment, source defaults, unused category drop) and accurate output category counts when unused categories are dropped

### Changed

- **Auto-detection UX overhaul**:
  - VOC: no longer requires `JPEGImages/` for auto-detection (matches reader behavior)
  - VOC: uses flat XML scan (matches reader's non-recursive `collect_xml_files`)
  - YOLO: requires `images/` directory for detection (matches reader); labels without images reported as incomplete layout
  - HF: detects parquet shard layouts (`data/train-*.parquet`)
  - Ambiguity errors now list concrete evidence for each matched format
  - Partial layout errors explain what markers were found and what's missing
  - Unrecognized directory errors list all expected layouts
  - Missing file errors include path context instead of bare "IO error"
- `stats` no longer silently falls back to IR JSON when JSON is malformed (only falls back when JSON parses but format is ambiguous)
- HF duplicate `file_name` errors now show source file and line for both the original and duplicate rows

### Dependencies

- Updated rand 0.9 → 0.10, roxmltree 0.20 → 0.21, zip 0.6.6 → 8.2.0

## v0.3.0

First Hugging Face Datasets release for panlabel, including local and remote HF ingestion.

### Added

- **HF ImageFolder format support (`hf`)** in `convert`, `validate`, and `list-formats`
- Local HF metadata ingestion:
  - `metadata.jsonl` read/write
  - `metadata.parquet` read (feature-gated)
  - split-parquet shard layout support
- Remote HF Hub ingestion in `convert` via `--hf-repo`, including:
  - repo ID/URL resolution
  - viewer preflight metadata probing
  - authenticated acquisition with `--token`/`HF_TOKEN`
- HF-specific CLI options:
  - `--hf-bbox-format`, `--hf-objects-column`, `--hf-category-map`
  - `--hf-repo`, `--split`, `--revision`, `--config`, `--token`
- HF provenance attributes on datasets (repo/revision/split/license/description context)
- Zip-style remote split fallback (for repos shipping `train.zip`/`valid.zip`/`test.zip`) with payload routing to COCO/YOLO/VOC/HF readers
- Safer zip extraction guards (path traversal checks + extraction limits)

### Changed

- COCO reader now accepts `info.year` as either integer or string
- Improved clippy/CI compatibility for feature-gated builds

## v0.2.0

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

## v0.1.0

Placeholder initial release on crates.io.

