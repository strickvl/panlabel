# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.7.0

Twenty-five new format adapters covering the major cloud annotation platforms, autonomous-driving and aerial datasets, document layout, synthetic data, and the long tail of academic/community formats. Panlabel now reads and writes 40+ object detection annotation formats.

### Added

#### Cloud annotation platforms

- **Labelbox JSON/NDJSON support (`labelbox`)**: Labelbox current export rows (`data_row` / `projects.*.labels` shape) for both single-JSON and newline-delimited exports. Reads bbox annotations from labels, preserves metadata as `labelbox_*` attributes.
- **Scale AI JSON support (`scale-ai`)**: Scale AI image annotation task and response JSON, single file or `annotations/` directory layout. Bbox annotations from task responses with attribute and metadata preservation.
- **SuperAnnotate JSON support (`superannotate`)**: SuperAnnotate JSON export, single file or `annotations/` directory. Bbox instances with class/attribute mapping.
- **Supervisely JSON support (`supervisely`)**: Supervisely JSON project / dataset format with `ann/` + `meta.json` project layout. Reads class metadata from `meta.json`, bbox figures from per-image annotations.
- **AWS SageMaker Ground Truth manifest support (`sagemaker`)**: object-detection job manifest (`.manifest` / `.jsonl`) with per-row image refs and bbox annotations. Reads SageMaker labeling job output directly.
- **IBM Cloud Annotations JSON support (`ibm-cloud-annotations`)**: IBM Cloud Annotations localization JSON (`_annotations.json` file or directory) with absolute pixel bboxes.
- **Google Cloud AutoML Vision CSV support (`automl-vision`)**: AutoML Vision sparse 9/11-column layout, GCS URI image refs, ML split (`TRAIN`/`VALIDATION`/`TEST`) preserved as attributes.
- **V7 Darwin JSON support (`v7-darwin`)**: V7 Darwin JSON bbox subset with annotation attribute mapping.
- **Edge Impulse labels support (`edge-impulse`)**: Edge Impulse `bounding_boxes.labels` file or containing directory. Reads bounding-box labels from Edge Impulse studio exports.

#### Autonomous driving and aerial

- **TFRecord support (`tfrecord`)**: TensorFlow Object Detection API-style `tf.train.Example` records (single-file, uncompressed, bbox-only in v1). Reads/writes the standard TFOD `image/encoded` + `image/object/bbox/{xmin,ymin,xmax,ymax}` + `image/object/class/{label,text}` schema. Auto-detected via TFRecord framing + first-record probe.
- **Cityscapes JSON support (`cityscapes`)**: Cityscapes polygon JSON, single file or `gtFine/<split>/<city>/*_gtFine_polygons.json` dataset root. Polygons are flattened to axis-aligned bbox envelopes (lossy).
- **BDD100K / Scalabel JSON support (`bdd100k`)**: BDD100K detection subset with bbox annotations and per-frame attribute preservation.
- **Udacity Self-Driving Car CSV support (`udacity`)**: Udacity dataset CSV with absolute pixel coordinates (same column shape as TFOD but without normalization). Auto-detected by coordinate range/header inspection.
- **OpenImages CSV support (`openimages`)**: Google OpenImages CSV with normalized coordinates, 8/13-column variants, confidence preservation, and image-dimension resolution from disk.
- **VoTT CSV support (`vott-csv`)**: Microsoft VoTT CSV export (`image,xmin,ymin,xmax,ymax,label`).
- **VoTT JSON support (`vott-json`)**: Microsoft VoTT JSON export, single-file `assets`-keyed JSON or `vott-json-export/` directory with per-asset JSON files.

#### Document layout, synthetic, and benchmark datasets

- **Marmot XML support (`marmot`)**: Marmot document-layout XML composites (`<Page CropBox="...">`) with hex-encoded coordinate doubles converted to pixel bboxes. Same-stem companion image resolution.
- **Unity Perception JSON support (`unity-perception`)**: Unity Perception / SOLO synthetic-data bbox JSON, single file or SOLO-like directory.
- **Datumaro JSON support (`datumaro`)**: Datumaro JSON annotation format used by OpenVINO Training Extensions and CVAT exports.
- **WIDER Face TXT support (`wider-face`)**: WIDER Face aggregate TXT format (single `face` class in panlabel's bbox-only scope).
- **OIDv4 Toolkit TXT support (`oidv4`)**: OIDv4 Toolkit TXT labels under `Label/` directories. Directory probe specifically uses `Label/` to disambiguate from YOLO `labels/`.
- **Kaggle Wheat CSV support (`kaggle-wheat`)**: Kaggle Global Wheat Detection single-class CSV with bracketed `[x,y,w,h]` bbox strings and source attribute mapping.

#### Standards and academic formats

- **ASAM OpenLABEL JSON support (`openlabel`)**: ASAM OpenLABEL JSON static-image 2D bbox subset (the ISO-standard ADAS labeling schema).
- **VIA CSV support (`via-csv`)**: VGG Image Annotator CSV export (separate format from VIA JSON).
- **YOLO Keras TXT support (`yolo-keras`)** and **YOLOv4 PyTorch TXT support (`yolov4-pytorch`)**: absolute-coordinate space-separated TXT files (`image_path xmin,ymin,xmax,ymax,class_id ...`). File-based or directory-based with named annotation files (`yolo_keras.txt`, `yolov4_pytorch.txt`, `train_annotation.txt`).

#### Auto-detection

- `.tfrecord` files: TFRecord framing + first-record probe for TFOD-style payloads.
- `.csv` content-based detection: 8 columns disambiguate TFOD vs Udacity by coordinate range/header; OpenImages, AutoML Vision, Kaggle Wheat, VoTT CSV detected by header signatures.
- `.xml` root-element disambiguation: `<annotations>` → CVAT, `<Page CropBox="...">` → Marmot.
- `.jsonl` / `.ndjson` / `.manifest`: Labelbox export-row shape is checked before SageMaker manifest rows.
- `.json`: object-root and array-root JSON now disambiguate Labelbox, Scale AI, Unity Perception, SuperAnnotate, Supervisely, Cityscapes, VoTT JSON, IBM Cloud Annotations, BDD100K, V7 Darwin, OpenLABEL, and Datumaro by schema markers.
- Directory layouts: Cityscapes (`gtFine/<split>/<city>/*_gtFine_polygons.json` or `gtFine/` root), Marmot (XML files with same-stem companion images), YOLO Keras / YOLOv4 PyTorch (matching named TXT files), Scale AI (`annotations/`), Unity Perception (SOLO-like), VoTT JSON (`vott-json-export/`), SuperAnnotate, Supervisely, Edge Impulse (containing directory), OIDv4 (`Label/` directory).

#### Tests and tooling

- New roundtrip tests for every adapter listed above (`tests/*_roundtrip.rs`).
- Property tests added/expanded for new adapters where appropriate.
- Shared internal helpers: `io_adapter_common.rs`, `io_bbox_adapters_common.rs`, `io_super_json_common.rs` extract common reader/writer machinery to keep adapters consistent.

### Changed

- `docs/formats.md`, `docs/cli.md`, `docs/tasks.md`, `docs/conversion.md`, and `docs/README.md` updated to cover all new adapters and detection rules.
- Conversion analyzers refactored to share lossiness-warning helpers (`add_common_csv_lossiness_warnings`, `add_annotation_drop_warnings`, etc.); TFOD and RetinaNet analyzers retroactively use the shared helpers, eliminating ~200 lines of duplicated code.
- TFRecord adapter exposes the `prost`-generated `Kind` oneof; an `enum_variant_names` clippy allow is applied at the type level.

### Dependencies

- Dependabot grouping reorganized into minor-patch and major groups for less PR noise.
- Cargo dependency batch updates (multiple groups) and GitHub Actions bumps:
  - `actions/checkout` 4 → 6
  - `astral-sh/setup-uv` 5 → 7
  - `docker/setup-buildx-action` 3 → 4
  - `docker/login-action` 3 → 4
  - `docker/setup-qemu-action` 3 → 4
  - `docker/build-push-action` 6 → 7

## v0.6.0

Five new format adapters, YOLO improvements, and CLI UX enhancements.

### Added

- **KITTI format support (`kitti`)**: standard autonomous driving annotation format. Per-image `.txt` files with 15 space-separated fields (type, truncated, occluded, alpha, bbox, 3D dims, location, rotation, optional score). Directory-based with `label_2/` + `image_2/`. Non-bbox KITTI fields preserved as `kitti_*` annotation attributes. Optional score field maps to IR confidence.
- **VGG Image Annotator JSON support (`via`)**: popular academic annotation tool format. Single JSON file keyed by `filename+size` with rectangle regions. Label resolution from `region_attributes` with `label`/`class`/sole-attribute precedence. VIA metadata preserved as `via_*` attributes.
- **RetinaNet Keras CSV support (`retinanet`)**: simple `path,x1,y1,x2,y2,class_name` CSV format used with keras-retinanet. Absolute pixel coordinates. Supports empty rows (`path,,,,,`) for unannotated images. Optional header row tolerated.
- **CSV auto-detection**: `.csv` files are now detected by content (8 columns → TFOD, 6 columns → RetinaNet) rather than always assuming TFOD.
- **VIA JSON auto-detection**: object-root JSON with entries containing `filename` + `regions` keys → VIA.
- **KITTI directory auto-detection**: directories with `label_2/` + `image_2/` → KITTI.
- **Fuzz targets**: `kitti_txt_parse`, `via_json_parse`, `retinanet_csv_parse` for parser fuzzing.
- **LabelMe JSON format support (`labelme`)**: per-image JSON annotation format with `shapes` array. Supports `rectangle` (2-point) and `polygon` (3+ point, flattened to axis-aligned bbox envelope) shape types. Reads single files, `annotations/` directory layouts, and co-located JSON+image directories. Writer produces canonical `annotations/` directory layout.
- **CreateML JSON format support (`create-ml`)**: Apple's annotation format for Core ML training. Flat JSON array with center-based absolute pixel coordinates (`{x, y, width, height}`). Image dimensions resolved from local image files. File-based read/write.
- **Auto-detection for LabelMe and CreateML**: `.json` files with `shapes` array → LabelMe; array-root JSON with `image`+`annotations` keys → CreateML. Empty JSON arrays are now ambiguous between Label Studio and CreateML (requires explicit `--from`).
- **Fuzz targets**: `labelme_json_parse` and `createml_json_parse` for parser fuzzing.
- **Property tests**: `proptest_labelme` and `proptest_createml` for roundtrip/idempotency checks.
- **YOLO optional confidence token**: YOLO label rows now accept an optional 6th float as a confidence score, mapped to IR `Annotation.confidence`. Rows with 7+ tokens are still rejected as segmentation/pose. The writer emits the 6th token only when confidence is present.
- **YOLO Darknet flat-directory layout**: flat `images/` + `labels/` layouts without `data.yaml` are now explicitly supported and tested. Class names are read from `classes.txt` when present, or inferred from label files (`class_0`, `class_1`, ...) when absent.
- `list-formats --output json` machine-readable format discovery, including aliases and file/directory layout hints
- `sample --output-format json` structured conversion/sample reports (with `--report` alias support)
- `convert --dry-run` to preview conversion/validation/lossiness results without writing output files
- `sample --dry-run` to preview sampled output and conversion/lossiness results without writing output files

### Changed

- Empty JSON array auto-detection now reports ambiguity between Label Studio and CreateML (previously assumed Label Studio)
- YOLO conversion reports no longer emit `drop_annotation_confidence` warnings (confidence is now preserved)
- YOLO `yolo_writer_data_yaml_policy` message now accurately describes the writer output (names mapping only, no split paths or nc)
- YOLO `yolo_writer_float_precision` message now mentions confidence alongside coordinates
- `validate` now uses typed enum parsing for `--format` and report output mode values
- `--output-format` is now the consistent cross-command spelling for structured stdout (`validate`, `stats`, `diff`, and `list-formats` also keep `--output`; `convert` and `sample` keep `--report` as an alias)
- Contributor guidance (`AGENTS.md`, `CLAUDE.md`) now reflects the current `stats` module and agent-oriented CLI usage guidance
- JSON stdout is now pretty-printed on TTYs and compact when piped or captured
- `stats` text output now switches to a plain text renderer when stdout is piped or captured

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
