# ROADMAP

This roadmap is intentionally high-level and may evolve.

For current, implemented behavior, use:
- [README.md](./README.md)
- [docs/README.md](./docs/README.md)
- [docs/tasks.md](./docs/tasks.md)

## Current baseline

- ✅ Detection task support (axis-aligned bboxes)
- ✅ Formats: IR JSON, COCO JSON, Label Studio JSON, TFOD CSV, Ultralytics YOLO directory, Pascal VOC XML directory
- ✅ Conversion lossiness analysis and report JSON output
- ✅ CLI: convert, validate, stats, diff, sample, list-formats

## Near-term priorities

Strategy: complete detection format coverage and add dataset utility commands
before expanding to new annotation tasks (segmentation, classification).

### Format support (detection)

- ✅ Pascal VOC XML
- ✅ Label Studio JSON
- ✅ CVAT XML
- ⏳ Broader YOLO family variants only when they fit IR safely
- ⏳ Hugging Face Datasets ImageFolder (`metadata.jsonl`) — read/write HF's standard object detection format (COCO xywh bbox convention by default, with `--hf-bbox-format xyxy` for `detection-datasets`-style uploads). Pure JSON, no Arrow/Parquet dependency. See [`design/huggingface-datasets-research.md`](./design/huggingface-datasets-research.md) for full research notes.

### CLI commands

- ✅ `panlabel diff` — semantic diff between two datasets
- ✅ `panlabel stats` — richer statistics (per-category distributions, bbox quality)
- ✅ `panlabel sample` — subset a dataset (random, stratified, by category)

### Testing & robustness

- ✅ Property-based testing (proptest: IR JSON exact roundtrip, plus semantic roundtrip/idempotency checks for COCO, TFOD, Label Studio, VOC, and YOLO)
- ✅ Expanded fuzz targets for parser surfaces (COCO JSON, VOC XML, TFOD CSV, Label Studio JSON, IR JSON, YOLO line parser)

### UX / CLI

- ⏳ Continue improving auto-detection ergonomics and clear error messages
- ⏳ Continue improving conversion policy explainability

### Documentation

- ⏳ Improve task docs and boundaries for detection workflows
- ⏳ Split docs into per-format/per-task pages when content volume justifies it
- ⏳ Keep docs tightly aligned with behavior covered by tests
- ✅ Keep Label Studio docs aligned with `src/ir/io_label_studio_json.rs` + `tests/label_studio_roundtrip.rs` (strict schema, legacy `completions`, rotation-envelope behavior)

## Later priorities

These are deferred until detection format coverage is solid:

### Task support

- ⏳ Evaluate IR design options for segmentation support
- ⏳ Evaluate IR design options for classification-only support

### Provider / workflow support

- ⏳ Provider-oriented documentation structure when real provider integrations are added
- ⏳ Better end-to-end examples for common training/export pipelines
- ⏳ HF Datasets Parquet direct reading (via `arrow-rs`) for Hub dataset conversion without Python export step
- ⏳ HF Hub streaming / URL-based dataset loading

## Change policy

- This file tracks direction and priority.
- It should not be treated as a strict release commitment.
- When priorities shift, update this file in the same PR/commit set as related docs changes.
