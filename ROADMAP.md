# ROADMAP

This roadmap is intentionally high-level and may evolve.

For current, implemented behavior, use:
- [README.md](./README.md)
- [docs/README.md](./docs/README.md)
- [docs/tasks.md](./docs/tasks.md)

## Current baseline

- ✅ Detection task support (axis-aligned bboxes)
- ✅ Formats: IR JSON, COCO JSON, TFOD CSV, Ultralytics YOLO directory
- ✅ Conversion lossiness analysis and report JSON output
- ✅ CLI: convert, validate, inspect, list-formats

## Near-term priorities

Strategy: complete detection format coverage and add dataset utility commands
before expanding to new annotation tasks (segmentation, classification).

### Format support (detection)

- 🔧 Pascal VOC XML (in progress)
- ⏳ Label Studio JSON
- ⏳ CVAT XML
- ⏳ Broader YOLO family variants only when they fit IR safely

### CLI commands

- ⏳ `panlabel diff` — semantic diff between two datasets
- ⏳ `panlabel stats` — richer statistics (per-category distributions, bbox quality)
- ⏳ `panlabel sample` — subset a dataset (random, stratified, by category)

### Testing & robustness

- ⏳ Property-based testing (proptest: random IR → write → read → roundtrip compare)
- ⏳ Expand fuzz targets for new format parsers

### UX / CLI

- ⏳ Continue improving auto-detection ergonomics and clear error messages
- ⏳ Continue improving conversion policy explainability

### Documentation

- ⏳ Improve task docs and boundaries for detection workflows
- ⏳ Split docs into per-format/per-task pages when content volume justifies it
- ⏳ Keep docs tightly aligned with behavior covered by tests

## Later priorities

These are deferred until detection format coverage is solid:

### Task support

- ⏳ Evaluate IR design options for segmentation support
- ⏳ Evaluate IR design options for classification-only support

### Provider / workflow support

- ⏳ Provider-oriented documentation structure when real provider integrations are added
- ⏳ Better end-to-end examples for common training/export pipelines

## Change policy

- This file tracks direction and priority.
- It should not be treated as a strict release commitment.
- When priorities shift, update this file in the same PR/commit set as related docs changes.
