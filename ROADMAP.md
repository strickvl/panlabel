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

## Near-term priorities

### Task support

- ⏳ Improve task docs and boundaries for detection workflows
- ⏳ Evaluate IR design options for segmentation support
- ⏳ Evaluate IR design options for classification-only support

### Format support

- ⏳ Additional multi-file formats (e.g. Pascal VOC family)
- ⏳ Broader YOLO family variants only when they fit IR safely

### Provider / workflow support

- ⏳ Provider-oriented documentation structure when real provider integrations are added
- ⏳ Better end-to-end examples for common training/export pipelines

### UX / CLI

- ⏳ Continue improving auto-detection ergonomics and clear error messages
- ⏳ Continue improving conversion policy explainability

### Documentation

- ⏳ Split docs into per-format/per-task pages when content volume justifies it
- ⏳ Keep docs tightly aligned with behavior covered by tests

## Change policy

- This file tracks direction and priority.
- It should not be treated as a strict release commitment.
- When priorities shift, update this file in the same PR/commit set as related docs changes.
