# ROADMAP

This roadmap is intentionally high-level and may evolve.

For current, implemented behavior, use:
- [README.md](./README.md)
- [docs/README.md](./docs/README.md)
- [docs/tasks.md](./docs/tasks.md)

## Current baseline

- ✅ Detection task support (axis-aligned bboxes)
- ✅ Formats: IR JSON, COCO JSON, CVAT XML, Label Studio JSON, LabelMe JSON, CreateML JSON, TFOD CSV, Ultralytics YOLO directory, Pascal VOC XML directory, Hugging Face ImageFolder metadata
- ✅ Conversion lossiness analysis and report JSON output
- ✅ CLI: convert, validate, stats, diff, sample, list-formats

## Near-term priorities

Strategy: complete detection format coverage and add dataset utility commands
before expanding to new annotation tasks (segmentation, classification).

### Format support (detection)

Goal: comprehensive coverage of every object detection annotation format in
active or legacy use, so that any dataset can be converted to any other format.

#### Already supported

- ✅ COCO JSON
- ✅ Pascal VOC XML (directory)
- ✅ CVAT XML (image-level)
- ✅ Label Studio JSON
- ✅ TFOD CSV (TensorFlow Object Detection)
- ✅ Ultralytics YOLO (directory, `images/` + `labels/`)
- ✅ Hugging Face ImageFolder (`metadata.jsonl` + `metadata.parquet`) — local read/write plus optional Parquet/remote Hub import (`--hf-repo`, feature-gated)
- ✅ HF zip-style remote split archives (`*.zip`) with payload routing to supported layouts
- ✅ LabelMe JSON (per-image JSON with `shapes` array — rectangle + polygon → bbox envelope)
- ✅ CreateML JSON (Apple's center-based absolute pixel bbox format)
- ✅ IR JSON (panlabel's lossless intermediate representation)

#### YOLO variant improvements

- ✅ YOLO split-aware reading — `data.yaml` with `train:`/`val:`/`test:` path keys (Roboflow, Ultralytics Hub exports)
- ✅ YOLO optional confidence token — parse optional 6th float as IR `confidence`
- ✅ YOLO Darknet flat-directory layout — single `images/` + `labels/` directory without `data.yaml`, `classes.txt` for class names
- ⏳ Scaled-YOLOv4 TXT — same token format as Ultralytics but different directory conventions
- ⏳ YOLOv4 PyTorch TXT — single annotations file listing `image_path x1,y1,x2,y2,class_id` per line (absolute pixel coords, not normalized)
- ⏳ YOLO Keras TXT — similar per-line format with absolute pixel coordinates

Note: YOLOv5/v6/v7 PyTorch TXT all use the same normalized `class cx cy w h`
format that panlabel already supports. Differences are in directory layout and
`data.yaml` structure, covered by the split-aware reading item above.

#### Dataset formats

- ✅ KITTI — space-separated `.txt` per image, 15 fields (type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score); standard in autonomous driving
- ⏳ OpenImages CSV — Google's large-scale detection format; CSV with ImageID, Source, LabelName, Confidence, and bbox columns
- ⏳ Cityscapes — JSON polygons per image with bounding box extraction; urban scene understanding (bbox subset of the polygon annotations)
- ⏳ Kaggle Wheat Format — CSV with `image_id, width, height, bbox` columns (`[xmin, ymin, w, h]` as a string); encountered in Kaggle competitions
- ⏳ Udacity Self-Driving Car Dataset — CSV with frame/xmin/ymin/xmax/ymax/label columns; legacy autonomous driving dataset
- ⏳ Marmot XML — XML format for document layout detection (table/figure regions); niche but datasets exist

#### Annotation-tool formats

- ⏳ Google Cloud AutoML Vision CSV — GCP's annotation/import format; CSV rows with `set,path,label,x1,y1,,,x2,y2,,`
- ⏳ Sagemaker GroundTruth Manifest — AWS annotation format; JSON Lines with `source-ref` and label job output as nested object per line
- ⏳ SuperAnnotate JSON — commercial annotation platform export; per-image JSON with `instances` array
- ⏳ Supervisely JSON — annotation platform with nested project structure; per-image JSON in a `ann/` directory with `objects` containing `classTitle` and geometry
- ⏳ Scale AI JSON — commercial data labeling export; JSON with `annotations` array, bbox as `{left, top, width, height}`
- ⏳ LabelBox JSON — Labelbox platform export (NDJSON); nested structure with `objects` containing `bbox` and `schemaId`
- ✅ VGG Image Annotator (VIA) JSON — popular academic tool; single JSON file keyed by `filename+size` with `regions` containing `shape_attributes`
- ⏳ VoTT JSON/CSV — Microsoft Video Object Tagging Tool (discontinued, but legacy datasets exist); per-asset JSON with `regions` array, or CSV export
- ⏳ IBM Cloud Annotations JSON — IBM Watson Visual Recognition export; JSON with `annotations` per image
- ⏳ Unity Perception JSON — Unity engine synthetic data output; per-frame JSON with `captures` containing 2D bbox annotations

#### Model-specific formats

- ⏳ TFRecords — TensorFlow binary protobuf format; harder to support (requires protobuf parsing), but widely used for TF training pipelines
- ✅ RetinaNet Keras CSV — simple `path,x1,y1,x2,y2,class_name` CSV (one row per annotation); used with keras-retinanet

### CLI commands

- ✅ `panlabel diff` — semantic diff between two datasets
- ✅ `panlabel stats` — richer statistics (per-category distributions, bbox quality)
- ✅ `panlabel sample` — subset a dataset (random, stratified, by category)

### Testing & robustness

- ✅ Property-based testing (proptest: IR JSON exact roundtrip, plus semantic roundtrip/idempotency checks for COCO, TFOD, Label Studio, VOC, YOLO, LabelMe, and CreateML)
- ✅ Expanded fuzz targets for parser surfaces (COCO JSON, VOC XML, TFOD CSV, Label Studio JSON, IR JSON, YOLO line parser, LabelMe JSON, CreateML JSON)

### UX / CLI

- ⏳ Continue improving auto-detection ergonomics and clear error messages
- ✅ Conversion policy explainability: blocked paths emit full reports, text output shows stable `[code]` brackets, `stage` field in JSON schema, COCO/HF/Label Studio/YOLO adapter policy notes, drift-prevention tests

### Documentation

- ✅ Improve task docs and boundaries for detection workflows
- ⏳ Split docs into per-format/per-task pages when content volume justifies it
- ✅ Keep docs tightly aligned with behavior covered by tests
- ✅ Keep Label Studio docs aligned with `src/ir/io_label_studio_json.rs` + `tests/label_studio_roundtrip.rs` (strict schema, legacy `completions`, rotation-envelope behavior)
- ⏳ Format Museum — illustrated archive of every annotation format with history, examples, real dataset links, and timeline (see `design/format-museum.md`); doubles as SEO content and panlabel conversion funnel

## Later priorities

These are deferred until detection format coverage is solid:

### Task support

- ⏳ Evaluate IR design options for segmentation support
- ⏳ Evaluate IR design options for classification-only support

### YOLO variants (blocked by IR design)

- ⏳ YOLO OBB (8-token oriented bbox rows) — requires rotated-bbox IR support
- ⏳ YOLO segmentation (variable-length polygon rows) — requires polygon/mask IR support
- ⏳ YOLO pose (keypoint rows) — requires keypoint IR support

### Provider / workflow support

- ⏳ Provider-oriented documentation structure when real provider integrations are added
- ⏳ Better end-to-end examples for common training/export pipelines
- ⏳ HF remote support for `stats` / `sample` / `diff` (remote is currently `convert`-only)
- ⏳ HF self-contained/viewer-converted Parquet variants (embedded image bytes and Hub auto-converted parquet refs)
- ⏳ HF split-parquet + external-images layouts that currently fail in the wild (for example `KRAFTON/ArtiBench`)
- ⏳ HF Hub streaming / partial reads

## Change policy

- This file tracks direction and priority.
- It should not be treated as a strict release commitment.
- When priorities shift, update this file in the same PR/commit set as related docs changes.
