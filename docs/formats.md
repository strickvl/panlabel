# Formats

This page describes how each annotation format works inside panlabel — what gets
read, what gets written, and what you should expect.

Panlabel converts through a canonical intermediate representation (IR). All
bounding boxes are represented as **pixel-space XYXY** in the IR, and each
format adapter handles the mapping to/from its own coordinate system.

Current scope: **object detection** bounding boxes only.

## Format matrix

| Format | Path kind | Read | Write | Lossiness vs IR |
|---|---|---|---|---|
| `ir-json` | file (`.json`) | yes | yes | lossless |
| `coco` | file (`.json`) | yes | yes | conditional |
| `ibm-cloud-annotations` | file (`_annotations.json`) or directory | yes | yes | lossy |
| `cvat` | file (`.xml`) or directory (`annotations.xml`) | yes | yes | lossy |
| `label-studio` | file (`.json`) | yes | yes | lossy |
| `labelbox` | file (`.json`, `.jsonl`, `.ndjson`) | yes | yes | lossy |
| `scale-ai` | file (`.json`) or directory (`annotations/` or co-located JSONs) | yes | yes | lossy |
| `unity-perception` | file (`.json`) or SOLO-like directory | yes | yes (directory only) | lossy |
| `tfod` | file (`.csv`) | yes | yes | lossy |
| `tfrecord` | file (`.tfrecord`) | yes | yes | lossy |
| `vott-csv` | file (`.csv`) | yes | yes | lossy |
| `vott-json` | file (`.json`) or directory (`vott-json-export/`) | yes | yes | lossy |
| `yolo` | directory (`images/` + `labels/`) or split image-list `.txt` via `data.yaml` | yes | yes | lossy |
| `yolo-keras` | file (`.txt`) or directory (`yolo_keras.txt`, `annotations.txt`, `train.txt`) | yes | yes | lossy |
| `yolov4-pytorch` | file (`.txt`) or directory (`yolov4_pytorch.txt`, `train_annotation.txt`, `train.txt`) | yes | yes | lossy |
| `voc` | directory (`Annotations/` + `JPEGImages/`) | yes | yes | lossy |
| `hf` | directory (`metadata.jsonl` / `metadata.parquet`) | yes | yes (`metadata.jsonl`) | lossy |
| `sagemaker` | file (`.manifest` / `.jsonl`) | yes | yes | lossy |
| `labelme` | file (`.json`) or directory (`annotations/`) | yes | yes | lossy |
| `superannotate` | file (`.json`) or directory (`annotations/` or co-located JSONs) | yes | yes | lossy |
| `supervisely` | file (`.json`) or directory (`ann/` dataset or `meta.json` project) | yes | yes | lossy |
| `cityscapes` | file (`.json`), `gtFine/`, or dataset root with `gtFine/` | yes | yes | lossy |
| `marmot` | file (`.xml`) or directory of `.xml` files with companion images | yes | yes | lossy |
| `create-ml` | file (`.json`) | yes | yes | lossy |
| `kitti` | directory (`label_2/` + `image_2/`) | yes | yes | lossy |
| `via` | file (`.json`) | yes | yes | lossy |
| `retinanet` | file (`.csv`) | yes | yes | lossy |
| `openimages` | file (`.csv`) | yes | yes | lossy |
| `kaggle-wheat` | file (`.csv`) | yes | yes | lossy |
| `automl-vision` | file (`.csv`) | yes | yes | lossy |
| `udacity` | file (`.csv`) | yes | yes | lossy |

## IR JSON (`ir-json`)

- Canonical panlabel representation.
- Preserves dataset info, licenses, image metadata, and annotation attributes.
- Bboxes are stored in XYXY form.

## COCO JSON (`coco` / `coco-json`)

- Path kind: JSON file.
- Bbox format: `[x, y, width, height]` (absolute pixel coordinates).
- Converted to IR XYXY via bbox helpers.
- Writer behavior is deterministic (stable ordering by IDs).
- COCO `score` can map to IR `confidence` when present.
- COCO `segmentation` is accepted on read but ignored/dropped (panlabel currently models detection bboxes only). On write, panlabel emits `segmentation` as an empty array.

## Label Studio JSON (`label-studio` / `label-studio-json` / `ls`)

- Path kind: JSON file.
- Supported shape: Label Studio task export array (empty array is accepted as an empty dataset).
- Supported annotation type: `rectanglelabels` only.
- Coordinates are percentages; adapter maps to/from IR pixel XYXY.
- Reader supports legacy `completions` as fallback when `annotations` is absent.
- Label Studio result `score` (when present) maps to IR `confidence` (from either `annotations` or `predictions`).

Reader behavior:
- derives `Image.file_name` from `data.image` basename (normalizes `\` to `/`, strips query/fragment)
- requires derived basenames to be unique across tasks
- preserves full image reference in `Image.attributes["ls_image_ref"]`
- accepts either `annotations` or legacy `completions` per task (both present is an error)
- supports `predictions` alongside annotation sets
- each of `annotations` / `completions` / `predictions` may contain at most one result-set entry
- enforces `type == "rectanglelabels"` and exactly one label per result
- requires `original_width`/`original_height` on each result; if a task has zero results, falls back to `data.width`/`data.height`
- requires consistent `from_name`/`to_name` values within a task; when present, stores them in `Image.attributes["ls_from_name"]` and `Image.attributes["ls_to_name"]`
- stores non-zero rotation as `Annotation.attributes["ls_rotation_deg"]` and uses an axis-aligned envelope bbox in IR

Deterministic policy:
- reader image IDs: by derived basename (lexicographic)
- reader category IDs: by label name (lexicographic)
- reader annotation IDs: by image order then result order
- writer task order: by image file_name (lexicographic)

Writer behavior:
- writes Label Studio task export JSON
- splits results by confidence:
  - `confidence == None` -> `annotations`
  - `confidence == Some(_)` -> `predictions` + `score`
  - this means any IR annotation with confidence is written under `predictions`
- uses `ls_from_name` / `ls_to_name` image attributes if present, else defaults to `label` / `image`
- requires unique image basenames (derived from `data.image`) to avoid ambiguous `Image.file_name` mapping

Limitations:
- currently only rectanglelabels bbox annotations are supported
- rotation is flattened to axis-aligned geometry (angle retained as `ls_rotation_deg` only)
- Label Studio-specific metadata outside this mapping is not preserved

## Labelbox JSON/NDJSON (`labelbox` / `labelbox-json` / `labelbox-ndjson`)

- Path kind: JSON/JSONL/NDJSON file.
- Supported input shapes:
  - `.jsonl` / `.ndjson` with one Labelbox export row per line
  - single JSON export-row object
  - JSON array of export-row objects
- Supported row shape: `data_row`, `media_attributes`, and nested `projects.*.labels[].annotations.objects[]`.
- Bounding boxes use Labelbox `bounding_box` / `bbox` (`left`, `top`, `width`, `height`) and map directly to IR pixel-space XYXY.
- Polygons use the envelope of their point array and are marked with `labelbox_polygon_enveloped=true`.
- Unsupported non-detection object kinds such as points, masks, and lines are skipped with warnings; the image row is still preserved.

Deterministic policy:
- reader image IDs: by derived `file_name` (lexicographic)
- reader category IDs: by label name (lexicographic)
- reader annotation IDs: by image order, then project ID, label index, and object index
- writer rows: ordered by image file_name
- writer objects: ordered by annotation ID

Writer behavior:
- `.jsonl` and `.ndjson` outputs write newline-delimited Labelbox export rows
- other outputs write a JSON array of rows
- emits all IR boxes as `ImageBoundingBox` objects with `bounding_box` geometry
- preserves images without annotations as rows with empty `objects`
- does **not** copy image binaries

Limitations:
- no dataset-level metadata/licenses
- no category supercategory
- no annotation confidence in writer output
- polygons are flattened to axis-aligned bbox envelopes on read
- segmentation masks, points, lines, and classifications are not represented in IR detection output

## IBM Cloud Annotations JSON (`ibm-cloud-annotations` / `cloud-annotations`)

- Path kind: `_annotations.json` file or directory containing `_annotations.json`.
- Supported type: `"localization"`.
- Coordinates are normalized `x`, `y`, `x2`, `y2`; the reader converts them to IR pixel-space XYXY by probing image dimensions.
- Image lookup tries `<json_dir>/<image>` and then `<json_dir>/images/<image>`.

Deterministic policy:
- reader image IDs: by image key (lexicographic)
- reader category IDs: by the source `labels` array, with extra labels appended lexicographically if annotations mention labels absent from `labels`
- writer labels: by IR category ID
- writer image keys: by `Image.file_name`; objects by annotation ID

Writer behavior:
- writes a Cloud Annotations-style localization JSON object
- file outputs write the requested JSON file; directory outputs write `_annotations.json` plus `images/README.txt`
- does **not** copy image binaries

Limitations:
- no dataset-level metadata/licenses
- no image-level license/date metadata
- no annotation confidence/attributes

## TFOD CSV (`tfod` / `tfod-csv`)

- Path kind: CSV file.
- Columns: `filename,width,height,class,xmin,ymin,xmax,ymax`.
- Coordinates are normalized (0..1).

Deterministic policy:
- reader image IDs: by filename (lexicographic)
- reader category IDs: by class name (lexicographic)
- reader annotation IDs: by CSV row order
- writer row order: by annotation ID

Limitations:
- no dataset-level metadata/licenses
- no image-level license/date metadata
- no annotation confidence/attributes
- images without annotations are not represented in TFOD output

## TFRecord (`tfrecord` / `tfrecords` / `tf-record` / `tfod-tfrecord` / `tfod-tfrerecord`)

- Path kind: single `.tfrecord` file.
- V1 scope: **uncompressed TFOD-style `tf.train.Example` object-detection bbox records only**.
- TFRecord is a container format; arbitrary payloads are intentionally out of scope in v1.
- Bounding boxes use normalized `xmin/xmax/ymin/ymax` feature lists and map to/from IR pixel-space XYXY.
- One TFRecord Example maps to one image plus zero or more objects.

Deterministic policy:
- reader image IDs: by filename (lexicographic)
- reader category IDs: by class name (lexicographic)
- reader annotation IDs: by record order then object order
- writer example order: by image filename then image ID
- writer object order: by annotation ID

Limitations:
- no dataset-level metadata/licenses
- no image-level license/date metadata
- arbitrary/non-TFOD Example payloads are not supported
- sharded directories, compression, and embedded image-byte roundtrip are out of scope in v1

## VoTT CSV (`vott-csv` / `vott`)

- Path kind: CSV file.
- Columns: headered `image,xmin,ymin,xmax,ymax,label`.
- Coordinates are absolute pixel-space XYXY values.
- Image dimensions are not stored in the CSV; reader lookup tries `<csv_dir>/<image>` then `<csv_dir>/images/<image>`.

Deterministic policy:
- reader image IDs: by image path (lexicographic)
- reader category IDs: by label (lexicographic)
- reader annotation IDs: by sorted row content
- writer row order: by image filename, then annotation ID

Limitations:
- no dataset-level metadata/licenses
- no image-level license/date metadata
- no annotation confidence/attributes
- images without annotations are not represented in VoTT CSV output

## Scale AI JSON (`scale-ai` / `scale` / `scale-ai-json`)

- Path kind: JSON file or directory.
- Supported input shapes:
  - Scale task object with `params` and optional `response.annotations`
  - callback/response object with `response.annotations` and optional nested `task`
  - response object with root `annotations`
  - JSON array of task/response objects
  - directory with Scale JSON files under `annotations/`, or matching root-level JSON files
- Plain boxes use Scale `left`, `top`, `width`, `height` and map directly to IR pixel-space XYXY.
- Polygons use the envelope of their `vertices` array and are marked with `scale_ai_enveloped=true` and `scale_ai_geometry_type=polygon`.
- Rotated boxes with `vertices` use the envelope of those vertices and preserve `rotation` as `scale_ai_rotation_rad`.
- Unsupported geometry types such as lines, points, cuboids, and ellipses are rejected clearly instead of being silently skipped.

Deterministic policy:
- reader image IDs: by derived `file_name` (lexicographic)
- reader category IDs: by label name (lexicographic)
- reader annotation IDs: by image order, then source annotation order
- writer task objects: ordered by image file_name
- writer annotations: ordered by annotation ID

Writer behavior:
- single-image file outputs write one Scale-like `imageannotation` task object
- multi-image file outputs write a JSON array of task objects
- directory outputs write `annotations/<image-stem>.json` plus `images/README.txt`
- emits all IR boxes as `type: "box"` response annotations with `left`/`top`/`width`/`height`
- preserves images without annotations as task objects with empty `response.annotations`
- does **not** copy image binaries

Limitations:
- no dataset-level metadata/licenses
- no category supercategory
- annotation confidence is not represented
- non-Scale annotation attributes are not represented unless they are already `scale_ai_attribute_*` attributes

## Unity Perception JSON (`unity-perception` / `unity` / `solo`)

- Path kind: SOLO frame JSON file, narrow legacy `captures` JSON file, or directory containing SOLO frame/captures JSON files.
- Supported annotation type: Unity/SOLO `BoundingBox2D` only.
- Bounding boxes import from `values` entries using either `x`, `y`, `width`, `height` or `origin` + `dimension`.
- Non-bbox annotation blocks such as segmentation/keypoints are skipped with warnings; the capture/image row is still preserved.
- Image dimensions come from capture `dimension`, then local image probing, then bbox extents as a last resort.

Deterministic policy:
- reader image IDs: by derived `file_name` (lexicographic)
- reader category IDs: by `annotation_definitions.json` label order when available, then extra label names lexicographically
- reader annotation IDs: by image order, then frame annotation/value order
- writer frames: ordered by image file_name
- writer bbox values: ordered by annotation ID

Writer behavior:
- emits directory datasets only; `.json` file output is rejected as ambiguous
- writes `annotation_definitions.json` plus `sequence.0/step*.frame_data.json`
- emits all IR boxes as `BoundingBox2DAnnotation` values using `x`/`y`/`width`/`height`
- preserves images without annotations as frame captures with empty bbox values
- writes `images/README.txt` and does **not** copy image binaries

Limitations:
- no dataset-level metadata/licenses
- no category supercategory
- annotation confidence is not represented
- non-bbox Unity annotations are not represented in the IR detection output

## VoTT JSON (`vott-json` / `vott-json-export`)

- Path kind: JSON file or directory.
- Supported file shapes:
  - aggregate project JSON with top-level `assets`
  - per-asset JSON with top-level `asset` and `regions`
- Supported directory shapes:
  - `vott-json-export/panlabel-export.json`
  - root `panlabel-export.json`
  - top-level per-asset `.json` files when `--from vott-json` is used explicitly
- Rectangle regions use VoTT `boundingBox` (`left`, `top`, `width`, `height`) and map to IR pixel-space XYXY.
- Polygon-like regions without `boundingBox` use the envelope of their `points` array.
- Regions with multiple `tags` expand to one IR annotation per tag.
- Image dimensions come from `asset.size` when present; otherwise the reader probes `<json_dir>/<image>`, `<json_dir>/images/<image>`, and local `file:` asset paths.

Deterministic policy:
- reader image IDs: by image filename (lexicographic)
- reader category IDs: by source project `tags` order, with extra region tags appended lexicographically
- reader annotation IDs: by sorted image order, then source region order, then tag order
- writer assets: ordered by image filename
- writer regions: ordered by annotation ID

Writer behavior:
- file outputs write a deterministic aggregate VoTT JSON project to the requested `.json` path
- directory outputs write `vott-json-export/panlabel-export.json` plus `vott-json-export/images/README.txt`
- emits all IR boxes as `RECTANGLE` regions with `boundingBox` and corner `points`
- preserves images without annotations as assets with empty `regions`
- does **not** copy image binaries

Limitations:
- no dataset-level metadata/licenses beyond a simple project name
- no image-level license/date metadata
- no annotation confidence/attributes in writer output
- polygon point geometry is flattened to an axis-aligned bbox envelope on read

## YOLO directory (`yolo` / `ultralytics` / `yolov8` / `yolov5` / `scaled-yolov4` / `scaled-yolov4-txt`)

- Path kind: directory.
- Accepted input path:
  - dataset root containing `images/` and `labels/`
  - or `labels/` directory directly (with sibling `../images/`)
- Supports both flat layouts (Darknet-style, no `data.yaml` required) and split-aware layouts.
- Label row format (one line per bbox):
  - `<class_id> <x_center> <y_center> <width> <height> [confidence]`
  - normalized values
  - 5 tokens: detection bbox (confidence = None)
  - 6 tokens: detection bbox + confidence score (mapped to IR `Annotation.confidence`)
  - 7+ tokens: rejected (segmentation/pose not supported)

Reader behavior:
- class map precedence: `data.yaml` → `classes.txt` → inferred from labels
- flat layouts work without `data.yaml`: class names come from `classes.txt` (if present) or are inferred as `class_0`, `class_1`, etc.
- image resolution is read from image headers in `images/`
- each label file must map to a matching image file (same relative stem) under `images/`
- expected image extensions (lookup order): `jpg`, `png`, `jpeg`, `bmp`, `webp`
- lines with 7+ tokens are rejected (segmentation/pose not supported)

### Split-aware reading

When `data.yaml` contains `train:`, `val:`, or `test:` path keys (common in Roboflow/Ultralytics Hub exports), panlabel detects a split-aware layout and reads all splits.

Supported path patterns in `data.yaml`:
- Pattern A: `images/<split>` (e.g. `train: images/train`, labels inferred at `labels/train`)
- Pattern B: `<split>/images` (e.g. `train: train/images`, labels at `train/labels`)
- Pattern C: bare `<split>` pointing to a directory containing `images/` + `labels/`
- Pattern D: image-list `.txt` file (e.g. `train: train.txt`, common in Scaled-YOLOv4-style exports)

Behavior:
- **Default (no `--split`):** all found splits are merged into a single IR Dataset. Image `file_name` values are prefixed with the split name (e.g. `train/img001.jpg`, `val/img002.jpg`) to avoid collisions.
- **`--split <name>`:** only the named split is read. Image `file_name` values are still prefixed with the split name for provenance.
- Class map: resolved from `data.yaml` `names:` when present, otherwise inferred from the selected label files.
- `data.yaml` `path:` key (if present) is used as the base for resolving split-relative paths.
- For image-list `.txt` splits, each non-empty non-comment row is an image path. Relative rows resolve relative to the list file's parent directory.
- For image-list `.txt` splits, label paths are derived from each image path by replacing the rightmost `images` path component with `labels` and changing the extension to `.txt`; if that label file is absent, panlabel falls back to a same-directory `.txt` next to the image. A missing label file means the image has no annotations.
- Image-list logical image names are deterministic. If two rows would produce the same split-prefixed logical name, panlabel errors instead of silently merging them.
- Split provenance is stored in `Dataset.info.attributes`:
  - `yolo_layout_mode`: `"split_aware"` or `"flat"`
  - `yolo_splits_found`: comma-separated list of splits found (e.g. `"train,val,test"`)
  - `yolo_splits_read`: comma-separated list of splits actually read
- An error is raised if `--split` names a split not present in `data.yaml`, or if `--split` is used on a flat (non-split-aware) layout.

Writer behavior:
- creates output `images/` and `labels/` directories
- writes `data.yaml` with a `names:` mapping (sorted by class index); does not emit train/val paths or `nc`
- creates empty `.txt` files for images without annotations
- does **not** copy image binaries
- writes normalized floats with 6 decimal places
- emits an optional 6th confidence token when `Annotation.confidence` is `Some`

## YOLO Keras / YOLOv4 PyTorch TXT (`yolo-keras`, `yolov4-pytorch`)

These two public formats share one adapter because their object-detection TXT
shape is the same:

```text
<image_ref> [xmin,ymin,xmax,ymax,class_id ...]
```

Reader behavior:
- accepts a single `.txt` annotation file, or a directory containing a canonical annotation file
- canonical directory search for `yolo-keras`: `yolo_keras.txt`, `yolo-keras.txt`, `annotations.txt`, `train_annotations.txt`, then `train.txt`
- canonical directory search for `yolov4-pytorch`: `yolov4_pytorch.txt`, `yolov4-pytorch.txt`, `yolov4_train.txt`, `train_annotation.txt`, `train_annotations.txt`, then `train.txt`
- each box token is absolute pixel-space XYXY plus a zero-based class ID
- a row with only `image_ref` is kept as an unannotated image
- class names come from `classes.txt`, `class_names.txt`, `classes.names`, or `obj.names`; missing names fall back to `class_<id>`
- image dimensions are probed from disk: relative refs are tried beside the annotation file/directory first, then under `images/`
- malformed boxes include the annotation file and line number in the error

Writer behavior:
- writes deterministic rows ordered by image `file_name`, with boxes ordered by annotation ID
- writes `classes.txt` ordered by category ID; class IDs in rows are zero-based positions in that order
- writes image-only rows for unannotated images
- creates only annotation/class files; image binaries are not copied

Auto-detection note: the row grammar cannot distinguish YOLO Keras from YOLOv4
PyTorch. A specifically named file such as `yolo_keras.txt` or
`yolov4_pytorch.txt` can be auto-detected. Shared/generic names such as
`train.txt` or `train_annotations.txt` that match this grammar are reported as
ambiguous; use `--from yolo-keras` or `--from yolov4-pytorch`.

## Pascal VOC XML (`voc` / `pascal-voc` / `voc-xml`)

- Path kind: directory.
- Accepted input path:
  - dataset root containing `Annotations/`
  - or `Annotations/` directory directly (with optional sibling `../JPEGImages/`)
- Reader uses `<size>/<width>` and `<size>/<height>` from XML (no image-header probing).
- Reader stores object fields `pose`, `truncated`, `difficult`, `occluded` in `Annotation.attributes`.
- Reader stores `<size>/<depth>` as image attribute `depth`.
- Coordinate policy: reads `xmin/ymin/xmax/ymax` exactly as provided (no 0/1-based adjustment).
- Reader scans `Annotations/` flat (non-recursive); nested XML files are skipped with a warning.

Deterministic policy:
- reader image IDs: by `<filename>` (lexicographic)
- reader category IDs: by class name (lexicographic)
- reader annotation IDs: by XML file order, then `<object>` order

Writer behavior:
- creates `Annotations/` and `JPEGImages/README.txt`
- writes one XML per image (including images without annotations)
- preserves image subdirectory structure in XML output path (`train/001.jpg` -> `Annotations/train/001.xml`)
- does **not** copy image binaries
- normalizes boolean attribute values when writing:
  - `true`/`yes`/`1` -> `1`
  - `false`/`no`/`0` -> `0`
  - any other value -> omitted

## Hugging Face ImageFolder metadata (`hf` / `hf-imagefolder` / `huggingface`)

- Path kind: directory.
- Accepted local input layout:
  - dataset root containing `metadata.jsonl` or `metadata.parquet`
  - split subdirectories (for example `train/`, `validation/`) each containing metadata
  - parquet shard layouts (for example `data/train-00000-of-00001.parquet`, `data/validation-*.parquet`, or `<config>/<split>/*.parquet`)
- Remote Hub import is supported in `convert` via `--hf-repo` (requires `hf-remote` feature).
- Remote zip-style split archives (for example `data/train.zip`) are also supported when they extract to YOLO, VOC, COCO JSON, or HF metadata layouts.

Reader behavior:
- object-container auto-detection: `objects` first, then `faces` (override with `--hf-objects-column`)
- category field aliases: `categories` or `category`
- category values may be names or integer IDs
- integer category name resolution precedence:
  - preflight ClassLabel names (remote)
  - then `--hf-category-map`
  - then integer fallback (`"0"`, `"1"`, ...)
- bbox interpretation is controlled by `--hf-bbox-format`:
  - `xywh` (default) treats bbox as `[x, y, width, height]`
  - `xyxy` treats bbox as `[x1, y1, x2, y2]`
- keeps bbox rows as parsed (validation reports degenerate/OOB issues later)
- width/height read from metadata when present, otherwise from image headers
- duplicate `file_name` rows are rejected
- when both `metadata.jsonl` and `metadata.parquet` are present, JSONL is preferred
- when no `metadata.jsonl` exists, panlabel can read supported parquet layouts (`metadata.parquet` or split parquet shards) with `hf-parquet`
- for parquet rows without `file_name`, panlabel derives it from `image.path` (or fallback IDs)

Writer behavior:
- writes `metadata.jsonl` (one row per image)
- writes `file_name`, `width`, `height`, and `objects.{bbox,categories}`
- deterministic output ordering:
  - metadata rows by image `file_name` (lexicographic)
  - per-image annotation lists by annotation ID
- does **not** copy image binaries
- output bbox format follows `--hf-bbox-format` (`xywh` default)

IR provenance notes:
- reader stores HF provenance in `Dataset.info.attributes` (for example `hf_bbox_format`)
- remote imports may also populate `hf_repo_id`, `hf_revision`, `hf_split`, `hf_license`, `hf_description`

## SageMaker Ground Truth Manifest (`sagemaker` / `sagemaker-manifest` / `sagemaker-ground-truth` / `ground-truth` / `groundtruth` / `aws-sagemaker`)

- Path kind: JSON Lines file (`.manifest` or `.jsonl`).
- Scope: annotated **object-detection only** (`groundtruth/object-detection`).
- One JSON object per line with:
  - `source-ref` (image reference)
  - dynamic label attribute object (commonly `bounding-box`) with `annotations` and `image_size`
  - matching `<label>-metadata` object
- Bboxes are read as absolute pixel `left/top/width/height` and converted to IR XYXY.

Reader behavior:
- auto-detects a single object-detection label attribute per row
- rejects ambiguous rows (multiple candidate label attributes) and manifests mixing label attribute names across rows
- resolves category names from metadata `class-map`; falls back to numeric `class_id` strings when needed
- preserves per-object confidence from `<label>-metadata.objects[].confidence` to IR `Annotation.confidence`
- preserves source and metadata provenance in attributes (`sagemaker_source_ref`, `sagemaker_label_attribute_name`, etc.)

Deterministic policy:
- reader image IDs: by derived `file_name` (lexicographic)
- reader category IDs: by numeric source `class_id` order
- reader annotation IDs: by sorted image order then source annotation order
- writer rows: sorted by `Image.file_name` (lexicographic)

Writer behavior:
- emits deterministic JSONL with one row per image (including unannotated images)
- label attribute name:
  - uses `Dataset.info.attributes["sagemaker_label_attribute_name"]` when present
  - otherwise defaults to `bounding-box`
- metadata defaults are deterministic: `type=groundtruth/object-detection`, `human-annotated=yes`, `job-name=panlabel-export`
- writes category `class-map` for all categories and assigns output class IDs by `CategoryId` order
- does **not** copy image binaries

Limitations:
- object-detection manifests only (segmentation/classification Ground Truth task types are rejected)
- one label attribute per manifest (mixed or ambiguous attributes are rejected)
- no S3 probing for image dimensions (dimensions come from manifest `image_size`)

## CVAT XML (`cvat` / `cvat-xml`)

- Path kind: XML file (`.xml`) or directory containing `annotations.xml`.
- Supported export: CVAT "for images" XML with `<annotations>` root.
- Supported annotation type: `<box>` only.
- Unsupported image-level annotation elements (for example `<polygon>`, `<points>`) are hard parse errors.
- Coordinates: absolute pixels (`xtl/ytl/xbr/ybr`) mapped 1:1 to IR pixel XYXY.

Reader behavior:
- accepts file input or directory input with root `annotations.xml`
- if `<meta><task><labels>` is present:
  - keeps labels with `<type>bbox</type>` (or no `<type>`)
  - verifies every `<box label="...">` exists in meta labels
- if meta labels are missing, infers categories from `<box label="...">`
- stores `<image id>` as `Image.attributes["cvat_image_id"]`
- stores box attributes as:
  - `occluded="1"` -> `Annotation.attributes["occluded"] = "1"`
  - non-zero `z_order` -> `Annotation.attributes["z_order"]`
  - non-empty `source` -> `Annotation.attributes["source"]`
  - `<attribute name="k">v</attribute>` -> `Annotation.attributes["cvat_attr_k"] = "v"`

Deterministic policy:
- reader image IDs: by `<image name>` (lexicographic)
- reader category IDs: by label name (lexicographic)
- reader annotation IDs: by image order then `<box>` order

Writer behavior:
- writes a single XML file (or `annotations.xml` inside output directory)
- emits minimal `<meta><task>` with `name='panlabel export'`, `mode='annotation'`, and `size` equal to image count
- writes labels only for categories referenced by annotations (unused categories are dropped)
- writes `<image>` entries for all images, including unannotated images
- image ordering: by `file_name` (lexicographic)
- image IDs are reassigned sequentially (0, 1, 2, ...) by sorted order; original `cvat_image_id` attributes are not preserved in output
- writes `<box>` entries sorted by annotation ID per image
- writes `cvat_attr_*` annotation attributes as `<attribute>` children of `<box>`
- normalizes `occluded` values:
  - `true`/`yes`/`1` -> `1`
  - `false`/`no`/`0` -> `0`
  - otherwise or missing -> `0`
- defaults missing or empty `source` attribute to `manual`
- defaults missing or invalid `z_order` to `0`

## LabelMe JSON (`labelme` / `labelme-json`)

- Path kind: JSON file or directory.
- One JSON file per image containing a `shapes` array with rectangle and polygon annotations.
- Supported shapes: `rectangle` (2 points: top-left, bottom-right), `polygon` (3+ points: converted to axis-aligned bbox envelope). Other shape types are rejected.
- Coordinates: absolute pixels.
- Missing `shape_type` defaults to `rectangle`.

Reader input modes:
- **Single file**: one `.json` file → one-image dataset
- **Separate directory**: `annotations/` subdirectory containing `.json` files
- **Co-located directory**: `.json` files alongside image files (identified by presence of `shapes` key)

Reader behavior:
- requires `imagePath`, `imageWidth`, and `imageHeight` in each JSON file
- derives `Image.file_name` from `imagePath` basename (single-file mode) or from the relative JSON path stem + image extension (directory mode)
- stores original `imagePath` value in `Image.attributes["labelme_image_path"]`
- polygons are flattened to axis-aligned bounding box envelopes; original shape type stored as `Annotation.attributes["labelme_shape_type"] = "polygon"`
- requires unique derived image names across all JSON files in directory mode

Deterministic policy:
- reader image IDs: by derived file_name (lexicographic)
- reader category IDs: by label name (lexicographic)
- reader annotation IDs: by image order then shape order
- writer file order: by image file_name (lexicographic)

Writer behavior:
- single-image datasets to a `.json` path: writes one LabelMe JSON file
- multi-image datasets or directory paths: writes canonical `annotations/<stem>.json` + `images/README.txt` layout
- all annotations are written as `rectangle` shapes with 2 corner points (polygons are not restored)
- does **not** copy image binaries
- uses `labelme_image_path` image attribute for `imagePath` if present, otherwise `file_name`

Limitations:
- only `rectangle` and `polygon` shape types are supported (others are rejected)
- polygon geometry is flattened to axis-aligned bbox envelope (shape type retained as attribute only)
- `imageData` (embedded base64 image data) is not preserved
- LabelMe flags and group_id are not preserved

## SuperAnnotate JSON (`superannotate` / `superannotate-json` / `sa`)

- Path kind: JSON file or directory.
- Supported annotation schema: top-level `metadata` + `instances`.
- Supported geometries:
  - `bbox` / `bounding_box` / `rectangle` (direct bbox mapping)
  - `polygon`, `rotated_bbox`, `rotated_box`, `oriented_bbox`, `oriented_box` (flattened to axis-aligned bbox envelope)
- Unsupported geometries are rejected with a clear parse/layout error.

Reader input modes:
- **Single file**: one annotation JSON
- **Directory**: scans `annotations/` recursively when present, otherwise scans root recursively for matching annotation JSON files
- Optional class metadata: `classes/classes.json` and `classes.json` are read when present

Reader behavior:
- requires `metadata.width`, `metadata.height`, and `instances`
- image name comes from `metadata.name` (fallback: file stem)
- stores image name in `Image.attributes["superannotate_image_name"]`
- stores geometry provenance in `Annotation.attributes["superannotate_geometry_type"]`
- stores instance IDs when present in `Annotation.attributes["superannotate_instance_id"]`
- preserves confidence from `probability`/`confidence` when finite

Deterministic policy:
- reader image IDs: by derived `file_name` (lexicographic)
- reader category IDs: by label name (lexicographic)
- reader annotation IDs: by image order then instance order
- writer file order: by image `file_name` (lexicographic)

Writer behavior:
- single-image dataset + `.json` output path: writes one annotation JSON
- otherwise writes canonical directory layout:
  - `annotations/<image-stem>.json`
  - `classes/classes.json`
  - `images/README.txt`
- emits all annotations as `bbox` instances
- preserves IR confidence as `probability`
- does **not** copy image binaries

## Supervisely JSON (`supervisely` / `supervisely-json` / `sly`)

- Path kind: JSON file or directory.
- Supported annotation schema: top-level `size` + `objects`.
- Supported geometries:
  - `rectangle` (direct rectangle envelope)
  - `polygon` (flattened to axis-aligned bbox envelope)
- Unsupported geometries are rejected with a clear parse/layout error.

Reader input modes:
- **Single file**: one annotation JSON
- **Dataset directory**: `<root>/ann/*.json` (recursive)
- **Project directory**: `<root>/meta.json` plus one or more `<dataset>/ann/*.json` trees

Reader behavior:
- requires `size.width`, `size.height`, and `objects`
- derives `Image.file_name` from annotation path (`*.jpg.json` -> `*.jpg`)
- for project roots, prefixes image names with dataset folder (e.g. `dataset_01/img.jpg`)
- stores dataset name in `Image.attributes["supervisely_dataset"]` when available
- stores relative annotation path in `Image.attributes["supervisely_ann_path"]`
- stores geometry provenance in `Annotation.attributes["supervisely_geometry_type"]`
- stores object IDs in `Annotation.attributes["supervisely_object_id"]` when present
- reads optional object `confidence` / `score` when finite
- writer does not emit confidence/score fields, so IR confidence is not preserved on Supervisely write

Deterministic policy:
- reader image IDs: by derived `file_name` (lexicographic)
- reader category IDs: by class title/name (lexicographic)
- reader annotation IDs: by image order then object order
- writer file order: by image `file_name` (lexicographic)

Writer behavior:
- single-image dataset + `.json` output path: writes one annotation JSON
- otherwise writes canonical project layout:
  - `meta.json`
  - `dataset/ann/<image.file_name>.json`
  - `dataset/img/README.txt`
- emits all annotations as `rectangle` objects
- does **not** copy image binaries

## Cityscapes JSON (`cityscapes` / `cityscapes-json`)

- Path kind: JSON file or directory.
- Supported source schema: Cityscapes polygon JSON with `imgWidth`, `imgHeight`, and `objects`.
- Supported input layouts:
  - single `*_gtFine_polygons.json` file
  - directory containing matching polygon JSON files
  - `gtFine/` root
  - full dataset root containing `gtFine/<split>/<city>/*_gtFine_polygons.json`
- Polygon coordinates are converted to the smallest axis-aligned bbox envelope; coordinates are not clipped.
- Deleted objects and Cityscapes ignored/stuff labels are skipped.
- Group labels such as `cargroup` / `persongroup` are mapped to their base instance label and marked with `cityscapes_is_group=true`.
- Unknown labels are kept and marked with `cityscapes_label_status=unknown`.
- Annotation attributes include `cityscapes_original_label` and `cityscapes_bbox_source=polygon_envelope`.

Deterministic policy:
- reader image IDs: by derived `file_name` (lexicographic)
- reader category IDs: by label name (lexicographic)
- reader annotation IDs: by image order then object order
- writer file order: by image `file_name` (lexicographic)

Writer behavior:
- single-image dataset + `.json` output path: writes one polygon JSON file
- otherwise writes `gtFine/<split>/<city>/*_gtFine_polygons.json`; `cityscapes_split` / `cityscapes_city` image attributes are used when present, otherwise `train/panlabel` is used
- emits every bbox as a four-point rectangle polygon
- writes a placeholder `leftImg8bit/README.txt`
- does **not** copy image binaries

## Marmot XML (`marmot` / `marmot-xml`)

- Path kind: XML file or directory.
- Supported source schema: a root `<Page CropBox="...">` element with `<Composite>` children under `<Composites>` blocks.
- The reader intentionally ignores `<Leaf>` elements and any `<Composite>` not directly under `<Composites>`.
- `Page@CropBox` and `Composite@BBox` must each contain exactly four 16-hex-character big-endian f64 tokens.
- Rectangle token order is `x_left y_top x_right y_bottom` in Marmot/PDF-like page coordinates.
- The reader requires a same-stem companion image (`page.xml` + `page.bmp` / `page.png` / etc.) to get pixel dimensions; CropBox values alone are not treated as image dimensions.
- Coordinates are scaled through the CropBox and converted to IR pixel-space XYXY with a top-left origin, including the Y-axis flip.
- Category names come from `Composite@Label`, then parent `Composites@Label`, then `Composite`.

Deterministic policy:
- reader image IDs: by companion image `file_name` (lexicographic)
- reader category IDs: by label name (lexicographic)
- reader annotation IDs: by XML/object order
- writer file order: by image `file_name` (lexicographic)

Writer behavior:
- single-image dataset + `.xml` output path: writes one Marmot XML file
- directory output writes one `.xml` file per image path with the image extension replaced by `.xml`
- emits minimal `<Page>`, `<Composites>`, and `<Composite>` elements
- encodes CropBox/BBox values as big-endian f64 hex tokens
- does **not** copy image binaries

## CreateML JSON (`create-ml` / `createml` / `create-ml-json`)

- Path kind: JSON file.
- Apple's annotation format for Core ML training.
- Flat JSON array where each element represents one image with its annotations.
- Bbox format: center-based absolute pixel coordinates `{x, y, width, height}` where `(x, y)` is the center of the box.
- Image dimensions are **not** stored in the JSON — the reader resolves them from local image files relative to the JSON file's parent directory.

Reader behavior:
- parses top-level JSON array of `{image, annotations}` objects
- `image` must be a non-empty relative path (absolute paths and `..` traversal are rejected)
- resolves image dimensions from disk by probing `<base_dir>/<image>` then `<base_dir>/images/<image>`
- rejects duplicate `image` entries
- rejects empty annotation labels

Deterministic policy:
- reader image IDs: by image filename (lexicographic)
- reader category IDs: by label name (lexicographic)
- reader annotation IDs: by image order then annotation order

Writer behavior:
- writes a single JSON array with one object per image
- uses center-based absolute pixel coordinates: `{x, y, width, height}`
- deterministic output: image rows sorted by filename, annotations sorted by annotation ID
- images without annotations are included (empty `annotations` array)
- does **not** write image dimensions (this is by design — CreateML resolves them at training time)

Limitations:
- no dataset-level metadata/licenses
- no image-level metadata (dimensions, license, date)
- no annotation confidence/attributes
- requires image files on disk for reading (to resolve dimensions)

## KITTI (`kitti` / `kitti-txt`)

- Path kind: directory.
- Accepted input path:
  - dataset root containing `label_2/` and `image_2/`
  - or `label_2/` directory directly (with sibling `../image_2/`)
- Standard format in autonomous driving research.
- Per-image `.txt` files with 15 space-separated fields per line (optional 16th field: score).
- Fields: `type truncated occluded alpha xmin ymin xmax ymax dim_height dim_width dim_length loc_x loc_y loc_z rotation_y [score]`
- Bbox: fields 4–7 (`xmin ymin xmax ymax`) are absolute pixel coordinates.

Reader behavior:
- scans `label_2/` flat (non-recursive, top-level `.txt` files only)
- resolves images from `image_2/` with extension precedence: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`
- maps `type` → category name, fields 4–7 → `BBoxXYXY<Pixel>`, optional field 15 → `Annotation.confidence`
- stores remaining numeric fields as annotation attributes with `kitti_*` prefix: `kitti_truncated`, `kitti_occluded`, `kitti_alpha`, `kitti_dim_height`, `kitti_dim_width`, `kitti_dim_length`, `kitti_loc_x`, `kitti_loc_y`, `kitti_loc_z`, `kitti_rotation_y`

Deterministic policy:
- reader image IDs: by resolved image filename (lexicographic)
- reader category IDs: by class/type name (lexicographic)
- reader annotation IDs: by label file order then line number

Writer behavior:
- creates `label_2/` + `image_2/README.txt`
- one `.txt` per image, empty files for unannotated images
- sorts images by `file_name`, annotations within each image by ID
- sources KITTI-specific fields from `kitti_*` annotation attributes; uses defaults for missing values: truncated=0, occluded=0, alpha=−10, dims=−1, loc=−1000, rotation_y=−10
- rejects `Image.file_name` with path separators (KITTI layout is flat)
- does **not** copy image binaries

Limitations:
- no dataset-level metadata/licenses
- no image-level metadata (license, date)
- no annotation attributes outside the `kitti_*` set
- confidence is preserved via the optional `score` field

## VGG Image Annotator JSON (`via` / `via-json` / `vgg-via`)

- Path kind: JSON file.
- Popular academic annotation tool.
- Single JSON file with object-root keyed by arbitrary strings (typically `filename+size`).
- Each entry: `{ filename, size, regions, file_attributes }`.
- Supported region type: `rect` only (`shape_attributes.name == "rect"` with `x`, `y`, `width`, `height`).
- Image dimensions are **not** stored in the JSON — resolved from local image files.

Reader behavior:
- supports `regions` as either an array or an object map (both forms exist in real VIA exports)
- label resolution precedence from `region_attributes`: `label`, then `class`, then sole scalar attribute
- non-rect shapes are skipped with a warning
- image dimension resolution: `<json_dir>/<filename>` then `<json_dir>/images/<filename>`
- rejects duplicate filenames across entries
- stores `via_size_bytes` as image attribute; scalar `file_attributes` as `via_file_attr_<key>` image attributes
- stores scalar `region_attributes` (excluding the label key) as `via_region_attr_<key>` annotation attributes

Deterministic policy:
- reader image IDs: by filename (lexicographic)
- reader category IDs: by resolved label (lexicographic)
- reader annotation IDs: by image order then region order (for object-form regions, keys sorted lexicographically)

Writer behavior:
- writes JSON object keyed by `<filename><size>`
- `regions` always emitted as array, sorted by annotation ID
- uses canonical `label` key in `region_attributes` for category name
- reconstructs `file_attributes` from `via_file_attr_*` image attributes
- unannotated images preserved with `regions: []`
- does **not** copy image binaries

Limitations:
- only rectangle regions are supported
- no dataset-level metadata/licenses
- no annotation confidence
- requires image files on disk for reading (to resolve dimensions)

## RetinaNet Keras CSV (`retinanet` / `retinanet-csv` / `keras-retinanet`)

- Path kind: CSV file.
- Simple format used with keras-retinanet: `path,x1,y1,x2,y2,class_name`.
- Coordinates are absolute pixels (unlike TFOD which uses normalized coordinates).
- No header required (optional header row is tolerated).
- Unannotated images: `path,,,,,` (all-empty row).
- Image dimensions are **not** in the CSV — resolved from local image files.

Reader behavior:
- tolerates optional header row exactly matching `path,x1,y1,x2,y2,class_name`
- supports empty rows (`path,,,,,`) for unannotated images
- rejects partial rows (some bbox fields present, others empty)
- resolves image paths relative to CSV parent directory; absolute paths used as-is
- caches dimension lookups per image path

Deterministic policy:
- reader image IDs: by path (lexicographic)
- reader category IDs: by class name (lexicographic)
- reader annotation IDs: by CSV row order

Writer behavior:
- headerless CSV (matches keras-retinanet conventions)
- rows grouped by image, images sorted by `file_name`, annotations by ID
- unannotated images emit exactly one `path,,,,,` row
- does **not** copy image binaries

Limitations:
- no dataset-level metadata/licenses
- no image-level metadata (dimensions, license, date)
- no annotation confidence/attributes
- requires image files on disk for reading (to resolve dimensions)

## OpenImages CSV (`openimages` / `openimages-csv` / `open-images`)

- Path kind: CSV file.
- Column layout: `ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax` (8 columns) or extended 13-column form with trailing boolean flags.
- Note: column order is `XMin,XMax,YMin,YMax` (not `XMin,YMin,XMax,YMax`).
- Coordinates are **normalized** (0–1); reader resolves pixel dimensions from local image files.
- Confidence is preserved through roundtrip.
- Reader stores `openimages_source` as an annotation attribute and `openimages_image_id` as an image attribute.

Reader behavior:
- accepts 8-column or 13-column rows
- optional header is detected and skipped (case-insensitive)
- resolves image dimensions from `base_dir/<ImageID>` or `base_dir/images/<ImageID>`, probing common extensions if ImageID has none

Deterministic policy:
- image IDs: by ImageID (lexicographic)
- category IDs: by LabelName (lexicographic)
- annotation IDs: by CSV row order

Writer behavior:
- emits 8-column CSV with header
- rows ordered by annotation ID
- derives ImageID from `openimages_image_id` image attribute or file stem
- default `Source` is `xclick`; default `Confidence` is `1.0`

Limitations:
- requires image files on disk for reading
- no dataset-level metadata/licenses
- images without annotations are not emitted

## Kaggle Wheat CSV (`kaggle-wheat` / `kaggle-wheat-csv`)

- Path kind: CSV file.
- Column layout: `image_id,width,height,bbox,source` (5 columns).
- `bbox` is a bracketed string `[x, y, width, height]` in absolute pixel coordinates.
- **Single-class format**: no label column; all annotations are implicitly `wheat_head`.
- Converting a multi-class dataset to this format will collapse all categories.

Reader behavior:
- parses bbox string with whitespace tolerance
- validates dimension consistency per image_id
- stores `source` as `kaggle_wheat_source` image attribute

Deterministic policy:
- image IDs: by image_id (lexicographic)
- single category: `wheat_head` (ID 1)
- annotation IDs: by CSV row order

Writer behavior:
- emits headered CSV
- rows ordered by annotation ID
- bbox canonical form: `[x, y, width, height]` with `, ` separators

Limitations:
- single-class only
- no confidence/attributes
- no dataset-level metadata/licenses
- images without annotations are not emitted

## Google Cloud AutoML Vision CSV (`automl-vision` / `automl-vision-csv` / `google-cloud-automl`)

- Path kind: CSV file.
- Sparse row layout: `set,path,label,xmin,ymin,,,xmax,ymax,,` (9 or 11 columns).
- Coordinates are **normalized** (0–1); reader resolves pixel dimensions from local image files.
- First column (`set`) indicates ML split: `TRAIN`, `VALIDATION`, `TEST`, or `UNASSIGNED`.

Reader behavior:
- accepts 9-column or 11-column rows
- optional header detected and skipped
- coordinates at fixed positions: xmin=3, ymin=4, xmax=7, ymax=8
- GCS URIs (`gs://bucket/path`) resolved by path suffix then basename
- stores `automl_ml_use` and `automl_image_uri` as image attributes

Deterministic policy:
- image IDs: by URI (lexicographic)
- category IDs: by label (lexicographic)
- annotation IDs: by CSV row order

Writer behavior:
- headerless 11-column sparse rows
- rows ordered by annotation ID
- default ML_USE is `UNASSIGNED`

Limitations:
- requires image files on disk for reading
- no confidence/attributes
- no dataset-level metadata/licenses
- images without annotations are not emitted

## Udacity Self-Driving Car CSV (`udacity` / `udacity-csv` / `self-driving-car`)

- Path kind: CSV file.
- Column layout: `filename,width,height,class,xmin,ymin,xmax,ymax` (8 columns).
- Same header as TFOD CSV but coordinates are **absolute pixels** (not normalized).
- Auto-detection heuristic: if any coordinate exceeds 1.0, detected as Udacity; otherwise TFOD.

Reader behavior:
- serde-based with header
- validates dimension consistency per filename
- absolute pixel coordinates map directly to IR (no normalization)

Deterministic policy:
- image IDs: by filename (lexicographic)
- category IDs: by class name (lexicographic)
- annotation IDs: by CSV row order

Writer behavior:
- emits headered CSV with absolute pixel coordinates
- rows ordered by annotation ID

Limitations:
- no dataset-level metadata/licenses
- no confidence/attributes
- images without annotations are not emitted
- TFOD/Udacity auto-detection uses coordinate range heuristic

## Future expansion rule

When formats become numerous, split this page into per-format files under `docs/formats/<format>.md` and keep this page as an index.
