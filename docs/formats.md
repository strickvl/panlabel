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
| `cvat` | file (`.xml`) or directory (`annotations.xml`) | yes | yes | lossy |
| `label-studio` | file (`.json`) | yes | yes | lossy |
| `tfod` | file (`.csv`) | yes | yes | lossy |
| `yolo` | directory (`images/` + `labels/`) | yes | yes | lossy |
| `voc` | directory (`Annotations/` + `JPEGImages/`) | yes | yes | lossy |
| `hf` | directory (`metadata.jsonl` / `metadata.parquet`) | yes | yes (`metadata.jsonl`) | lossy |

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

## YOLO directory (`yolo` / `ultralytics` / `yolov8` / `yolov5`)

- Path kind: directory.
- Accepted input path:
  - dataset root containing `images/` and `labels/`
  - or `labels/` directory directly (with sibling `../images/`)
- Label row format (one line per bbox):
  - `<class_id> <x_center> <y_center> <width> <height>`
  - normalized values

Reader behavior:
- class map precedence: `data.yaml` → `classes.txt` → inferred from labels
- image resolution is read from image headers in `images/`
- each label file must map to a matching image file (same relative stem) under `images/`
- expected image extensions (lookup order): `jpg`, `png`, `jpeg`, `bmp`, `webp`
- lines with more than 5 tokens are rejected (segmentation/pose not supported)

Writer behavior:
- creates output `images/` and `labels/` directories
- writes `data.yaml` class map
- creates empty `.txt` files for images without annotations
- does **not** copy image binaries
- writes normalized floats with 6 decimal places

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
  - or split subdirectories (for example `train/`, `validation/`) each containing metadata
- Remote Hub import is supported in `convert` via `--hf-repo` (requires `hf-remote` feature).

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
- when only `metadata.parquet` exists, reading requires `hf-parquet`

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
- emits minimal `<meta><task>` with `name='panlabel export'`
- writes labels only for categories referenced by annotations
- writes `<image>` entries for all images, including unannotated images
- writes `<box>` entries sorted by annotation ID per image
- normalizes `occluded` values:
  - `true`/`yes`/`1` -> `1`
  - `false`/`no`/`0` -> `0`
  - otherwise or missing -> `0`

## Future expansion rule

When formats become numerous, split this page into per-format files under `docs/formats/<format>.md` and keep this page as an index.
