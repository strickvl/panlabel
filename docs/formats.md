# Formats

Panlabel converts through a canonical IR:
- bounding boxes are represented as **pixel-space XYXY** in IR
- format adapters map each external format into/out of that IR

Current scope: **object detection** only.

## Format matrix

| Format | Path kind | Read | Write | Lossiness vs IR |
|---|---|---|---|---|
| `ir-json` | file (`.json`) | yes | yes | lossless |
| `coco` | file (`.json`) | yes | yes | conditional |
| `tfod` | file (`.csv`) | yes | yes | lossy |
| `yolo` | directory (`images/` + `labels/`) | yes | yes | lossy |

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

## Future expansion rule

When formats become numerous, split this page into per-format files under `docs/formats/<format>.md` and keep this page as an index.
