# Tasks and use cases

Panlabel doesn't try to support everything at once. This page shows what's
supported today, what's not yet implemented, and what to expect when working
within those boundaries.

## Current support snapshot

| Task / use case | Status | Notes |
|---|---|---|
| Object detection (axis-aligned bbox) | ✅ supported | Canonical IR task today |
| Instance segmentation | ❌ not supported | Polygon/mask structures are not represented in IR |
| Classification-only labels | ❌ not supported | No classification-only schema/adapter yet |
| Keypoints / pose | ❌ not supported | Keypoint fields are not modeled in IR |
| Oriented bounding boxes (OBB) | ❌ not supported | Rotated-box schema not implemented |
| Tracking / video IDs | ❌ not supported | Track identity schema not implemented |

## Detection task (supported)

### Canonical representation in panlabel

- Task: object detection
- Geometry: axis-aligned bbox
- Internal bbox representation: **pixel-space XYXY**

### Format support for detection

| Format | Read | Write | Important behavior |
|---|---|---|---|
| `ir-json` | yes | yes | canonical/lossless representation |
| `coco` | yes | yes | bbox `[x,y,w,h]` mapped to/from IR XYXY |
| `ibm-cloud-annotations` | yes | yes | IBM Cloud Annotations localization JSON; normalized `x,y,x2,y2`; file or directory based |
| `cvat` | yes | yes | CVAT "for images" XML; `<box>` annotations only; absolute pixel coordinates |
| `label-studio` | yes | yes | task-export JSON (`rectanglelabels`), percentage coordinates, lossy (rotations flattened to axis-aligned bbox envelopes) |
| `labelbox` | yes | yes | current export rows (`.json`, `.jsonl`, `.ndjson`); boxes direct, polygons flattened to bbox envelopes, unsupported objects skipped with warnings |
| `scale-ai` | yes | yes | Scale AI image annotation task/response JSON; boxes direct, polygons and rotated boxes with vertices flattened to bbox envelopes, unsupported geometry rejected clearly |
| `unity-perception` | yes | yes | Unity Perception/SOLO frame and captures JSON; `BoundingBox2D` values direct, non-bbox annotations skipped with warnings |
| `tfod` | yes | yes | normalized CSV format; lossy |
| `vott-csv` | yes | yes | Microsoft VoTT headered CSV; absolute pixel XYXY coordinates; file based |
| `vott-json` | yes | yes | Microsoft VoTT aggregate/per-asset JSON; rectangles direct, polygon-like point regions flattened to bbox envelopes; file or directory based |
| `yolo` | yes | yes | directory/list-split based; normalized center-format rows |
| `yolo-keras` / `yolov4-pytorch` | yes | yes | shared single-file TXT grammar; absolute pixel XYXY boxes with zero-based class IDs |
| `voc` | yes | yes | directory-based Pascal VOC XML; pixel-space XYXY |
| `hf` | yes | yes (`metadata.jsonl`) | HF ImageFolder metadata (`metadata.jsonl` / `metadata.parquet`), bbox mode via `--hf-bbox-format`; remote Hub import currently in `convert` |
| `sagemaker` | yes | yes | AWS Ground Truth manifest JSONL (`.manifest` / `.jsonl`); dynamic label attribute + `<label>-metadata`; object-detection rows only |
| `labelme` | yes | yes | per-image JSON; `rectangle` and `polygon` shapes (polygons flattened to bbox envelopes); file or directory based |
| `superannotate` | yes | yes | per-image JSON (`metadata` + `instances`), file or directory based; polygon/rotated geometries flattened to bbox envelopes |
| `supervisely` | yes | yes | per-image JSON (`size` + `objects`), dataset `ann/` or project (`meta.json` + dataset `ann/`); polygons flattened to bbox envelopes |
| `cityscapes` | yes | yes | Cityscapes polygon JSON (`imgWidth` / `imgHeight` / `objects`), file or `gtFine/` dataset root; polygons flattened to bbox envelopes |
| `marmot` | yes | yes | Marmot XML document-layout pages; `<Composite>` BBox hex doubles under `<Composites>` converted to pixel-space XYXY using companion image dimensions |
| `create-ml` | yes | yes | Apple CreateML JSON array; center-based absolute pixel coordinates; file based |
| `kitti` | yes | yes | directory-based; per-image `.txt` files with 15-field KITTI rows; absolute pixel coordinates |
| `via` | yes | yes | VGG Image Annotator single-file JSON; rectangle regions; absolute pixel coordinates |
| `retinanet` | yes | yes | keras-retinanet CSV; absolute pixel XYXY coordinates; file based |
| `openimages` | yes | yes | Google OpenImages CSV; normalized XYXY coordinates plus confidence/source metadata |
| `kaggle-wheat` | yes | yes | Kaggle Global Wheat Detection CSV; single-class bbox strings (`[xmin, ymin, width, height]`) |
| `automl-vision` | yes | yes | Google Cloud AutoML Vision CSV; sparse GCS/local path rows with normalized bbox corners |
| `udacity` | yes | yes | Udacity Self-Driving Car CSV; TFOD-like header with absolute pixel coordinates |

For per-format details, see [formats.md](./formats.md).

## Why panlabel rejects unsupported data instead of silently dropping it

You might wonder why panlabel errors out on YOLO rows with 7+ tokens or ignores
COCO segmentation payloads. The principle is simple: **no silent surprises**.
If panlabel can't faithfully represent something, it tells you rather than
quietly producing incomplete output.

Examples:
- YOLO rows with 7+ tokens (segmentation, pose, or OBB data) are rejected with a clear error. 6-token rows are accepted as detection + confidence.
- COCO segmentation payloads are accepted during read but not converted into IR (bbox-only).
- Label Studio result types other than `rectanglelabels` are rejected in the current detection-only adapter.
- Label Studio `rotation` does not add OBB support: geometry is flattened to axis-aligned envelopes (angle retained as metadata).
- Labelbox polygons are flattened to bbox envelopes; points, masks, lines, and other non-detection object kinds are skipped with warnings while preserving the image row.
- Scale AI polygons and rotated boxes with vertices are flattened to bbox envelopes; lines, points, cuboids, ellipses, and other unsupported geometry are rejected clearly.
- Unity Perception imports `BoundingBox2D` values and skips segmentation/keypoint/other non-bbox annotation blocks with warnings while preserving captures/images.

## Detection boundaries by format

Each adapter enforces detection-only boundaries differently. Here's what each
format accepts and rejects:

| Format | Accepts | Rejects / ignores |
|---|---|---|
| `coco` | bbox annotations (`annotations[].bbox`) | `segmentation` is accepted on read but ignored (not converted to IR); on write, emitted as `[]` |
| `cvat` | `<box>` annotation elements only | `<polygon>`, `<points>`, `<polyline>`, and other annotation elements are hard parse errors |
| `label-studio` | `rectanglelabels` results only | Other result types are rejected; `rotation` is flattened to an axis-aligned envelope (angle kept as `ls_rotation_deg` attribute) |
| `labelbox` | `bounding_box` / `bbox` objects, plus `polygon` objects flattened to bbox envelopes | Points, masks, lines, and classification-style objects are skipped with warnings; image rows remain in the dataset |
| `scale-ai` | `type: "box"` objects, plus `polygon`/rotated-box `vertices` flattened to bbox envelopes | Unsupported geometry types are rejected so users see exactly which shape cannot enter the bbox-only IR |
| `unity-perception` | SOLO `BoundingBox2DAnnotation` / `BoundingBox2D` values with `x,y,width,height` or `origin` + `dimension` | Non-bbox annotation blocks are skipped with warnings; writer emits bbox-only directory output and rejects ambiguous `.json` file output |
| `yolo` | 5-token bbox rows (`class cx cy w h`) and 6-token rows (`class cx cy w h confidence`) | Rows with 7+ tokens (segmentation, pose, OBB) are rejected with a clear error |
| `yolo-keras` / `yolov4-pytorch` | Rows like `image xmin,ymin,xmax,ymax,class_id ...`; image-only rows for unannotated images | Malformed box tokens and non-XYXY boxes are rejected with file/line context |
| `voc` | `<object>` elements with `<bndbox>` | All `<object>` entries are read; no non-bbox geometry exists in VOC |
| `tfod` | Rows with `filename,width,height,class,xmin,ymin,xmax,ymax` | Fixed schema; no non-bbox geometry |
| `vott-csv` | Headered rows with `image,xmin,ymin,xmax,ymax,label` | Fixed schema; no non-bbox geometry |
| `vott-json` | `RECTANGLE` regions with `boundingBox`, plus point-based polygon-like regions flattened to bbox envelopes | Unsupported tagged regions with no `boundingBox` or `points` are rejected |
| `ibm-cloud-annotations` | Localization JSON objects with normalized `x,y,x2,y2,label` | Fixed localization schema; no non-bbox geometry |
| `hf` | Bbox arrays in the objects container (`objects.bbox`) | Fixed bbox schema; bbox interpretation depends on `--hf-bbox-format` |
| `sagemaker` | Object-detection label block with `annotations` + `image_size`, plus `<label>-metadata` (`groundtruth/object-detection`) | Segmentation/classification Ground Truth task types are rejected; mixed/ambiguous label attributes are rejected |
| `labelme` | `rectangle` shapes (2 points) and `polygon` shapes (3+ points, flattened to bbox envelope) | Other shape types (e.g. `circle`, `line`) are rejected with a clear error |
| `superannotate` | `bbox`/`rectangle` plus polygon/rotated/oriented boxes (flattened to bbox envelopes) | Unsupported geometry types are rejected with a clear error |
| `supervisely` | `rectangle` and `polygon` object geometries (`geometry.points.exterior`) | Unsupported `geometryType` values (e.g. bitmap/point/line) are rejected |
| `cityscapes` | `objects[].polygon` arrays (flattened to bbox envelopes) | Deleted objects plus ignored/stuff labels are skipped; unknown kept labels are marked with attributes |
| `marmot` | `<Composite BBox="...">` elements directly under `<Composites>` | `<Leaf>` elements and composites outside `<Composites>` are ignored; companion image dimensions are required |
| `create-ml` | `coordinates` objects with center-based pixel bboxes (`x`, `y`, `width`, `height`) | Fixed bbox schema; no non-bbox geometry |
| `kitti` | 15/16-field space-separated rows (type + bbox + 3D fields + optional score) | Fixed 15/16-field schema; no non-bbox geometry |
| `via` | `rect` regions with `shape_attributes` (`x`, `y`, `width`, `height`) | Non-rect shape types (circle, polygon, etc.) are skipped with a warning |
| `retinanet` | 6-column CSV rows (`path,x1,y1,x2,y2,class_name`) plus empty rows for unannotated images | Fixed 6-column schema; no non-bbox geometry |
| `openimages` | OpenImages CSV rows with normalized `XMin/XMax/YMin/YMax`, label, confidence, and source columns | Fixed bbox schema; confidence is preserved, no non-bbox geometry |
| `kaggle-wheat` | CSV rows with `image_id,width,height,bbox`, where `bbox` is `[xmin, ymin, width, height]` | Single-class format; multiple IR categories collapse to `wheat` on write |
| `automl-vision` | Sparse AutoML CSV rows with ML-use split, image URI, label, and normalized bbox corner columns | Fixed bbox schema; no non-bbox geometry |
| `udacity` | CSV rows with `frame,xmin,ymin,xmax,ymax,label` | Fixed bbox schema; no non-bbox geometry |

## Adding a new task in the future

When implementing a new task type, update these places together:

1. IR schema in `src/ir/` (task data model)
2. Relevant adapter behavior in `src/ir/io_*.rs`
3. Validation and/or conversion lossiness rules as needed
4. `docs/tasks.md` (this page)
5. `docs/formats.md` (task notes per format)
6. Tests that assert user-visible behavior

## Planned docs expansion

As the task surface grows, split this file into:
- `docs/tasks/detection.md`
- `docs/tasks/segmentation.md`
- `docs/tasks/classification.md`
- etc.

Also add `docs/providers/` when provider-specific execution/training/export docs become real implementation concerns.

## Planning link

For future priorities and sequencing, see [ROADMAP.md](../ROADMAP.md).
