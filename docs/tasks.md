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
| `cvat` | yes | yes | CVAT "for images" XML; `<box>` annotations only; absolute pixel coordinates |
| `label-studio` | yes | yes | task-export JSON (`rectanglelabels`), percentage coordinates, lossy (rotations flattened to axis-aligned bbox envelopes) |
| `tfod` | yes | yes | normalized CSV format; lossy |
| `yolo` | yes | yes | directory-based; normalized center-format rows |
| `voc` | yes | yes | directory-based Pascal VOC XML; pixel-space XYXY |
| `hf` | yes | yes (`metadata.jsonl`) | HF ImageFolder metadata (`metadata.jsonl` / `metadata.parquet`), bbox mode via `--hf-bbox-format`; remote Hub import currently in `convert` |

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

## Detection boundaries by format

Each adapter enforces detection-only boundaries differently. Here's what each
format accepts and rejects:

| Format | Accepts | Rejects / ignores |
|---|---|---|
| `coco` | bbox annotations (`annotations[].bbox`) | `segmentation` is accepted on read but ignored (not converted to IR); on write, emitted as `[]` |
| `cvat` | `<box>` annotation elements only | `<polygon>`, `<points>`, `<polyline>`, and other annotation elements are hard parse errors |
| `label-studio` | `rectanglelabels` results only | Other result types are rejected; `rotation` is flattened to an axis-aligned envelope (angle kept as `ls_rotation_deg` attribute) |
| `yolo` | 5-token bbox rows (`class cx cy w h`) and 6-token rows (`class cx cy w h confidence`) | Rows with 7+ tokens (segmentation, pose, OBB) are rejected with a clear error |
| `voc` | `<object>` elements with `<bndbox>` | All `<object>` entries are read; no non-bbox geometry exists in VOC |
| `tfod` | Rows with `filename,width,height,class,xmin,ymin,xmax,ymax` | Fixed schema; no non-bbox geometry |
| `hf` | Bbox arrays in the objects container (`objects.bbox`) | Fixed bbox schema; bbox interpretation depends on `--hf-bbox-format` |

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
