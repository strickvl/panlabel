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
| `label-studio` | yes | yes | task-export JSON (`rectanglelabels`), percentage coordinates, lossy (rotations flattened to axis-aligned bbox envelopes) |
| `tfod` | yes | yes | normalized CSV format; lossy |
| `yolo` | yes | yes | directory-based; normalized center-format rows |
| `voc` | yes | yes | directory-based Pascal VOC XML; pixel-space XYXY |

For per-format details, see [formats.md](./formats.md).

## Why panlabel rejects unsupported data instead of silently dropping it

You might wonder why panlabel errors out on YOLO rows with 6+ tokens or ignores
COCO segmentation payloads. The principle is simple: **no silent surprises**.
If panlabel can't faithfully represent something, it tells you rather than
quietly producing incomplete output.

Examples:
- YOLO rows with more than 5 tokens (often segmentation or pose data) are rejected with a clear error.
- COCO segmentation payloads are accepted during read but not converted into IR — and the conversion report notes this.
- Label Studio result types other than `rectanglelabels` are rejected in the current detection-only adapter.
- Label Studio `rotation` does not add OBB support: geometry is flattened to axis-aligned envelopes (angle retained as metadata).

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
