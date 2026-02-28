# Panlabel documentation

This is the main documentation hub for panlabel.

## Supported scope (implemented today)

Panlabel currently supports **object detection bounding boxes** only.

Implemented formats:
- `ir-json` (Panlabel IR JSON)
- `coco` / `coco-json` (COCO JSON)
- `tfod` / `tfod-csv` (TensorFlow Object Detection CSV)
- `yolo` / `ultralytics` / `yolov8` / `yolov5` (Ultralytics-style YOLO directory)

Not implemented yet:
- segmentation
- keypoints / pose
- oriented bounding boxes (OBB)
- classification-only label formats

## Start here

- [CLI reference](./cli.md)
- [Format reference](./formats.md)
- [Conversion and lossiness](./conversion.md)
- [Contributing and docs rules](./contributing.md)

## Source of truth map (for humans and LLMs)

| Topic | Primary source |
|---|---|
| CLI commands/flags/auto-detect | `src/lib.rs` |
| COCO format behavior | `src/ir/io_coco_json.rs` |
| TFOD format behavior | `src/ir/io_tfod_csv.rs` |
| YOLO format behavior | `src/ir/io_yolo.rs` |
| Lossiness logic | `src/conversion/mod.rs` |
| Stable conversion issue codes | `src/conversion/report.rs` |
| User-visible CLI behavior tests | `tests/cli.rs` |
| YOLO roundtrip behavior tests | `tests/yolo_roundtrip.rs` |

## Growth model

This docs layout is intentionally small at first. As the project grows, split into:
- `docs/formats/<format>.md`
- `docs/providers/<provider>.md`
- `docs/tasks/<task>.md` (for detection vs segmentation vs classification, etc.)

Until then, keep core facts centralized in the four docs pages linked above.
