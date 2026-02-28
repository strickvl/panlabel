# Conversion, lossiness, and reports

Not all annotation formats carry the same information. When you convert between
formats, some fields may not have an equivalent in the target format — that's
what panlabel calls "lossiness." Rather than silently dropping data, panlabel
tells you exactly what would be lost and asks you to opt in with `--allow-lossy`.

Every `convert` command generates a report explaining what happened.

## Lossiness model

A conversion report issue has a severity:

- `warning`: real information loss risk; conversion is blocked unless `--allow-lossy` is set
- `info`: deterministic policy note; never blocks conversion

Format-level lossiness relative to IR:
- `ir-json`: lossless
- `coco`: conditional
- `tfod`: lossy
- `yolo`: lossy

## JSON report shape

With `--report json`, output follows this shape:

```json
{
  "from": "coco",
  "to": "yolo",
  "input": {"images": 10, "categories": 3, "annotations": 40},
  "output": {"images": 10, "categories": 3, "annotations": 40},
  "issues": [
    {"severity": "warning", "code": "drop_dataset_info", "message": "..."},
    {"severity": "info", "code": "yolo_writer_float_precision", "message": "..."}
  ]
}
```

## Stable issue codes

These codes are designed to be stable for programmatic use.

### Warning codes

| Code | Meaning |
|---|---|
| `drop_dataset_info` | Dataset-level metadata is dropped |
| `drop_licenses` | License list is dropped |
| `drop_image_metadata` | Image metadata fields (license/date) are dropped |
| `drop_category_supercategory` | Category supercategory is dropped |
| `drop_annotation_confidence` | Annotation confidence values are dropped |
| `drop_annotation_attributes` | Annotation attributes are dropped |
| `drop_images_without_annotations` | Images without annotations will not appear (TFOD behavior) |
| `drop_dataset_info_name` | `info.name` has no COCO equivalent |
| `coco_attributes_may_not_be_preserved` | Some COCO-tool roundtrips may not preserve nonstandard attributes |

### Info codes

| Code | Meaning |
|---|---|
| `tfod_reader_id_assignment` | TFOD reader deterministic ID policy |
| `tfod_writer_row_order` | TFOD writer deterministic row order |
| `yolo_reader_id_assignment` | YOLO reader deterministic ID policy |
| `yolo_reader_class_map_source` | YOLO class map precedence/source note |
| `yolo_writer_class_order` | YOLO writer class index assignment policy |
| `yolo_writer_empty_label_files` | YOLO writer creates empty label files for unannotated images |
| `yolo_writer_float_precision` | YOLO normalized float precision policy |

## Practical guidance

- If conversion is blocked, run once with `--report json` to see exactly which warning codes triggered the block.
- Prefer explicit `--allow-lossy` only when you accept those specific losses.
