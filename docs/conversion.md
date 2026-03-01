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
- `cvat`: lossy
- `label-studio`: lossy
- `tfod`: lossy
- `yolo`: lossy
- `voc`: lossy
- `hf`: lossy

The format-level class is a general capability signal. Conversions are actually blocked only when the report contains one or more `warning` issues.

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
| `label_studio_rotation_dropped` | Rotated Label Studio boxes are flattened to axis-aligned envelopes; angle is kept as `ls_rotation_deg` attribute |
| `hf_metadata_lost` | HF metadata cannot represent full IR dataset metadata/licenses/supercategory fields |
| `hf_attributes_lost` | HF metadata drops image/annotation attributes outside its flat schema |
| `hf_confidence_lost` | HF metadata does not preserve annotation confidence |

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
| `voc_reader_id_assignment` | VOC reader deterministic ID assignment policy |
| `voc_reader_attribute_mapping` | VOC reader mapping of pose/truncated/difficult/occluded attributes |
| `voc_reader_coordinate_policy` | VOC reader coordinate policy (no 0/1-based adjustment) |
| `voc_reader_depth_handling` | VOC reader depth metadata handling note |
| `voc_writer_file_layout` | VOC writer XML path/layout policy |
| `voc_writer_no_image_copy` | VOC writer placeholder JPEGImages policy |
| `voc_writer_bool_normalization` | VOC writer boolean normalization policy |
| `label_studio_reader_id_assignment` | Label Studio reader deterministic ID assignment policy |
| `label_studio_reader_image_ref_policy` | Label Studio reader image reference mapping policy |
| `label_studio_writer_from_to_defaults` | Label Studio writer default `from_name` / `to_name` policy |
| `cvat_reader_id_assignment` | CVAT reader deterministic ID assignment policy |
| `cvat_reader_attribute_policy` | CVAT reader coordinate + attribute mapping policy |
| `cvat_writer_meta_defaults` | CVAT writer minimal `<meta>` block policy |
| `hf_reader_category_resolution` | HF reader category-name resolution precedence policy |
| `hf_writer_deterministic_order` | HF writer deterministic metadata/annotation ordering policy |

## Practical guidance

- If conversion is blocked, run once with `--report json` to see exactly which warning codes triggered the block.
- Prefer explicit `--allow-lossy` only when you accept those specific losses.
