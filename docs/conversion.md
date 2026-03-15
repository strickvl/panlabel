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
- `labelme`: lossy
- `create-ml`: lossy
- `kitti`: lossy
- `via`: lossy
- `retinanet`: lossy

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
    {"severity": "warning", "stage": "analysis", "code": "drop_dataset_info", "message": "..."},
    {"severity": "info", "stage": "source_reader", "code": "coco_reader_attribute_mapping", "message": "..."},
    {"severity": "info", "stage": "target_writer", "code": "yolo_writer_float_precision", "message": "..."}
  ]
}
```

The `stage` field indicates where in the conversion pipeline the issue originates:
- `analysis`: lossiness analysis (warnings about data loss)
- `source_reader`: source format reader policy
- `target_writer`: target format writer policy

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
| `cvat_writer_drop_unused_categories` | CVAT writer drops categories not referenced by any annotation from `<meta><labels>` |

### Info codes

| Code | Meaning |
|---|---|
| `coco_reader_attribute_mapping` | COCO reader maps score→confidence and stores area/iscrowd as annotation attributes |
| `coco_writer_deterministic_order` | COCO writer sorts licenses/images/categories/annotations by ID |
| `coco_writer_score_mapping` | COCO writer maps IR confidence to the COCO score field |
| `coco_writer_area_iscrowd_mapping` | COCO writer reads area/iscrowd from attributes; defaults to bbox area and iscrowd=0 |
| `coco_writer_empty_segmentation` | COCO writer emits empty segmentation arrays for detection-only output |
| `tfod_reader_id_assignment` | TFOD reader deterministic ID policy |
| `tfod_writer_row_order` | TFOD writer deterministic row order |
| `yolo_reader_id_assignment` | YOLO reader deterministic ID policy |
| `yolo_reader_class_map_source` | YOLO class map precedence/source note |
| `yolo_writer_class_order` | YOLO writer class index assignment policy |
| `yolo_writer_empty_label_files` | YOLO writer creates empty label files for unannotated images |
| `yolo_writer_float_precision` | YOLO normalized float and confidence precision policy (6 decimal places) |
| `yolo_writer_deterministic_order` | YOLO writer orders images and labels by file_name |
| `yolo_writer_no_image_copy` | YOLO writer creates only label files; images are not copied |
| `yolo_reader_split_handling` | YOLO reader split-aware layout: notes which splits were found and which were read |
| `yolo_writer_data_yaml_policy` | YOLO writer emits data.yaml with a names: mapping only (no split paths or nc) |
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
| `label_studio_writer_confidence_routing` | Label Studio writer routes confident annotations to predictions block |
| `cvat_reader_id_assignment` | CVAT reader deterministic ID assignment policy |
| `cvat_reader_attribute_policy` | CVAT reader coordinate + attribute mapping policy |
| `cvat_writer_meta_defaults` | CVAT writer minimal `<meta>` block policy |
| `cvat_writer_deterministic_order` | CVAT writer deterministic ordering (images by filename, boxes by annotation ID) |
| `cvat_writer_image_id_reassignment` | CVAT writer reassigns image IDs sequentially (original `cvat_image_id` not preserved) |
| `cvat_writer_source_default` | CVAT writer defaults missing `source` attribute to `manual` |
| `hf_reader_category_resolution` | HF reader category-name resolution precedence policy |
| `hf_reader_object_container_precedence` | HF reader selects object container: --hf-objects-column, then 'objects', then 'faces' |
| `hf_reader_bbox_format_dependence` | HF reader bbox interpretation depends on --hf-bbox-format flag |
| `hf_writer_deterministic_order` | HF writer deterministic metadata/annotation ordering policy |
| `labelme_reader_id_assignment` | LabelMe reader deterministic ID assignment policy |
| `labelme_reader_path_policy` | LabelMe reader file-name/path derivation policy |
| `labelme_polygon_envelope_applied` | LabelMe reader converted polygon shapes to axis-aligned bbox envelopes |
| `labelme_writer_file_layout` | LabelMe writer canonical annotations/ directory layout |
| `labelme_writer_rectangle_policy` | LabelMe writer emits all annotations as rectangle shapes |
| `labelme_writer_no_image_copy` | LabelMe writer does not copy image files |
| `createml_reader_id_assignment` | CreateML reader deterministic ID assignment policy |
| `createml_reader_image_resolution` | CreateML reader image dimension resolution precedence |
| `createml_writer_deterministic_order` | CreateML writer deterministic ordering policy |
| `createml_writer_coordinate_mapping` | CreateML writer center-based coordinate mapping |
| `createml_writer_no_image_copy` | CreateML writer does not copy image files |
| `kitti_reader_id_assignment` | KITTI reader deterministic ID assignment policy |
| `kitti_reader_field_mapping` | KITTI reader maps non-bbox fields to kitti_* annotation attributes |
| `kitti_reader_image_resolution` | KITTI reader image dimension resolution from image_2/ |
| `kitti_writer_file_layout` | KITTI writer creates label_2/ with one .txt per image |
| `kitti_writer_default_field_values` | KITTI writer uses defaults for missing kitti_* attributes |
| `kitti_writer_deterministic_order` | KITTI writer deterministic ordering policy |
| `kitti_writer_no_image_copy` | KITTI writer does not copy image files |
| `via_reader_id_assignment` | VIA reader deterministic ID assignment policy |
| `via_reader_label_resolution` | VIA reader label resolution from region_attributes |
| `via_reader_image_resolution` | VIA reader image dimension resolution from disk |
| `via_writer_deterministic_order` | VIA writer deterministic ordering policy |
| `via_writer_label_attribute_key` | VIA writer canonical 'label' key in region_attributes |
| `via_writer_no_image_copy` | VIA writer does not copy image files |
| `retinanet_reader_id_assignment` | RetinaNet reader deterministic ID assignment policy |
| `retinanet_reader_image_resolution` | RetinaNet reader image dimension resolution from disk |
| `retinanet_reader_empty_row_handling` | RetinaNet reader empty-row handling for unannotated images |
| `retinanet_writer_deterministic_order` | RetinaNet writer deterministic ordering policy |
| `retinanet_writer_empty_rows` | RetinaNet writer empty-row convention for unannotated images |
| `retinanet_writer_no_image_copy` | RetinaNet writer does not copy image files |

## Blocked conversions

When a conversion is blocked (lossy without `--allow-lossy`), panlabel still
emits the full conversion report to **stdout** before printing the blocking
error to **stderr** and exiting non-zero. This means:

- **Text mode** (default): the report appears on stdout with stable codes in
  brackets (e.g. `[drop_dataset_info]`), then the error on stderr.
- **`--report json`**: stdout contains the full JSON report (parseable by
  downstream tools), stderr contains the concise blocking error.

This lets you inspect exactly what would change before deciding to use
`--allow-lossy`.

## Practical guidance

- Blocked conversions print the full report to stdout — review it to understand
  which warnings triggered the block.
- Use `--report json` for machine-readable output, even on blocked conversions.
- Prefer explicit `--allow-lossy` only when you accept those specific losses.
