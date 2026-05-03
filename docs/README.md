# Panlabel documentation

Welcome! This is the documentation hub for panlabel. Whether you're converting
your first dataset or integrating panlabel into a larger pipeline, you'll find
what you need here.

## What does panlabel support today?

Panlabel currently supports **mainstream/static-image 2D axis-aligned object detection bounding boxes**. It can read and
write these formats:

- **IR JSON** (`ir-json`) — panlabel's own lossless intermediate representation
- **COCO JSON** (`coco` / `coco-json`) — the widely-used COCO format
- **IBM Cloud Annotations JSON** (`ibm-cloud-annotations` / `cloud-annotations`) — localization `_annotations.json` files/directories
- **CVAT XML** (`cvat` / `cvat-xml`) — CVAT for Images annotation export
- **Label Studio JSON** (`label-studio` / `label-studio-json` / `ls`) — task export JSON (`rectanglelabels`)
- **Labelbox JSON/NDJSON** (`labelbox` / `labelbox-json` / `labelbox-ndjson`) — current export rows with `data_row` and nested project labels
- **Scale AI JSON** (`scale-ai` / `scale` / `scale-ai-json`) — image annotation task/response JSON with boxes, polygon envelopes, and rotated-box envelopes
- **Unity Perception JSON** (`unity-perception` / `unity` / `solo`) — Unity/SOLO synthetic-data frame/captures JSON with `BoundingBox2D` values
- **LabelMe JSON** (`labelme` / `labelme-json`) — LabelMe per-image JSON (single file or directory)
- **Apple CreateML JSON** (`create-ml` / `createml` / `create-ml-json`) — Apple CreateML annotation format
- **VIA JSON** (`via` / `via-json` / `vgg-via`) — VGG Image Annotator JSON
- **VIA CSV** (`via-csv` / `vgg-via-csv`) — VGG Image Annotator CSV (separate from VIA JSON)
- **SuperAnnotate JSON** (`superannotate` / `superannotate-json` / `sa`) — SuperAnnotate JSON export (file or `annotations/` directory)
- **Supervisely JSON** (`supervisely` / `supervisely-json` / `sly`) — Supervisely project / dataset JSON (file, `ann/` directory, or full project with `meta.json`)
- **Cityscapes JSON** (`cityscapes` / `cityscapes-json`) — Cityscapes polygon JSON (file, `gtFine/`, or dataset root), flattened to bbox envelopes
- **Marmot XML** (`marmot` / `marmot-xml`) — Marmot document-layout XML with hex-double CropBox/BBox composites
- **TFOD CSV** (`tfod` / `tfod-csv`) — TensorFlow Object Detection CSV
- **TFRecord** (`tfrecord` / `tfrecords` / `tf-record` / `tfod-tfrecord` / `tfod-tfrerecord`) — TensorFlow Object Detection API-style `tf.train.Example` records (single-file, uncompressed, bbox-only in v1)
- **VoTT CSV** (`vott-csv` / `vott`) — Microsoft VoTT headered `image,xmin,ymin,xmax,ymax,label` CSV
- **VoTT JSON** (`vott-json` / `vott-json-export`) — Microsoft VoTT aggregate/per-asset JSON with `regions`
- **YOLO directory/list splits** (`yolo` / `ultralytics` / `yolov8` / `yolov5` / `scaled-yolov4` / `scaled-yolov4-txt`) — YOLO label directories, including `data.yaml` splits that point to image-list `.txt` files
- **YOLO Keras TXT** (`yolo-keras` / `yolo-keras-txt` / `keras-yolo`) — single-file absolute XYXY rows: `image xmin,ymin,xmax,ymax,class_id ...`
- **YOLOv4 PyTorch TXT** (`yolov4-pytorch` / `yolov4-pytorch-txt` / `pytorch-yolov4`) — same shared absolute-coordinate TXT grammar as YOLO Keras
- **Pascal VOC XML** (`voc` / `pascal-voc` / `voc-xml`) — VOC-style XML directories
- **KITTI** (`kitti` / `kitti-txt`) — KITTI object detection labels (`label_2/` + `image_2/`)
- **RetinaNet Keras CSV** (`retinanet` / `retinanet-csv` / `keras-retinanet`) — keras-retinanet CSV format
- **OpenImages CSV** (`openimages` / `openimages-csv` / `open-images`) — Google OpenImages CSV annotation format
- **Kaggle Global Wheat CSV** (`kaggle-wheat` / `kaggle-wheat-csv`) — Kaggle Global Wheat Detection CSV
- **Google Cloud AutoML Vision CSV** (`automl-vision` / `automl-vision-csv` / `google-cloud-automl`) — Google Cloud AutoML Vision CSV
- **Udacity Self-Driving Car CSV** (`udacity` / `udacity-csv` / `self-driving-car`) — Udacity Self-Driving Car Dataset CSV
- **Hugging Face ImageFolder metadata** (`hf` / `hf-imagefolder` / `huggingface`) — `metadata.jsonl` / `metadata.parquet` directories (remote Hub import is supported in `convert`)
- **SageMaker Ground Truth Manifest** (`sagemaker` / `sagemaker-manifest` / `sagemaker-ground-truth` / `ground-truth` / `groundtruth` / `aws-sagemaker`) — `.manifest` / `.jsonl` object-detection JSON Lines
- **Datumaro JSON** (`datumaro` / `datumaro-json` / `datumaro-dataset`) — Datumaro annotation JSON
- **WIDER Face TXT** (`wider-face` / `widerface` / `wider-face-txt`) — aggregate face bbox TXT (single `face` category in panlabel)
- **OIDv4 TXT** (`oidv4` / `oidv4-txt` / `openimages-v4-txt` / `oid`) — toolkit TXT labels with `Label/` directories
- **BDD100K / Scalabel JSON** (`bdd100k` / `bdd100k-json` / `scalabel` / `scalabel-json`) — bbox detection subset
- **V7 Darwin JSON** (`v7-darwin` / `darwin` / `darwin-json` / `v7`) — bbox detection subset
- **Edge Impulse labels JSON** (`edge-impulse` / `edge-impulse-labels`) — `bounding_boxes.labels`
- **ASAM OpenLABEL JSON** (`openlabel` / `asam-openlabel` / `openlabel-json`) — static-image 2D bbox subset

Not yet supported as first-class tasks: segmentation, keypoints/pose, oriented bounding boxes (OBB),
video tracking IDs, 3D/multisensor labels, or classification-only label formats.
When these richer structures appear inside broad schemas, panlabel skips/reports them or treats the conversion as lossy.
See the [roadmap](../ROADMAP.md) for what's planned.

## Which page do I need?

| I want to... | Go to |
|---|---|
| See every CLI flag and command | [CLI reference](./cli.md) |
| Understand how a specific format works | [Format reference](./formats.md) |
| Know what tasks/use cases are supported | [Tasks and use cases](./tasks.md) |
| Understand what gets lost in conversion | [Conversion and lossiness](./conversion.md) |
| Contribute to panlabel | [Contributing guide](../CONTRIBUTING.md) |
| Run and maintain fuzzing coverage | [Fuzzing guide](./fuzzing.md) |
| See what's coming next | [Roadmap](../ROADMAP.md) |

## For contributors: source of truth map

If you're working on panlabel's code or docs, here's where the authoritative
behavior lives:

| Topic | Primary source |
|---|---|
| CLI commands, flags, auto-detection | `src/lib.rs` |
| COCO format behavior | `src/ir/io_coco_json.rs` |
| IBM Cloud Annotations behavior | `src/ir/io_cloud_annotations_json.rs` |
| CVAT XML format behavior | `src/ir/io_cvat_xml.rs` |
| TFOD format behavior | `src/ir/io_tfod_csv.rs` |
| TFRecord format behavior | `src/ir/io_tfrecord.rs` |
| VoTT CSV format behavior | `src/ir/io_vott_csv.rs` |
| VoTT JSON format behavior | `src/ir/io_vott_json.rs` |
| Label Studio format behavior | `src/ir/io_label_studio_json.rs` |
| Labelbox format behavior | `src/ir/io_labelbox_json.rs` |
| Scale AI format behavior | `src/ir/io_scale_ai_json.rs` |
| Unity Perception format behavior | `src/ir/io_unity_perception_json.rs` |
| LabelMe format behavior | `src/ir/io_labelme_json.rs` |
| Apple CreateML format behavior | `src/ir/io_createml_json.rs` |
| VIA JSON format behavior | `src/ir/io_via_json.rs` |
| VIA CSV format behavior | `src/ir/io_via_csv.rs` |
| SuperAnnotate format behavior | `src/ir/io_superannotate_json.rs` (+ `src/ir/io_super_json_common.rs`) |
| Supervisely format behavior | `src/ir/io_supervisely_json.rs` (+ `src/ir/io_super_json_common.rs`) |
| Cityscapes format behavior | `src/ir/io_cityscapes_json.rs` |
| Marmot format behavior | `src/ir/io_marmot_xml.rs` |
| Datumaro format behavior | `src/ir/io_datumaro_json.rs` |
| WIDER Face format behavior | `src/ir/io_wider_face_txt.rs` |
| OIDv4 format behavior | `src/ir/io_oidv4_txt.rs` |
| BDD100K format behavior | `src/ir/io_bdd100k_json.rs` |
| V7 Darwin format behavior | `src/ir/io_v7_darwin_json.rs` |
| Edge Impulse format behavior | `src/ir/io_edge_impulse_labels.rs` |
| OpenLABEL format behavior | `src/ir/io_openlabel_json.rs` |
| YOLO format behavior | `src/ir/io_yolo.rs` |
| YOLO Keras / YOLOv4 PyTorch TXT behavior | `src/ir/io_yolo_keras_txt.rs` |
| Pascal VOC format behavior | `src/ir/io_voc_xml.rs` |
| KITTI format behavior | `src/ir/io_kitti.rs` |
| RetinaNet Keras CSV format behavior | `src/ir/io_retinanet_csv.rs` |
| OpenImages CSV format behavior | `src/ir/io_openimages_csv.rs` |
| Kaggle Wheat CSV format behavior | `src/ir/io_kaggle_wheat_csv.rs` |
| AutoML Vision CSV format behavior | `src/ir/io_automl_vision_csv.rs` |
| Udacity CSV format behavior | `src/ir/io_udacity_csv.rs` |
| HF ImageFolder format behavior | `src/ir/io_hf_imagefolder.rs` (+ `src/ir/io_hf_parquet.rs` with `hf-parquet`) |
| SageMaker Ground Truth Manifest behavior | `src/ir/io_sagemaker_manifest.rs` |
| HF remote resolve/preflight/acquire | `src/hf/` (`hf-remote` feature) |
| Lossiness logic | `src/conversion/mod.rs` |
| Stable conversion issue codes | `src/conversion/report.rs` |
| User-visible CLI behavior tests | `tests/cli.rs` |
| Format roundtrip behavior tests | `tests/*_roundtrip.rs` |
| Property-based adapter invariants | `tests/proptest_*.rs` + `tests/proptest_helpers/mod.rs` |
| Fuzz parser coverage | `fuzz/fuzz_targets/*.rs` + `fuzz/corpus/*` |

Design notes in `design/` (for example `design/label-studio-plan.md`) are historical background only. For implemented behavior, prefer code + tests + these docs.
