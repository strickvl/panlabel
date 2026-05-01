# Panlabel documentation

Welcome! This is the documentation hub for panlabel. Whether you're converting
your first dataset or integrating panlabel into a larger pipeline, you'll find
what you need here.

## What does panlabel support today?

Panlabel currently supports **object detection bounding boxes**. It can read and
write these formats:

- **IR JSON** (`ir-json`) — panlabel's own lossless intermediate representation
- **COCO JSON** (`coco` / `coco-json`) — the widely-used COCO format
- **CVAT XML** (`cvat` / `cvat-xml`) — CVAT for Images annotation export
- **Label Studio JSON** (`label-studio` / `label-studio-json` / `ls`) — task export JSON (`rectanglelabels`)
- **LabelMe JSON** (`labelme` / `labelme-json`) — LabelMe per-image JSON (single file or directory)
- **Apple CreateML JSON** (`create-ml` / `createml` / `create-ml-json`) — Apple CreateML annotation format
- **VIA JSON** (`via` / `via-json` / `vgg-via`) — VGG Image Annotator JSON
- **SuperAnnotate JSON** (`superannotate` / `superannotate-json` / `sa`) — SuperAnnotate JSON export (file or `annotations/` directory)
- **Supervisely JSON** (`supervisely` / `supervisely-json` / `sly`) — Supervisely project / dataset JSON (file, `ann/` directory, or full project with `meta.json`)
- **TFOD CSV** (`tfod` / `tfod-csv`) — TensorFlow Object Detection CSV
- **YOLO directory** (`yolo` / `ultralytics` / `yolov8` / `yolov5`) — Ultralytics-style label directories
- **Pascal VOC XML** (`voc` / `pascal-voc` / `voc-xml`) — VOC-style XML directories
- **KITTI** (`kitti` / `kitti-txt`) — KITTI object detection labels (`label_2/` + `image_2/`)
- **RetinaNet Keras CSV** (`retinanet` / `retinanet-csv` / `keras-retinanet`) — keras-retinanet CSV format
- **OpenImages CSV** (`openimages` / `openimages-csv` / `open-images`) — Google OpenImages CSV annotation format
- **Kaggle Global Wheat CSV** (`kaggle-wheat` / `kaggle-wheat-csv`) — Kaggle Global Wheat Detection CSV
- **Google Cloud AutoML Vision CSV** (`automl-vision` / `automl-vision-csv` / `google-cloud-automl`) — Google Cloud AutoML Vision CSV
- **Udacity Self-Driving Car CSV** (`udacity` / `udacity-csv` / `self-driving-car`) — Udacity Self-Driving Car Dataset CSV
- **Hugging Face ImageFolder metadata** (`hf` / `hf-imagefolder` / `huggingface`) — `metadata.jsonl` / `metadata.parquet` directories (remote Hub import is supported in `convert`)
- **SageMaker Ground Truth Manifest** (`sagemaker` / `sagemaker-manifest` / `sagemaker-ground-truth` / `ground-truth` / `groundtruth` / `aws-sagemaker`) — `.manifest` / `.jsonl` object-detection JSON Lines

Not yet supported: segmentation, keypoints/pose, oriented bounding boxes (OBB),
or classification-only label formats. See the [roadmap](../ROADMAP.md) for
what's planned.

## Which page do I need?

| I want to... | Go to |
|---|---|
| See every CLI flag and command | [CLI reference](./cli.md) |
| Understand how a specific format works | [Format reference](./formats.md) |
| Know what tasks/use cases are supported | [Tasks and use cases](./tasks.md) |
| Understand what gets lost in conversion | [Conversion and lossiness](./conversion.md) |
| Contribute to panlabel | [Contributing guide](../CONTRIBUTING.md) |
| See what's coming next | [Roadmap](../ROADMAP.md) |

## For contributors: source of truth map

If you're working on panlabel's code or docs, here's where the authoritative
behavior lives:

| Topic | Primary source |
|---|---|
| CLI commands, flags, auto-detection | `src/lib.rs` |
| COCO format behavior | `src/ir/io_coco_json.rs` |
| CVAT XML format behavior | `src/ir/io_cvat_xml.rs` |
| TFOD format behavior | `src/ir/io_tfod_csv.rs` |
| Label Studio format behavior | `src/ir/io_label_studio_json.rs` |
| LabelMe format behavior | `src/ir/io_labelme_json.rs` |
| Apple CreateML format behavior | `src/ir/io_createml_json.rs` |
| VIA JSON format behavior | `src/ir/io_via_json.rs` |
| SuperAnnotate format behavior | `src/ir/io_superannotate_json.rs` (+ `src/ir/io_super_json_common.rs`) |
| Supervisely format behavior | `src/ir/io_supervisely_json.rs` (+ `src/ir/io_super_json_common.rs`) |
| YOLO format behavior | `src/ir/io_yolo.rs` |
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
