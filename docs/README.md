# Panlabel documentation

Welcome! This is the documentation hub for panlabel. Whether you're converting
your first dataset or integrating panlabel into a larger pipeline, you'll find
what you need here.

## What does panlabel support today?

Panlabel currently supports **object detection bounding boxes**. It can read and
write these formats:

- **IR JSON** (`ir-json`) — panlabel's own lossless intermediate representation
- **COCO JSON** (`coco` / `coco-json`) — the widely-used COCO format
- **Label Studio JSON** (`label-studio` / `label-studio-json` / `ls`) — task export JSON (`rectanglelabels`)
- **TFOD CSV** (`tfod` / `tfod-csv`) — TensorFlow Object Detection CSV
- **YOLO directory** (`yolo` / `ultralytics` / `yolov8` / `yolov5`) — Ultralytics-style label directories
- **Pascal VOC XML** (`voc` / `pascal-voc` / `voc-xml`) — VOC-style XML directories
- **Hugging Face ImageFolder metadata** (`hf` / `hf-imagefolder` / `huggingface`) — `metadata.jsonl` / `metadata.parquet` directories (remote Hub import is supported in `convert`)

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
| TFOD format behavior | `src/ir/io_tfod_csv.rs` |
| Label Studio format behavior | `src/ir/io_label_studio_json.rs` |
| YOLO format behavior | `src/ir/io_yolo.rs` |
| Pascal VOC format behavior | `src/ir/io_voc_xml.rs` |
| HF ImageFolder format behavior | `src/ir/io_hf_imagefolder.rs` (+ `src/ir/io_hf_parquet.rs` with `hf-parquet`) |
| HF remote resolve/preflight/acquire | `src/hf/` (`hf-remote` feature) |
| Lossiness logic | `src/conversion/mod.rs` |
| Stable conversion issue codes | `src/conversion/report.rs` |
| User-visible CLI behavior tests | `tests/cli.rs` |
| Format roundtrip behavior tests | `tests/*_roundtrip.rs` |
| Property-based adapter invariants | `tests/proptest_*.rs` + `tests/proptest_helpers/mod.rs` |
| Fuzz parser coverage | `fuzz/fuzz_targets/*.rs` + `fuzz/corpus/*` |

Design notes in `design/` (for example `design/label-studio-plan.md`) are historical background only. For implemented behavior, prefer code + tests + these docs.
