# CLI reference

Everything you can do with the `panlabel` command line.

## Global

- Binary name: `panlabel`
- Version: `panlabel -V`
- Help: `panlabel --help` and `panlabel <command> --help`

## Commands

### `validate`

Validate a dataset path and print a validation report.

- Positional: `input` (path; file or directory depending on format)
- `--format <string>` (default: `ir-json`)
  - supported values: `ir-json`, `coco`, `coco-json`, `cvat`, `cvat-xml`, `label-studio`, `label-studio-json`, `ls`, `tfod`, `tfod-csv`, `yolo`, `ultralytics`, `yolov8`, `yolov5`, `voc`, `pascal-voc`, `voc-xml`, `hf`, `hf-imagefolder`, `huggingface`
- `--strict` (treat warnings as errors)
- `--output <string>` (`text` or `json`, default: `text`)

---

### `convert`

Convert annotations between formats using IR as the internal hub.

- `--from`, `-f`: `auto`, `ir-json`, `coco`, `coco-json`, `cvat`, `cvat-xml`, `label-studio`, `label-studio-json`, `ls`, `tfod`, `tfod-csv`, `yolo`, `ultralytics`, `yolov8`, `yolov5`, `voc`, `pascal-voc`, `voc-xml`, `hf`, `hf-imagefolder`, `huggingface`
- `--to`, `-t`: `ir-json`, `coco`, `coco-json`, `cvat`, `cvat-xml`, `label-studio`, `label-studio-json`, `ls`, `tfod`, `tfod-csv`, `yolo`, `ultralytics`, `yolov8`, `yolov5`, `voc`, `pascal-voc`, `voc-xml`, `hf`, `hf-imagefolder`, `huggingface`
- `--input`, `-i`: input path (required for local inputs; optional with `--hf-repo` when `--from hf`)
- `--output`, `-o`: output path
- `--strict`
- `--no-validate`
- `--allow-lossy`
- `--report <text|json>` (default: `text`)

HF-specific options (meaningful only with `--from hf` or `--to hf`):
- `--hf-bbox-format <xywh|xyxy>` (default: `xywh`)
- `--hf-objects-column <name>`
- `--hf-category-map <path>`
- `--hf-repo <namespace/dataset-or-url>` (remote import, `convert` only)
- `--split <name>`
- `--revision <ref>`
- `--config <name>`
- `--token <token>` (also reads `HF_TOKEN`)

With `--report json`, panlabel prints JSON only to stdout.

Notes:
- `--hf-repo` can only be used with `--from hf`.
- `--revision`/`--config` require `--hf-repo`.
- Remote HF import (`--hf-repo`) needs a build with feature `hf-remote` (for full HF support from source: `cargo install panlabel --features hf`).
- Remote HF parquet datasets commonly use split shard files (for example `data/train-*.parquet`); these are supported with `hf-parquet`.

---

### `stats`

Show rich dataset statistics.

- Positional: `input`
- `--format <format>` (optional; if omitted panlabel auto-detects)
  - when detection fails for a **JSON file**, stats falls back to `ir-json`
- `--top <N>` (default: `10`) for label and co-occurrence top lists
- `--tolerance <PX>` (default: `0.5`) for OOB checks
- `--output <text|json|html>` (default: `text`)

`--output html` returns a self-contained HTML report on stdout.

---

### `diff`

Compare two datasets semantically.

Usage:
`panlabel diff [OPTIONS] <INPUT_A> <INPUT_B>`

- `--format-a <FORMAT>` (default: `auto`)
- `--format-b <FORMAT>` (default: `auto`)
- `--match-by <id|iou>` (default: `id`)
- `--iou-threshold <FLOAT>` (default: `0.5`, used by `--match-by iou`)
- `--detail` for item-level details
- `--output <text|json>` (default: `text`)

---

### `sample`

Create a subset dataset.

Usage:
`panlabel sample [OPTIONS] -i <INPUT> -o <OUTPUT>`

- `--from <FORMAT>` (default: `auto`)
- `--to <FORMAT>` (optional)
  - if omitted and `--from` is explicit, output uses same format
  - if omitted and `--from auto`, output defaults to `ir-json`
- `-n <COUNT>` or `--fraction <FLOAT>` (exactly one required)
- `--seed <INT>` for deterministic sampling
- `--strategy <random|stratified>` (default: `random`)
- `--categories <comma,separated,list>`
- `--category-mode <images|annotations>` (default: `images`)
- `--allow-lossy`

Sampling keeps original IDs and keeps all categories in output.

---

### `list-formats`

Show format capabilities and lossiness class.

## Auto-detection rules (`convert --from auto`, `diff --format-* auto`, `sample --from auto`, `stats` without `--format`)

1. If input path is a directory:
   - YOLO marker: `labels/` with `.txt` labels (or path itself is `labels/`)
   - VOC marker: `Annotations/` with `.xml` plus `JPEGImages/` (or path itself is `Annotations/` with sibling `JPEGImages/`)
   - CVAT marker: `annotations.xml` at directory root
   - HF marker: `metadata.jsonl` or `metadata.parquet` at root or in an immediate subdirectory
   - if multiple markers match, detection fails (ambiguous), including HF+YOLO, HF+VOC, HF+CVAT
2. If input path is a file:
   - `.csv` -> `tfod`
   - `.xml`:
     - root `<annotations>` -> `cvat`
   - `.json`:
     - array-root empty or Label Studio task shape -> `label-studio`
     - object-root with `annotations[0].bbox` array -> `coco`
     - object-root with bbox object (`min/max` or `xmin/ymin/xmax/ymax`) -> `ir-json`

## Examples

```bash
# Validate a YOLO dataset root
panlabel validate /data/my_yolo --format yolo

# Auto-detect YOLO from directory, convert to COCO
panlabel convert --from auto --to coco -i /data/my_yolo -o out.json

# Dataset stats as JSON
panlabel stats --output json tests/fixtures/sample_valid.coco.json

# Dataset diff with details
panlabel diff --match-by id --detail a.ir.json b.ir.json

# Category-focused sampling
panlabel sample -i in.coco.json -o out.ir.json --from coco --to ir-json --categories person,car --category-mode images -n 100 --seed 42

# Convert a local HF ImageFolder directory to COCO
panlabel convert --from hf --to coco -i ./hf_dataset -o out.coco.json

# Convert a remote HF dataset repo to IR JSON (requires build with --features hf)
panlabel convert --from hf --to ir-json --hf-repo rishitdagli/cppe-5 --split train -o out.ir.json
```
