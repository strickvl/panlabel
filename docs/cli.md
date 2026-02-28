# CLI reference

This page describes the current command-line contract implemented in `src/lib.rs`.

## Global

- Binary name: `panlabel`
- Version: `panlabel -V`
- Help: `panlabel --help` and `panlabel <command> --help`

## Commands

### `validate`

Validate a dataset path and print a validation report.

### Inputs
- Positional: `input` (path; file or directory depending on format)

### Flags
- `--format <string>` (default: `ir-json`)
  - supported values: `ir-json`, `coco`, `coco-json`, `tfod`, `tfod-csv`, `yolo`, `ultralytics`, `yolov8`, `yolov5`
- `--strict` (treat warnings as errors)
- `--output <string>` (`text` or `json`, default: `text`)

---

### `convert`

Convert annotations between formats using IR as the internal hub.

### Inputs
- `--from`, `-f` source format
- `--to`, `-t` target format
- `--input`, `-i` input path
- `--output`, `-o` output path

### Format values
- `--from`: `auto`, `ir-json`, `coco`, `coco-json`, `tfod`, `tfod-csv`, `yolo`, `ultralytics`, `yolov8`, `yolov5`
- `--to`: `ir-json`, `coco`, `tfod`, `yolo` (aliases also accepted by clap where configured)

### Flags
- `--strict`
- `--no-validate`
- `--allow-lossy`
- `--report <text|json>` (default: `text`)

### Output behavior
- Validation issues encountered during convert may print to **stderr**.
- With `--report text`, panlabel prints conversion summary text.
- With `--report json`, panlabel prints JSON report to stdout (machine-consumable).

---

### `inspect`

Show dataset summary statistics.

### Inputs
- Positional: `input` path

### Flags
- `--format <ir-json|coco|tfod|yolo>`
- `--top <usize>` (default: `10`)
- `--tolerance <f64>` (default: `0.5`)

---

### `list-formats`

Show format capabilities and lossiness class.

The table includes:
- format name
- read support
- write support
- lossiness (`lossless`, `conditional`, `lossy`)

## Auto-detection rules (`convert --from auto`)

Implemented detection logic:

1. If input path is a directory:
   - if `<path>/labels` exists and contains `.txt` label files recursively → `yolo`
   - OR if `<path>` is itself a `labels` directory with `.txt` files recursively → `yolo`
2. If input path is a file:
   - `.csv` → `tfod`
   - `.json` → inspect `annotations[0].bbox`
     - bbox array `[x,y,w,h]` → `coco`
     - bbox object (`min/max` or `xmin/ymin/xmax/ymax`) → `ir-json`

Auto-detect limitation for JSON: it requires a non-empty `annotations` array and a `bbox` in the first annotation. If these are missing/empty, detection fails; pass `--from` explicitly.

## Examples

```bash
# Validate a YOLO dataset root
panlabel validate /data/my_yolo --format yolo

# Auto-detect YOLO from directory, convert to COCO
panlabel convert --from auto --to coco -i /data/my_yolo -o out.json

# Convert IR -> YOLO (lossy, requires opt-in)
panlabel convert -f ir-json -t yolo -i in.ir.json -o out_yolo --allow-lossy
```
