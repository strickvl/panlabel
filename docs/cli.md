# CLI reference

Everything you can do with the `panlabel` command line. If you're just getting
started, the [Quick Start in the README](../README.md#quick-start) is a good
place to begin — then come back here when you need the full details.

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
  - supported values: `ir-json`, `coco`, `coco-json`, `tfod`, `tfod-csv`, `yolo`, `ultralytics`, `yolov8`, `yolov5`, `voc`, `pascal-voc`, `voc-xml`
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
- `--from`: `auto`, `ir-json`, `coco`, `coco-json`, `tfod`, `tfod-csv`, `yolo`, `ultralytics`, `yolov8`, `yolov5`, `voc`, `pascal-voc`, `voc-xml`
- `--to`: `ir-json`, `coco`, `tfod`, `yolo`, `voc` (aliases also accepted by clap where configured)

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
- `--format <ir-json|coco|tfod|yolo|voc>`
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
   - if `<path>/labels` exists and contains `.txt` label files recursively → YOLO marker
   - OR if `<path>` is itself a `labels` directory with `.txt` files recursively → YOLO marker
   - if `<path>/Annotations` exists with `.xml` files recursively and `<path>/JPEGImages` exists → VOC marker
   - OR if `<path>` is itself an `Annotations` directory with `.xml` files recursively and sibling `../JPEGImages` exists → VOC marker
   - if both YOLO and VOC markers match, detection fails with an ambiguity error (use `--from`)
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

# Convert Pascal VOC -> COCO
panlabel convert -f voc -t coco -i ./voc_dataset -o out.json
```
