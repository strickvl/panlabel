# CLI reference

Everything you can do with the `panlabel` command line.

## Global

- Binary name: `panlabel`
- Version: `panlabel -V`
- Help: `panlabel --help` and `panlabel <command> --help`

## Machine-readable output

Panlabel now has one cross-command spelling for structured stdout: `--output-format`.

- Read-only commands use `--output-format` and also accept `--output` as a backward-compatible alias:
  - `validate`
  - `stats`
  - `diff`
  - `list-formats`
- `convert` and `sample` use `--output-format <text|json>` for report formatting because `-o/--output` is already the filesystem output path.
- `convert` and `sample` also accept `--report <text|json>` as a backward-compatible alias.
- In JSON mode, structured payloads go to stdout. Fatal errors still go to stderr.
- JSON is pretty-printed when stdout is an interactive terminal, and compact when stdout is piped or captured.
- `stats` text output is rich/Unicode on a terminal, but switches to a plain text layout (ASCII framing/bars, no box-drawing or emoji) when stdout is piped or captured.

## Commands

### `validate`

Validate a dataset path and print a validation report.

- Positional: `input` (path; file or directory depending on format)
- `--format <format>` (default: `ir-json`)
  - supported values: `ir-json`, `coco`, `coco-json`, `cvat`, `cvat-xml`, `label-studio`, `label-studio-json`, `ls`, `tfod`, `tfod-csv`, `yolo`, `ultralytics`, `yolov8`, `yolov5`, `voc`, `pascal-voc`, `voc-xml`, `hf`, `hf-imagefolder`, `huggingface`
- `--strict` (treat warnings as errors)
- `--output-format <text|json>` (default: `text`)
- `--output <text|json>` (backward-compatible alias)

Invalid `--format` and output mode values are rejected by clap at parse time.

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
- `--dry-run` (run detection/validation/reporting without writing output files)
- `--output-format <text|json>` (default: `text`)
- `--report <text|json>` (backward-compatible alias for `--output-format`)

Shared options:
- `--split <name>` — select a single split for HF or YOLO imports (see below)

HF-specific options (meaningful only with `--from hf` or `--to hf`):
- `--hf-bbox-format <xywh|xyxy>` (default: `xywh`)
- `--hf-objects-column <name>`
- `--hf-category-map <path>`
- `--hf-repo <namespace/dataset-or-url>` (remote import, `convert` only)
- `--revision <ref>`
- `--config <name>`
- `--token <token>` (also reads `HF_TOKEN`)

With `--output-format json`, the conversion report is printed as JSON to stdout.
On blocked lossy conversions, stdout still contains the full JSON report
while the blocking error goes to stderr (exit code 1).
With `--dry-run`, panlabel still runs format detection, input validation, and lossiness analysis, but skips the final write step.

Notes:
- `--split` can be used with `--from hf` or `--from yolo`. For YOLO, it selects a single split from a split-aware dataset layout (e.g. `--split train`). Without `--split`, all splits are merged.
- `--hf-repo` can only be used with `--from hf`.
- `--revision`/`--config` require `--hf-repo`.
- Remote HF import (`--hf-repo`) needs a build with feature `hf-remote` (for full HF support from source: `cargo install panlabel --features hf`).
- Remote HF parquet datasets commonly use split shard files (for example `data/train-*.parquet`); these are supported with `hf-parquet`.
- Remote HF zip-style splits (for example `data/train.zip`) are supported when the extracted payload looks like YOLO, VOC, COCO JSON, or HF metadata layout.
- `--output` is still required even with `--dry-run`, so the report can say what would be written.
- `--dry-run` does **not** prove the output path is writable; it skips filesystem writes entirely.
- In `--output-format json` mode, dry runs emit the same conversion-report schema as normal runs (no extra wrapper field).

---

### `stats`

Show rich dataset statistics.

- Positional: `input`
- `--format <format>` (optional; if omitted panlabel auto-detects)
  - when detection fails for a **parseable JSON file**, stats falls back to `ir-json`
  - malformed JSON surfaces the parse error directly (no silent fallback)
- `--top <N>` (default: `10`) for label and co-occurrence top lists
- `--tolerance <PX>` (default: `0.5`) for OOB checks
- `--output-format <text|json|html>` (default: `text`)
- `--output <text|json|html>` (backward-compatible alias)

`--output html` returns a self-contained HTML report on stdout.
Text output uses the rich terminal renderer on a TTY and a plain text renderer when stdout is piped or captured.

---

### `diff`

Compare two datasets semantically.

Usage:
`panlabel diff [OPTIONS] <INPUT_A> <INPUT_B>`

- `--format-a <FORMAT>` (default: `auto`)
- `--format-b <FORMAT>` (default: `auto`)
- `--match-by <id|iou>` (default: `id`)
- `--iou-threshold <FLOAT>` (default: `0.5`, used by `--match-by iou`; must be in `(0.0, 1.0]`)
- `--detail` for item-level details
- `--output-format <text|json>` (default: `text`)
- `--output <text|json>` (backward-compatible alias)

Constraints:
- Each input dataset must have unique `image.file_name` values for reliable diffing.
- `--iou-threshold` is validated only when `--match-by iou` is used.

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
- `--dry-run` (sample in memory and report what would be written, without writing output files)
- `--output-format <text|json>` (default: `text`)
- `--report <text|json>` (alias for `--output-format`)

Sampling keeps original IDs and keeps all categories in output.

In text mode, sample prints a short summary line followed by the conversion report.
In JSON mode, sample prints only the conversion report JSON to stdout.
Blocked lossy sampling mirrors `convert`: stdout gets the full report, stderr gets the concise blocking error.

Notes:
- `--output` is still required even with `--dry-run`.
- `--dry-run` skips filesystem writes entirely, so it does not check whether the output path is writable.
- Use `--seed` if you want repeated dry runs to choose the same sampled subset.
- In `--output-format json` mode, dry runs emit the same conversion-report schema as normal runs.

---

### `list-formats`

Show format capabilities and lossiness class.

- `--output-format <text|json>` (default: `text`)
- `--output <text|json>` (backward-compatible alias)

`list-formats --output-format json` emits a JSON array. Each entry has:

- `name`
- `aliases`
- `read`
- `write`
- `lossiness` (`lossless`, `conditional`, or `lossy`)
- `description`
- `file_based`
- `directory_based`

## Auto-detection rules (`convert --from auto`, `diff --format-* auto`, `sample --from auto`, `stats` without `--format`)

1. If input path is a directory:
   - YOLO marker: `labels/` with `.txt` labels AND sibling `images/` directory (or path itself is `labels/` with sibling `images/`). If `labels/` with `.txt` files exist but `images/` is missing, this is reported as an incomplete layout.
   - VOC marker: `Annotations/` with top-level `.xml` files (or path itself is `Annotations/`). `JPEGImages/` is optional, matching the reader's behavior.
   - CVAT marker: `annotations.xml` at directory root
   - HF marker: `metadata.jsonl` or `metadata.parquet` at root or in an immediate subdirectory, or parquet shard files (e.g. `data/train-*.parquet`)
   - if multiple markers match, detection fails with an ambiguity error listing the evidence for each format
   - if only partial matches exist (e.g. YOLO labels without images), the error explains what's missing
2. If input path is a file:
   - `.csv` -> `tfod`
   - `.xml`:
     - root `<annotations>` -> `cvat`
   - `.json`:
     - array-root empty or Label Studio task shape -> `label-studio`
     - object-root with `annotations[0].bbox` array -> `coco`
     - object-root with bbox object (`min/max` or `xmin/ymin/xmax/ymax`) -> `ir-json`
3. `stats` fallback: when detection fails for a `.json` file, stats tries `ir-json` as a fallback — but only if the JSON is parseable. Malformed JSON is reported directly as a parse error.

## Examples

```bash
# Validate a YOLO dataset root
panlabel validate /data/my_yolo --format yolo

# Validate with machine-readable output
panlabel validate tests/fixtures/sample_valid.ir.json --output-format json

# Auto-detect YOLO from directory, convert to COCO
panlabel convert --from auto --to coco -i /data/my_yolo -o out.json

# Machine-readable conversion report
panlabel convert --from auto --to coco -i in.json -o out.coco.json --output-format json

# Preview a conversion without writing output files
panlabel convert --from auto --to coco -i in.json -o out.coco.json --dry-run

# Dataset stats as JSON
panlabel stats --output-format json tests/fixtures/sample_valid.coco.json

# Dataset diff with details
panlabel diff --match-by id --detail a.ir.json b.ir.json

# Category-focused sampling with JSON report output
panlabel sample -i in.coco.json -o out.ir.json --from coco --to ir-json --categories person,car --category-mode images -n 100 --seed 42 --output-format json

# Preview a deterministic sample without writing output files
panlabel sample -i in.coco.json -o out.ir.json --from coco --to ir-json -n 100 --seed 42 --dry-run

# Machine-readable format discovery
panlabel list-formats --output-format json

# Convert a local HF ImageFolder directory to COCO
panlabel convert --from hf --to coco -i ./hf_dataset -o out.coco.json

# Convert a remote HF dataset repo to IR JSON (requires build with --features hf)
panlabel convert --from hf --to ir-json --hf-repo rishitdagli/cppe-5 --split train -o out.ir.json

# Zip-style remote dataset (auto-routed after extraction, still invoked as --from hf)
panlabel convert --from hf --to ir-json --hf-repo keremberke/football-object-detection --split train -o out.ir.json

# Convert a split-aware YOLO dataset (merges all splits by default)
panlabel convert --from yolo --to coco -i ./yolo_dataset -o out.coco.json --allow-lossy

# Convert only the train split from a YOLO dataset
panlabel convert --from yolo --to coco -i ./yolo_dataset -o out.coco.json --split train --allow-lossy
```
