# Panlabel

[![CI](https://github.com/strickvl/panlabel/actions/workflows/ci.yml/badge.svg)](https://github.com/strickvl/panlabel/actions/workflows/ci.yml)
![Crates.io Version](https://img.shields.io/crates/v/panlabel)
![PyPI Version](https://img.shields.io/pypi/v/panlabel)
![GitHub License](https://img.shields.io/github/license/strickvl/panlabel)
![GitHub Repo stars](https://img.shields.io/github/stars/strickvl/panlabel)
![Crates.io Total Downloads](https://img.shields.io/crates/d/panlabel)
![PyPI Downloads](https://img.shields.io/pypi/dm/panlabel)

## The universal annotation converter

If you've ever written a one-off Python script to wrangle COCO annotations into
YOLO format (or vice versa), panlabel is here to save you the trouble. It's a
fast, single-binary CLI that converts between common object detection annotation
formats — with built-in validation, clear lossiness warnings, and no Python
dependencies to manage.

Panlabel is also available as a Rust library if you want to integrate format
conversion into your own tools.

> **Note**: Panlabel is in active development (v0.5.x). The CLI and library APIs
> may change between versions, so pin to a specific version if you're using it in
> production.

## Installation

### pip / uv (any platform)

```sh
pip install panlabel
# or
uv pip install panlabel
```

This installs a pre-built binary — no Rust toolchain needed.

### Homebrew (macOS / Linux)

```sh
brew install strickvl/tap/panlabel
```

### Shell script (macOS / Linux)

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/strickvl/panlabel/releases/latest/download/panlabel-installer.sh | sh
```

### PowerShell (Windows)

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/strickvl/panlabel/releases/latest/download/panlabel-installer.ps1 | iex"
```

### Cargo (from source)

```sh
cargo install panlabel
# Enable full HF support (remote Hub import + metadata.parquet)
cargo install panlabel --features hf
```

### Pre-built binaries

Download from the [latest GitHub Release](https://github.com/strickvl/panlabel/releases/latest). Builds are available for macOS (Intel + Apple Silicon), Linux (x86_64 + ARM64), and Windows.

### Docker

```sh
docker pull strickvl/panlabel
# Convert a COCO file in your current directory to YOLO
docker run --rm -v "$PWD":/data strickvl/panlabel convert -f coco -t yolo -i /data/annotations.json -o /data/yolo_out --allow-lossy
```

Multi-arch images (amd64 + arm64) are published for each release.

### As a Rust library

```sh
cargo add panlabel
```

## Quick start

```sh
# Convert between formats (auto-detects the input)
panlabel convert --from auto --to yolo -i annotations.json -o ./yolo_out --allow-lossy

# Check a dataset for problems before training
panlabel validate --format coco annotations.json

# Get a quick overview of what's in a dataset
panlabel stats --format coco annotations.json

# Compare two datasets semantically
panlabel diff --format-a auto --format-b auto old.json new.json

# Sample a smaller subset for quick experiments
panlabel sample -i annotations.json -o sample.ir.json --from auto --to ir-json -n 100 --seed 42

# See every supported format and its capabilities
panlabel list-formats
```

The `convert` shape is always `-f <source> -t <dest> -i <input> -o <output>` — pick any source/destination from the [Supported formats](#supported-formats) table. See [More convert examples](#more-convert-examples) below for lossless vs. lossy conversions, machine-readable JSON reports, dry runs, and remote Hugging Face datasets.

## What can panlabel do?

| Command | What it does |
|---------|-------------|
| `convert` | Convert between annotation formats, with clear warnings about what (if anything) gets lost |
| `validate` | Check your dataset for common problems — duplicate IDs, missing references, invalid bounding boxes |
| `stats` | Show rich dataset statistics in text, JSON, or HTML |
| `diff` | Compare two datasets semantically (summary or detailed output) |
| `sample` | Create subset datasets (random or stratified), with optional category filtering and JSON reports |
| `list-formats` | Show which formats are supported and their read/write/lossiness capabilities, including JSON discovery output |

## Supported formats

| Format | Extension / Layout | Description | Lossiness |
|--------|--------------------|-------------|-----------|
| `ir-json` | `.json` | Panlabel's own intermediate representation | Lossless |
| `coco` | `.json` | COCO object detection format | Conditional |
| `cvat` | `.xml` / `annotations.xml` export | CVAT for images XML annotation export | Lossy |
| `label-studio` | `.json` | Label Studio task export JSON (`rectanglelabels`) | Lossy |
| `tfod` | `.csv` | TensorFlow Object Detection | Lossy |
| `yolo` | `images/ + labels/` directory | YOLO `.txt` labels (flat or split-aware, optional confidence) | Lossy |
| `voc` | `Annotations/ + JPEGImages/` directory | Pascal VOC XML | Lossy |
| `hf` | `metadata.jsonl` / `metadata.parquet` directory | Hugging Face ImageFolder metadata | Lossy |
| `sagemaker` | `.manifest` / `.jsonl` file | AWS SageMaker Ground Truth object-detection manifest | Lossy |
| `labelme` | `.json` file or `annotations/` directory | LabelMe per-image JSON annotations | Lossy |
| `create-ml` | `.json` | Apple CreateML annotation format | Lossy |
| `kitti` | `label_2/ + image_2/` directory | KITTI object detection labels | Lossy |
| `via` | `.json` | VGG Image Annotator (VIA) JSON | Lossy |
| `retinanet` | `.csv` | keras-retinanet CSV format | Lossy |
| `openimages` | `.csv` | Google OpenImages CSV annotation format | Lossy |
| `kaggle-wheat` | `.csv` | Kaggle Global Wheat Detection CSV | Lossy |
| `automl-vision` | `.csv` | Google Cloud AutoML Vision CSV | Lossy |
| `udacity` | `.csv` | Udacity Self-Driving Car Dataset CSV | Lossy |
| `superannotate` | `.json` file or `annotations/` directory | SuperAnnotate JSON export | Lossy |
| `supervisely` | `.json` file or `ann/` / `meta.json` project directory | Supervisely JSON project / dataset | Lossy |

Run `panlabel list-formats` for the full details, or `panlabel list-formats --output json` for machine-readable format discovery.

`list-formats` shows canonical names (for example `label-studio`), while commands also accept aliases (for example `ls`, `label-studio-json`). Across commands, `--output-format` is the consistent way to request JSON reports; `convert` and `sample` also keep `--report` as an alias. JSON is pretty-printed on a terminal and compact when piped or captured, which makes it friendlier for scripts and agents. `stats` also adapts its text renderer: rich/Unicode on a terminal, plain text layout when piped.

### More convert examples

```sh
# COCO to IR JSON (lossless — no data lost)
panlabel convert -f coco -t ir-json -i input.json -o output.json

# IR JSON to TFOD (lossy — requires explicit opt-in)
panlabel convert -f ir-json -t tfod -i input.json -o output.csv --allow-lossy

# Auto-detect input format from file extension/content or directory layout
panlabel convert --from auto -t coco -i input.csv -o output.json

# Request a machine-readable conversion report
panlabel convert --from auto -t coco -i input.csv -o output.json --output-format json

# Preview a conversion without touching the output path
panlabel convert --from auto -t coco -i input.csv -o output.json --dry-run

# Convert a remote Hugging Face dataset repo to COCO JSON
# (requires --features hf when building from source)
panlabel convert -f hf -t coco --hf-repo rishitdagli/cppe-5 --split train -o coco_output.json

# Convert a zip-style HF dataset repo split to IR JSON (auto-detects extracted payload)
panlabel convert -f hf -t ir-json --hf-repo keremberke/football-object-detection --split train -o football.ir.json
```

Dry runs still do the real thinking work — format detection, validation, sampling/conversion analysis, and lossiness checks — but they skip the final filesystem write. That means they are good for “what would happen?” checks, but they do **not** prove that the output path is writable.

### Getting help

```sh
panlabel --help              # See all commands
panlabel convert --help      # Help for a specific command
panlabel -V                  # Show version
```

## Documentation

Want to go deeper? The full docs are readable right here on GitHub:

- [Documentation home](https://github.com/strickvl/panlabel/blob/HEAD/docs/README.md) — start here
- [CLI reference](https://github.com/strickvl/panlabel/blob/HEAD/docs/cli.md) — every flag and option
- [Format reference](https://github.com/strickvl/panlabel/blob/HEAD/docs/formats.md) — how each format works
- [Tasks and use cases](https://github.com/strickvl/panlabel/blob/HEAD/docs/tasks.md) — what's supported today
- [Conversion and lossiness](https://github.com/strickvl/panlabel/blob/HEAD/docs/conversion.md) — understanding what gets lost
- [Contributing](https://github.com/strickvl/panlabel/blob/HEAD/CONTRIBUTING.md) — we'd love your help
- [Roadmap](https://github.com/strickvl/panlabel/blob/HEAD/ROADMAP.md) — what's coming next

## Contributing

Contributions are welcome! Whether it's a bug report, a new format adapter, or
a documentation fix — we appreciate the help. For major changes, please
[open an issue](https://github.com/strickvl/panlabel/issues) first so we can
discuss the approach.

See the [contributing guide](CONTRIBUTING.md) for details on the codebase
structure and how to make changes.

## License

MIT — see [LICENSE](LICENSE) for details.
