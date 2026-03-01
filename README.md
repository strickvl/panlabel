# Panlabel

[![CI](https://github.com/strickvl/panlabel/actions/workflows/ci.yml/badge.svg)](https://github.com/strickvl/panlabel/actions/workflows/ci.yml)
![Crates.io Version](https://img.shields.io/crates/v/panlabel)
![GitHub License](https://img.shields.io/github/license/strickvl/panlabel)
![GitHub Repo stars](https://img.shields.io/github/stars/strickvl/panlabel)
![Crates.io Total Downloads](https://img.shields.io/crates/d/panlabel)
![Crates.io Size](https://img.shields.io/crates/size/panlabel)

## The universal annotation converter

If you've ever written a one-off Python script to wrangle COCO annotations into
YOLO format (or vice versa), panlabel is here to save you the trouble. It's a
fast, single-binary CLI that converts between common object detection annotation
formats — with built-in validation, clear lossiness warnings, and no Python
dependencies to manage.

Panlabel is also available as a Rust library if you want to integrate format
conversion into your own tools.

> **Note**: Panlabel is in active development (v0.1.x). The CLI and library APIs
> may change between versions, so pin to a specific version if you're using it in
> production.

## Installation

Install `panlabel` from [crates.io](https://crates.io/crates/panlabel):

```sh
cargo install panlabel
```

Want to use panlabel as a library in your own Rust project? Just run
`cargo add panlabel`.

## Quick start

```sh
# Convert COCO annotations to YOLO (auto-detects the input format)
panlabel convert --from auto --to yolo -i annotations.json -o ./yolo_out --allow-lossy

# Convert a YOLO dataset to COCO JSON
panlabel convert -f yolo -t coco -i ./my_dataset -o coco_output.json

# Convert a Pascal VOC dataset to COCO JSON
panlabel convert -f voc -t coco -i ./voc_dataset -o coco_output.json

# Convert Label Studio JSON to COCO JSON
panlabel convert -f label-studio -t coco -i export.json -o coco_output.json

# Check a dataset for problems before training
panlabel validate --format coco annotations.json

# Get a quick overview of what's in a dataset
panlabel inspect --format coco annotations.json
```

## What can panlabel do?

| Command | What it does |
|---------|-------------|
| `convert` | Convert between annotation formats, with clear warnings about what (if anything) gets lost |
| `validate` | Check your dataset for common problems — duplicate IDs, missing references, invalid bounding boxes |
| `inspect` | Show dataset statistics: image/annotation counts, label histogram, bounding box quality metrics |
| `list-formats` | Show which formats are supported and their read/write/lossiness capabilities |

## Supported formats

| Format | Extension / Layout | Description | Lossiness |
|--------|--------------------|-------------|-----------|
| `ir-json` | `.json` | Panlabel's own intermediate representation | Lossless |
| `coco` | `.json` | COCO object detection format | Conditional |
| `label-studio` | `.json` | Label Studio task export JSON (`rectanglelabels`) | Lossy |
| `tfod` | `.csv` | TensorFlow Object Detection | Lossy |
| `yolo` | `images/ + labels/` directory | Ultralytics YOLO `.txt` labels | Lossy |
| `voc` | `Annotations/ + JPEGImages/` directory | Pascal VOC XML | Lossy |

Run `panlabel list-formats` for the full details.

`list-formats` shows canonical names (for example `label-studio`), while commands also accept aliases (for example `ls`, `label-studio-json`).

### More convert examples

```sh
# COCO to IR JSON (lossless — no data lost)
panlabel convert -f coco -t ir-json -i input.json -o output.json

# IR JSON to TFOD (lossy — requires explicit opt-in)
panlabel convert -f ir-json -t tfod -i input.json -o output.csv --allow-lossy

# Auto-detect input format from file extension/content or directory layout
panlabel convert --from auto -t coco -i input.csv -o output.json
```

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

## Benchmarks

```sh
cargo bench    # Run Criterion benchmarks (COCO parsing, TFOD writing)
```

## Generating synthetic test data

Need test data? The included generator creates realistic COCO and TFOD datasets:

```sh
pip install numpy  # or: uv pip install numpy
python scripts/dataset_generator.py --num_images 1000 --annotations_per_image 10 --output_dir ./assets
```

Tweak the numbers to get datasets of whatever size you need.

## Contributing

Contributions are welcome! Whether it's a bug report, a new format adapter, or
a documentation fix — we appreciate the help. For major changes, please
[open an issue](https://github.com/strickvl/panlabel/issues) first so we can
discuss the approach.

See the [contributing guide](CONTRIBUTING.md) for details on the codebase
structure and how to make changes.

## License

MIT — see [LICENSE](LICENSE) for details.
