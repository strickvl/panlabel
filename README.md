# Panlabel

[![CI](https://github.com/strickvl/panlabel/actions/workflows/ci.yml/badge.svg)](https://github.com/strickvl/panlabel/actions/workflows/ci.yml)
![Crates.io Version](https://img.shields.io/crates/v/panlabel)
![GitHub License](https://img.shields.io/github/license/strickvl/panlabel)
![GitHub Repo stars](https://img.shields.io/github/stars/strickvl/panlabel)
![Crates.io Total Downloads](https://img.shields.io/crates/d/panlabel)
![Crates.io Size](https://img.shields.io/crates/size/panlabel)

## The universal annotation converter

Panlabel is a Rust library for converting between different annotation formats,
and a command-line tool that uses this library.

> ⚠️ **Warning**: This library is in active development and breaking changes
> should be expected between versions. Please pin to specific versions in
> production.

## Installation

You can install `panlabel` from [crates.io](https://crates.io/crates/panlabel) using cargo:

```sh
cargo install panlabel
```

If you wish to use the library in your own project, you can add it to your
`Cargo.toml` with `cargo add panlabel`.

## Supported Formats

| Format | Extension / Layout | Description | Lossiness |
|--------|--------------------|-------------|-----------|
| `ir-json` | `.json` | Panlabel's intermediate representation | Lossless |
| `coco` | `.json` | COCO object detection format | Conditional |
| `tfod` | `.csv` | TensorFlow Object Detection | Lossy |
| `yolo` | `images/ + labels/` directory | Ultralytics YOLO `.txt` labels | Lossy |

Run `panlabel list-formats` for details on format capabilities and lossiness.

## Usage

### Quick Start

```sh
# Convert COCO to TFOD (auto-detects input format)
panlabel convert --from auto --to tfod -i annotations.json -o annotations.csv --allow-lossy

# Convert IR JSON to YOLO directory (lossy)
panlabel convert --from ir-json --to yolo -i annotations.ir.json -o ./yolo_out --allow-lossy

# Validate a dataset (file or directory path)
panlabel validate --format coco annotations.json

# Inspect dataset statistics
panlabel inspect --format coco annotations.json

# List supported formats
panlabel list-formats
```

### Commands

| Command | Description |
|---------|-------------|
| `convert` | Convert between annotation formats |
| `validate` | Validate a dataset for errors and warnings |
| `inspect` | Display dataset statistics (counts, label histogram, bbox stats) |
| `list-formats` | Show supported formats and their capabilities |

### Convert Examples

```sh
# COCO to IR JSON (lossless)
panlabel convert -f coco -t ir-json -i input.json -o output.json

# IR JSON to TFOD (lossy - requires --allow-lossy)
panlabel convert -f ir-json -t tfod -i input.json -o output.csv --allow-lossy

# Auto-detect input format (file or directory path)
panlabel convert --from auto -t coco -i input.csv -o output.json

# Convert YOLO directory to COCO JSON
panlabel convert -f yolo -t coco -i ./dataset_root -o output.json
```

### Help

- `panlabel --help`: Shows available commands
- `panlabel <command> --help`: Shows help for a specific command
- `panlabel -V`: Displays version

## Benchmarks

Run the Criterion benchmarks with:

```sh
cargo bench
```

This benchmarks COCO JSON parsing and TFOD CSV writing performance.

## Synthetic Data for Benchmarking and Testing

To generate synthetic data, first install `numpy` into a fresh Python virtual environment and then run the following command:

```sh
python scripts/dataset_generator.py --num_images 1000 --annotations_per_image 10 --output_dir ./assets
```

Feel free to tweak the parameters to generate more or less data.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
