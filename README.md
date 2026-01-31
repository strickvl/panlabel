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

## Usage

You can use the following commands to get more information about `panlabel`:

- `panlabel -V` or `panlabel --version`: Displays the current version of panlabel.
- `panlabel -h` or `panlabel --help`: Shows the full CLI help, including available commands and options.

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
