//! Hugging Face ImageFolder `metadata.parquet` reader.
//!
//! This module is feature-gated because Parquet decoding pulls in heavier
//! dependencies than the JSONL path.

use std::fs;
use std::path::{Path, PathBuf};

use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::error::PanlabelError;

use super::io_hf_imagefolder::{
    dataset_from_rows, parse_jsonl_row, read_hf_imagefolder_with_options, HfReadOptions,
};
use super::Dataset;

/// Read HF ImageFolder metadata from `metadata.parquet`.
pub fn read_hf_parquet(path: &Path) -> Result<Dataset, PanlabelError> {
    read_hf_parquet_with_options(path, &HfReadOptions::default())
}

/// Read HF ImageFolder metadata from `metadata.parquet` with explicit options.
pub fn read_hf_parquet_with_options(
    path: &Path,
    options: &HfReadOptions,
) -> Result<Dataset, PanlabelError> {
    if has_any_jsonl(path)? {
        eprintln!(
            "Note: found metadata.jsonl alongside metadata.parquet; preferring metadata.jsonl."
        );
        return read_hf_imagefolder_with_options(path, options);
    }

    let split_dirs = discover_parquet_split_dirs(path, options.split.as_deref())?;
    let mut rows = Vec::new();

    for split_dir in &split_dirs {
        let metadata_path = split_dir.join("metadata.parquet");
        let file = fs::File::open(&metadata_path).map_err(PanlabelError::Io)?;
        let reader =
            SerializedFileReader::new(file).map_err(|source| PanlabelError::HfParquetParse {
                path: metadata_path.clone(),
                message: source.to_string(),
            })?;

        let row_iter =
            reader
                .get_row_iter(None)
                .map_err(|source| PanlabelError::HfParquetParse {
                    path: metadata_path.clone(),
                    message: source.to_string(),
                })?;

        for (idx, row_res) in row_iter.enumerate() {
            let row = row_res.map_err(|source| PanlabelError::HfParquetParse {
                path: metadata_path.clone(),
                message: source.to_string(),
            })?;

            // Reuse the JSONL normalization and parsing logic by converting
            // each parquet row to a JSON value.
            let row_json = row.to_json_value();
            let parsed = parse_jsonl_row(&metadata_path, idx + 1, &row_json, split_dir, options)?;
            rows.push(parsed);
        }
    }

    dataset_from_rows(rows, options)
}

fn has_any_jsonl(path: &Path) -> Result<bool, PanlabelError> {
    if !path.is_dir() {
        return Ok(false);
    }

    if path.join("metadata.jsonl").is_file() {
        return Ok(true);
    }

    let entries = fs::read_dir(path).map_err(PanlabelError::Io)?;
    for entry in entries {
        let entry = entry.map_err(PanlabelError::Io)?;
        let entry_path = entry.path();
        if entry_path.is_dir() && entry_path.join("metadata.jsonl").is_file() {
            return Ok(true);
        }
    }

    Ok(false)
}

fn discover_parquet_split_dirs(
    root: &Path,
    split: Option<&str>,
) -> Result<Vec<PathBuf>, PanlabelError> {
    if !root.is_dir() {
        return Err(PanlabelError::HfLayoutInvalid {
            path: root.to_path_buf(),
            message: "expected a directory containing metadata.parquet or split subdirectories"
                .to_string(),
        });
    }

    if let Some(split_name) = split {
        let split_dir = root.join(split_name);
        if split_dir.join("metadata.parquet").is_file() {
            return Ok(vec![split_dir]);
        }

        return Err(PanlabelError::HfLayoutInvalid {
            path: root.to_path_buf(),
            message: format!("split '{}' not found with metadata.parquet", split_name),
        });
    }

    let root_parquet = root.join("metadata.parquet");
    if root_parquet.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }

    let mut split_dirs = Vec::new();
    let entries = fs::read_dir(root).map_err(PanlabelError::Io)?;
    for entry in entries {
        let entry = entry.map_err(PanlabelError::Io)?;
        let entry_path = entry.path();
        if entry_path.is_dir() && entry_path.join("metadata.parquet").is_file() {
            split_dirs.push(entry_path);
        }
    }

    split_dirs.sort();

    if split_dirs.is_empty() {
        return Err(PanlabelError::HfLayoutInvalid {
            path: root.to_path_buf(),
            message:
                "missing metadata.parquet (expected at dataset root or inside split subdirectories)"
                    .to_string(),
        });
    }

    Ok(split_dirs)
}
