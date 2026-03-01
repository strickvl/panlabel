//! Hugging Face ImageFolder and hub-style Parquet reader.
//!
//! This module is feature-gated because Parquet decoding pulls in heavier
//! dependencies than the JSONL path.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::{Map, Value};

use crate::error::PanlabelError;

use super::io_hf_imagefolder::{
    dataset_from_rows, parse_jsonl_row, read_hf_imagefolder_with_options, HfReadOptions,
};
use super::Dataset;

/// Read HF ImageFolder metadata from Parquet.
pub fn read_hf_parquet(path: &Path) -> Result<Dataset, PanlabelError> {
    read_hf_parquet_with_options(path, &HfReadOptions::default())
}

/// Read HF ImageFolder metadata from Parquet with explicit options.
///
/// Supported layouts:
/// - `metadata.parquet` at root or split dir
/// - split parquet shards (for example `data/train-00000-of-00001.parquet`)
pub fn read_hf_parquet_with_options(
    path: &Path,
    options: &HfReadOptions,
) -> Result<Dataset, PanlabelError> {
    if has_any_jsonl(path, options.split.as_deref())? {
        eprintln!(
            "Note: found metadata.jsonl alongside metadata.parquet; preferring metadata.jsonl."
        );
        return read_hf_imagefolder_with_options(path, options);
    }

    let parquet_files = discover_parquet_files(path, options.split.as_deref())?;
    let mut rows = Vec::new();

    for parquet_path in &parquet_files {
        let split_dir = parquet_path.parent().unwrap_or(path);
        let file = fs::File::open(parquet_path).map_err(PanlabelError::Io)?;
        let reader =
            SerializedFileReader::new(file).map_err(|source| PanlabelError::HfParquetParse {
                path: parquet_path.clone(),
                message: source.to_string(),
            })?;

        let row_iter =
            reader
                .get_row_iter(None)
                .map_err(|source| PanlabelError::HfParquetParse {
                    path: parquet_path.clone(),
                    message: source.to_string(),
                })?;

        for (idx, row_res) in row_iter.enumerate() {
            let row = row_res.map_err(|source| PanlabelError::HfParquetParse {
                path: parquet_path.clone(),
                message: source.to_string(),
            })?;

            // Reuse JSONL normalization/parsing by converting each Parquet row to JSON,
            // then synthesizing missing fields expected by the JSONL parser.
            let row_json = row.to_json_value();
            let normalized = normalize_parquet_row(parquet_path, idx + 1, row_json, options)?;
            let parsed = parse_jsonl_row(parquet_path, idx + 1, &normalized, split_dir, options)
                .map_err(|error| map_row_parse_error(error, parquet_path, idx + 1))?;
            rows.push(parsed);
        }
    }

    dataset_from_rows(rows, options)
}

fn map_row_parse_error(error: PanlabelError, path: &Path, row_index: usize) -> PanlabelError {
    match error {
        PanlabelError::HfJsonlParse { message, .. } => PanlabelError::HfParquetParse {
            path: path.to_path_buf(),
            message: format!("row {row_index}: {message}"),
        },
        other => other,
    }
}

fn normalize_parquet_row(
    parquet_path: &Path,
    row_index: usize,
    value: Value,
    options: &HfReadOptions,
) -> Result<Value, PanlabelError> {
    let mut row_obj = value
        .as_object()
        .cloned()
        .ok_or_else(|| PanlabelError::HfParquetParse {
            path: parquet_path.to_path_buf(),
            message: format!("row {row_index}: expected a JSON object row"),
        })?;

    if !row_obj.contains_key("file_name") {
        let file_name =
            extract_parquet_file_name(&row_obj).ok_or_else(|| PanlabelError::HfParquetParse {
                path: parquet_path.to_path_buf(),
                message: format!(
                    "row {row_index}: missing file name (expected 'file_name' or 'image.path')"
                ),
            })?;
        row_obj.insert("file_name".to_string(), Value::String(file_name));
    }

    if !row_obj.contains_key("objects") && !row_obj.contains_key("faces") {
        if let Some(synthesized) = synthesize_objects_from_bboxes(&row_obj) {
            let key = options
                .objects_column
                .clone()
                .unwrap_or_else(|| "objects".to_string());
            row_obj.insert(key, Value::Object(synthesized));
        }
    }

    Ok(Value::Object(row_obj))
}

fn extract_parquet_file_name(row_obj: &Map<String, Value>) -> Option<String> {
    if let Some(file_name) = row_obj.get("file_name").and_then(Value::as_str) {
        if !file_name.trim().is_empty() {
            return Some(file_name.to_string());
        }
    }

    if let Some(path) = row_obj
        .get("image")
        .and_then(Value::as_object)
        .and_then(|image| image.get("path"))
        .and_then(Value::as_str)
    {
        if !path.trim().is_empty() {
            return Some(path.to_string());
        }
    }

    if let Some(id) = row_obj.get("id").and_then(Value::as_str) {
        if !id.trim().is_empty() {
            return Some(id.to_string());
        }
    }

    if let Some(image_id) = row_obj.get("image_id") {
        if let Some(number) = image_id.as_i64() {
            return Some(format!("image_{number}"));
        }
        if let Some(text) = image_id.as_str() {
            return Some(format!("image_{text}"));
        }
    }

    None
}

fn synthesize_objects_from_bboxes(row_obj: &Map<String, Value>) -> Option<Map<String, Value>> {
    let bboxes = row_obj.get("bboxes")?.as_array()?.clone();

    let mut objects = Map::new();
    let categories = vec![Value::from(0); bboxes.len()];
    objects.insert("bbox".to_string(), Value::Array(bboxes));
    objects.insert("category".to_string(), Value::Array(categories));
    Some(objects)
}

fn has_any_jsonl(path: &Path, split: Option<&str>) -> Result<bool, PanlabelError> {
    if !path.is_dir() {
        return Ok(false);
    }

    if path.join("metadata.jsonl").is_file() {
        return Ok(true);
    }

    if let Some(split_name) = split {
        let normalized = normalize_split_name(split_name).unwrap_or(split_name);
        return Ok(path.join(normalized).join("metadata.jsonl").is_file());
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

fn discover_parquet_files(root: &Path, split: Option<&str>) -> Result<Vec<PathBuf>, PanlabelError> {
    if !root.is_dir() {
        return Err(PanlabelError::HfLayoutInvalid {
            path: root.to_path_buf(),
            message: "expected a directory containing Parquet metadata or split parquet shards"
                .to_string(),
        });
    }

    let mut metadata_files = collect_metadata_parquet_files(root, split)?;
    if !metadata_files.is_empty() {
        metadata_files.sort();
        return Ok(metadata_files);
    }

    let mut shard_files = collect_parquet_shard_files(root, split)?;
    if shard_files.is_empty() {
        return Err(PanlabelError::HfLayoutInvalid {
            path: root.to_path_buf(),
            message:
                "missing readable parquet layout (expected metadata.parquet or split *.parquet shards)"
                    .to_string(),
        });
    }

    shard_files.sort();
    Ok(shard_files)
}

fn collect_metadata_parquet_files(
    root: &Path,
    split: Option<&str>,
) -> Result<Vec<PathBuf>, PanlabelError> {
    let root_metadata = root.join("metadata.parquet");

    if let Some(split_name) = split {
        let normalized = normalize_split_name(split_name).unwrap_or(split_name);
        let split_metadata = root.join(normalized).join("metadata.parquet");
        if split_metadata.is_file() {
            return Ok(vec![split_metadata]);
        }

        if root_metadata.is_file() {
            return Err(PanlabelError::HfLayoutInvalid {
                path: root.to_path_buf(),
                message: format!(
                    "split '{}' was requested, but this dataset has root-level metadata.parquet (no split subdirectories)",
                    split_name
                ),
            });
        }

        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    if root_metadata.is_file() {
        files.push(root_metadata);
    }

    let entries = fs::read_dir(root).map_err(PanlabelError::Io)?;
    for entry in entries {
        let entry = entry.map_err(PanlabelError::Io)?;
        let entry_path = entry.path();
        if entry_path.is_dir() && entry_path.join("metadata.parquet").is_file() {
            files.push(entry_path.join("metadata.parquet"));
        }
    }

    Ok(files)
}

fn collect_parquet_shard_files(
    root: &Path,
    split: Option<&str>,
) -> Result<Vec<PathBuf>, PanlabelError> {
    let mut files = Vec::new();
    let mut inferred_splits = BTreeSet::new();
    let target_split = split
        .and_then(normalize_split_name)
        .map(str::to_string)
        .or_else(|| split.map(str::to_string));

    for entry in walkdir::WalkDir::new(root).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::HfLayoutInvalid {
            path: root.to_path_buf(),
            message: format!("failed while scanning parquet shards: {source}"),
        })?;
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let is_parquet = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("parquet"))
            .unwrap_or(false);
        if !is_parquet {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.eq_ignore_ascii_case("metadata.parquet") {
            continue;
        }

        let inferred_split = infer_split_from_parquet_path(path);
        if let Some(inferred) = inferred_split.as_ref() {
            inferred_splits.insert(inferred.clone());
        }

        if let Some(target) = target_split.as_deref() {
            if inferred_split.as_deref() != Some(target) {
                continue;
            }
        }

        files.push(path.to_path_buf());
    }

    if files.is_empty() && target_split.is_some() {
        let requested = target_split.as_deref().unwrap_or_default();
        let available = if inferred_splits.is_empty() {
            "none".to_string()
        } else {
            inferred_splits.into_iter().collect::<Vec<_>>().join(", ")
        };

        return Err(PanlabelError::HfLayoutInvalid {
            path: root.to_path_buf(),
            message: format!(
                "no parquet shards found for split '{requested}'. Available inferred splits: {available}"
            ),
        });
    }

    Ok(files)
}

fn infer_split_from_parquet_path(path: &Path) -> Option<String> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase());

    if let Some(file_name) = file_name {
        if let Some((prefix, _)) = file_name.split_once('-') {
            if let Some(normalized) = normalize_split_name(prefix) {
                return Some(normalized.to_string());
            }
        }

        if let Some(stem) = file_name.strip_suffix(".parquet") {
            if let Some(normalized) = normalize_split_name(stem) {
                return Some(normalized.to_string());
            }
        }
    }

    for component in path.components().rev() {
        let Some(name) = component.as_os_str().to_str() else {
            continue;
        };
        if let Some(normalized) = normalize_split_name(name) {
            return Some(normalized.to_string());
        }
    }

    None
}

fn normalize_split_name(value: &str) -> Option<&str> {
    match value.to_ascii_lowercase().as_str() {
        "train" => Some("train"),
        "test" => Some("test"),
        "validation" | "valid" | "val" => Some("validation"),
        "dev" => Some("dev"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_split_from_filename_prefix() {
        let path = Path::new("data/train-00000-of-00003.parquet");
        assert_eq!(
            infer_split_from_parquet_path(path).as_deref(),
            Some("train")
        );

        let path = Path::new("data/validation-00000-of-00001.parquet");
        assert_eq!(
            infer_split_from_parquet_path(path).as_deref(),
            Some("validation")
        );
    }

    #[test]
    fn normalize_row_derives_file_name_from_image_path() {
        let row = serde_json::json!({
            "image": {"path": "img1.jpg"},
            "width": 100,
            "height": 80,
            "objects": {
                "bbox": [[1.0, 2.0, 3.0, 4.0]],
                "category": [0]
            }
        });

        let normalized = normalize_parquet_row(
            Path::new("data/train-00000-of-00001.parquet"),
            1,
            row,
            &HfReadOptions::default(),
        )
        .expect("normalize row");

        assert_eq!(
            normalized
                .get("file_name")
                .and_then(Value::as_str)
                .expect("file_name"),
            "img1.jpg"
        );
    }

    #[test]
    fn normalize_row_synthesizes_objects_from_bboxes() {
        let row = serde_json::json!({
            "image": {"path": "img1.jpg"},
            "width": 100,
            "height": 80,
            "bboxes": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        });

        let normalized = normalize_parquet_row(
            Path::new("data/train-00000-of-00001.parquet"),
            1,
            row,
            &HfReadOptions::default(),
        )
        .expect("normalize row");

        let objects = normalized
            .get("objects")
            .and_then(Value::as_object)
            .expect("objects");
        assert_eq!(
            objects
                .get("bbox")
                .and_then(Value::as_array)
                .map(|v| v.len()),
            Some(2)
        );
        assert_eq!(
            objects
                .get("category")
                .and_then(Value::as_array)
                .map(|v| v.len()),
            Some(2)
        );
    }

    #[test]
    fn split_aware_jsonl_probe_ignores_other_split_jsonl() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(temp.path().join("validation")).expect("create validation dir");
        std::fs::write(temp.path().join("validation/metadata.jsonl"), "{}\n")
            .expect("write metadata");

        assert!(has_any_jsonl(temp.path(), None).expect("scan all"));
        assert!(!has_any_jsonl(temp.path(), Some("train")).expect("scan train"));
    }

    #[test]
    fn split_with_root_metadata_parquet_is_invalid() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("metadata.parquet"), b"marker").expect("write marker");

        let err = collect_metadata_parquet_files(temp.path(), Some("train"))
            .expect_err("should reject split against root metadata");
        match err {
            PanlabelError::HfLayoutInvalid { message, .. } => {
                assert!(message.contains("root-level metadata.parquet"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
