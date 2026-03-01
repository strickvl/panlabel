//! Hugging Face ImageFolder (`metadata.jsonl`) reader and writer.
//!
//! This adapter is intentionally local-file only. Remote Hub orchestration lives
//! in `crate::hf` and resolves to a local directory before calling this module.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{BBoxXYXY, CategoryId, ImageId};
use crate::error::PanlabelError;

/// Bounding-box convention used by HF metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HfBboxFormat {
    /// `[x, y, width, height]` (COCO-style).
    #[default]
    Xywh,
    /// `[x1, y1, x2, y2]`.
    Xyxy,
}

impl HfBboxFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            HfBboxFormat::Xywh => "xywh",
            HfBboxFormat::Xyxy => "xyxy",
        }
    }
}

/// Reader options for HF ImageFolder metadata.
#[derive(Clone, Debug, Default)]
pub struct HfReadOptions {
    /// Source bbox format.
    pub bbox_format: HfBboxFormat,
    /// Optional override for the object container key.
    pub objects_column: Option<String>,
    /// Optional split to read when the dataset has split subdirectories.
    pub split: Option<String>,
    /// Optional category-ID to category-name mapping.
    pub category_map: BTreeMap<i64, String>,
    /// Optional dataset-level attributes to inject into IR provenance.
    pub provenance: BTreeMap<String, String>,
}

/// Writer options for HF ImageFolder metadata.
#[derive(Clone, Debug, Default)]
pub struct HfWriteOptions {
    /// Target bbox format.
    pub bbox_format: HfBboxFormat,
}

#[derive(Debug)]
pub(crate) struct ParsedRow {
    file_name: String,
    width: Option<u32>,
    height: Option<u32>,
    anns: Vec<ParsedAnn>,
}

#[derive(Debug)]
pub(crate) struct ParsedAnn {
    bbox: [f64; 4],
    category: ParsedCategory,
}

#[derive(Debug)]
pub(crate) enum ParsedCategory {
    Id(i64),
    Name(String),
}

/// Read an HF ImageFolder dataset from a local directory.
pub fn read_hf_imagefolder(path: &Path) -> Result<Dataset, PanlabelError> {
    read_hf_imagefolder_with_options(path, &HfReadOptions::default())
}

/// Read an HF ImageFolder dataset with explicit options.
pub fn read_hf_imagefolder_with_options(
    path: &Path,
    options: &HfReadOptions,
) -> Result<Dataset, PanlabelError> {
    let split_dirs = discover_jsonl_split_dirs(path, options.split.as_deref())?;

    let mut rows = Vec::new();
    for split_dir in &split_dirs {
        rows.extend(read_split_rows(split_dir, options)?);
    }

    dataset_from_rows(rows, options)
}

/// Write an IR dataset as HF ImageFolder `metadata.jsonl`.
pub fn write_hf_imagefolder(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    write_hf_imagefolder_with_options(path, dataset, &HfWriteOptions::default())
}

/// Write an IR dataset as HF ImageFolder `metadata.jsonl` with explicit options.
pub fn write_hf_imagefolder_with_options(
    path: &Path,
    dataset: &Dataset,
    options: &HfWriteOptions,
) -> Result<(), PanlabelError> {
    fs::create_dir_all(path).map_err(PanlabelError::Io)?;
    let out_path = path.join("metadata.jsonl");

    let image_lookup: BTreeMap<ImageId, &Image> = dataset
        .images
        .iter()
        .map(|image| (image.id, image))
        .collect();
    let category_lookup: BTreeMap<CategoryId, &Category> = dataset
        .categories
        .iter()
        .map(|category| (category.id, category))
        .collect();

    let mut anns_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        if !image_lookup.contains_key(&ann.image_id) {
            return Err(PanlabelError::HfWriteError {
                path: out_path.clone(),
                message: format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            });
        }
        if !category_lookup.contains_key(&ann.category_id) {
            return Err(PanlabelError::HfWriteError {
                path: out_path.clone(),
                message: format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            });
        }
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }

    let mut images_sorted: Vec<&Image> = dataset.images.iter().collect();
    images_sorted.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    let file = fs::File::create(&out_path).map_err(PanlabelError::Io)?;
    let mut writer = std::io::BufWriter::new(file);

    for image in images_sorted {
        let mut anns = anns_by_image.remove(&image.id).unwrap_or_default();
        anns.sort_by_key(|ann| ann.id);

        let mut bbox_values = Vec::with_capacity(anns.len());
        let mut category_values = Vec::with_capacity(anns.len());

        for ann in anns {
            let category = category_lookup.get(&ann.category_id).ok_or_else(|| {
                PanlabelError::HfWriteError {
                    path: out_path.clone(),
                    message: format!(
                        "annotation {} references missing category {}",
                        ann.id.as_u64(),
                        ann.category_id.as_u64()
                    ),
                }
            })?;

            let bbox = match options.bbox_format {
                HfBboxFormat::Xywh => {
                    let (x, y, w, h) = ann.bbox.to_xywh();
                    vec![x, y, w, h]
                }
                HfBboxFormat::Xyxy => vec![
                    ann.bbox.xmin(),
                    ann.bbox.ymin(),
                    ann.bbox.xmax(),
                    ann.bbox.ymax(),
                ],
            };
            bbox_values.push(Value::Array(bbox.into_iter().map(Value::from).collect()));
            category_values.push(Value::String(category.name.clone()));
        }

        let mut objects = Map::new();
        objects.insert("bbox".to_string(), Value::Array(bbox_values));
        objects.insert("categories".to_string(), Value::Array(category_values));

        let mut row = Map::new();
        row.insert(
            "file_name".to_string(),
            Value::String(image.file_name.clone()),
        );
        row.insert("width".to_string(), Value::from(image.width));
        row.insert("height".to_string(), Value::from(image.height));
        row.insert("objects".to_string(), Value::Object(objects));

        serde_json::to_writer(&mut writer, &Value::Object(row)).map_err(|source| {
            PanlabelError::HfWriteError {
                path: out_path.clone(),
                message: source.to_string(),
            }
        })?;
        writeln!(&mut writer).map_err(PanlabelError::Io)?;
    }

    writer.flush().map_err(PanlabelError::Io)?;
    Ok(())
}

fn discover_jsonl_split_dirs(
    root: &Path,
    split: Option<&str>,
) -> Result<Vec<PathBuf>, PanlabelError> {
    if !root.is_dir() {
        return Err(PanlabelError::HfLayoutInvalid {
            path: root.to_path_buf(),
            message: "expected a directory containing metadata.jsonl or split subdirectories"
                .to_string(),
        });
    }

    let root_jsonl = root.join("metadata.jsonl");
    let root_parquet = root.join("metadata.parquet");

    let mut split_dirs = Vec::new();
    let mut split_names = Vec::new();
    let mut parquet_only_splits = Vec::new();

    let entries = fs::read_dir(root).map_err(PanlabelError::Io)?;
    for entry in entries {
        let entry = entry.map_err(PanlabelError::Io)?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let jsonl = path.join("metadata.jsonl");
        let parquet = path.join("metadata.parquet");

        if jsonl.is_file() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                split_names.push(name.to_string());
            }
            split_dirs.push(path);
        } else if parquet.is_file() {
            parquet_only_splits.push(path);
        }
    }

    split_dirs.sort();

    if let Some(split_name) = split {
        if root_jsonl.is_file() {
            return Err(PanlabelError::HfLayoutInvalid {
                path: root.to_path_buf(),
                message: format!(
                    "split '{}' was requested, but this dataset has root-level metadata.jsonl (no split subdirectories)",
                    split_name
                ),
            });
        }

        let selected = root.join(split_name);
        if selected.join("metadata.jsonl").is_file() {
            return Ok(vec![selected]);
        }

        let available = if split_names.is_empty() {
            "none".to_string()
        } else {
            split_names.join(", ")
        };
        return Err(PanlabelError::HfLayoutInvalid {
            path: root.to_path_buf(),
            message: format!(
                "split '{}' not found. Available splits with metadata.jsonl: {}",
                split_name, available
            ),
        });
    }

    if root_jsonl.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }

    if !split_dirs.is_empty() {
        return Ok(split_dirs);
    }

    if root_parquet.is_file() || !parquet_only_splits.is_empty() {
        return Err(PanlabelError::HfLayoutInvalid {
            path: root.to_path_buf(),
            message:
                "found metadata.parquet but no metadata.jsonl. Use a build with feature 'hf-parquet' enabled"
                    .to_string(),
        });
    }

    Err(PanlabelError::HfLayoutInvalid {
        path: root.to_path_buf(),
        message: "missing metadata.jsonl (expected at dataset root or inside split subdirectories)"
            .to_string(),
    })
}

fn read_split_rows(
    split_dir: &Path,
    options: &HfReadOptions,
) -> Result<Vec<ParsedRow>, PanlabelError> {
    let metadata_path = split_dir.join("metadata.jsonl");
    let file = fs::File::open(&metadata_path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);

    let mut rows = Vec::new();

    for (line_idx, line_res) in reader.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line_res.map_err(PanlabelError::Io)?;
        if line.trim().is_empty() {
            continue;
        }

        let value: Value =
            serde_json::from_str(&line).map_err(|source| PanlabelError::HfJsonlParse {
                path: metadata_path.clone(),
                line: line_num,
                message: source.to_string(),
            })?;

        rows.push(parse_jsonl_row(
            &metadata_path,
            line_num,
            &value,
            split_dir,
            options,
        )?);
    }

    Ok(rows)
}

pub(crate) fn parse_jsonl_row(
    metadata_path: &Path,
    line: usize,
    value: &Value,
    split_dir: &Path,
    options: &HfReadOptions,
) -> Result<ParsedRow, PanlabelError> {
    let row_obj = value
        .as_object()
        .ok_or_else(|| PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: "line is not a JSON object".to_string(),
        })?;

    let file_name = row_obj
        .get("file_name")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: "missing required string field 'file_name'".to_string(),
        })?;

    let width = parse_optional_u32(row_obj.get("width"), "width", metadata_path, line)?;
    let height = parse_optional_u32(row_obj.get("height"), "height", metadata_path, line)?;

    let objects_key = resolve_objects_key(row_obj, options, metadata_path, line)?;
    let objects = row_obj
        .get(&objects_key)
        .and_then(Value::as_object)
        .ok_or_else(|| PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!("field '{}' must be a JSON object", objects_key),
        })?;

    let bboxes = objects
        .get("bbox")
        .and_then(Value::as_array)
        .ok_or_else(|| PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!(
                "missing required field '{}.bbox' (expected an array of 4-element arrays)",
                objects_key
            ),
        })?;

    let category_field = if objects.contains_key("categories") {
        "categories"
    } else if objects.contains_key("category") {
        "category"
    } else {
        return Err(PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!(
                "no recognized category field found in '{}'. Expected 'categories' or 'category'",
                objects_key
            ),
        });
    };

    let categories = objects
        .get(category_field)
        .and_then(Value::as_array)
        .ok_or_else(|| PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!(
                "field '{}.{}' must be an array",
                objects_key, category_field
            ),
        })?;

    if bboxes.len() != categories.len() {
        return Err(PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!(
                "length mismatch: '{}.bbox' has {} item(s) but '{}.{}' has {} item(s)",
                objects_key,
                bboxes.len(),
                objects_key,
                category_field,
                categories.len()
            ),
        });
    }

    let mut anns = Vec::with_capacity(bboxes.len());
    for (idx, (bbox_value, category_value)) in bboxes.iter().zip(categories.iter()).enumerate() {
        let bbox = parse_bbox_array(bbox_value, metadata_path, line, idx + 1)?;
        let category = parse_category_value(category_value, metadata_path, line, idx + 1)?;
        anns.push(ParsedAnn { bbox, category });
    }

    let (width, height) =
        ensure_dimensions(width, height, split_dir, &file_name, metadata_path, line)?;

    Ok(ParsedRow {
        file_name,
        width: Some(width),
        height: Some(height),
        anns,
    })
}

fn resolve_objects_key(
    row_obj: &Map<String, Value>,
    options: &HfReadOptions,
    metadata_path: &Path,
    line: usize,
) -> Result<String, PanlabelError> {
    if let Some(user_key) = options.objects_column.as_deref() {
        if row_obj.contains_key(user_key) {
            return Ok(user_key.to_string());
        }
        return Err(PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!(
                "object container '{}' not found. Available keys: {}",
                user_key,
                row_obj.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
        });
    }

    if row_obj.contains_key("objects") {
        return Ok("objects".to_string());
    }
    if row_obj.contains_key("faces") {
        return Ok("faces".to_string());
    }

    Err(PanlabelError::HfJsonlParse {
        path: metadata_path.to_path_buf(),
        line,
        message:
            "no recognized object container found. Expected 'objects' or 'faces'. Use --hf-objects-column to specify"
                .to_string(),
    })
}

fn parse_optional_u32(
    value: Option<&Value>,
    field_name: &str,
    metadata_path: &Path,
    line: usize,
) -> Result<Option<u32>, PanlabelError> {
    let Some(value) = value else {
        return Ok(None);
    };

    let number = value.as_u64().ok_or_else(|| PanlabelError::HfJsonlParse {
        path: metadata_path.to_path_buf(),
        line,
        message: format!("field '{}' must be an unsigned integer", field_name),
    })?;

    u32::try_from(number)
        .map(Some)
        .map_err(|_| PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!("field '{}' out of range for u32: {}", field_name, number),
        })
}

fn parse_bbox_array(
    value: &Value,
    metadata_path: &Path,
    line: usize,
    ann_index: usize,
) -> Result<[f64; 4], PanlabelError> {
    let arr = value
        .as_array()
        .ok_or_else(|| PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!("bbox #{} must be a 4-element array of numbers", ann_index),
        })?;

    if arr.len() != 4 {
        return Err(PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!(
                "bbox #{} must have exactly 4 values, found {}",
                ann_index,
                arr.len()
            ),
        });
    }

    let mut out = [0.0; 4];
    for (i, item) in arr.iter().enumerate() {
        out[i] = item.as_f64().ok_or_else(|| PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!("bbox #{} element {} is not a number", ann_index, i + 1),
        })?;
    }

    Ok(out)
}

fn parse_category_value(
    value: &Value,
    metadata_path: &Path,
    line: usize,
    ann_index: usize,
) -> Result<ParsedCategory, PanlabelError> {
    if let Some(name) = value.as_str() {
        return Ok(ParsedCategory::Name(name.to_string()));
    }

    if let Some(integer) = value.as_i64() {
        return Ok(ParsedCategory::Id(integer));
    }

    if let Some(float) = value.as_f64() {
        if float.fract() == 0.0 {
            return Ok(ParsedCategory::Id(float as i64));
        }
    }

    Err(PanlabelError::HfJsonlParse {
        path: metadata_path.to_path_buf(),
        line,
        message: format!(
            "category #{} must be an integer ID or string label",
            ann_index
        ),
    })
}

fn ensure_dimensions(
    width: Option<u32>,
    height: Option<u32>,
    split_dir: &Path,
    file_name: &str,
    metadata_path: &Path,
    line: usize,
) -> Result<(u32, u32), PanlabelError> {
    if let (Some(w), Some(h)) = (width, height) {
        return Ok((w, h));
    }

    let image_path = split_dir.join(file_name);
    let image_size =
        imagesize::size(&image_path).map_err(|source| PanlabelError::HfJsonlParse {
            path: metadata_path.to_path_buf(),
            line,
            message: format!(
                "missing width/height and failed to read image dimensions from '{}': {}",
                image_path.display(),
                source
            ),
        })?;

    Ok((
        width.unwrap_or(image_size.width as u32),
        height.unwrap_or(image_size.height as u32),
    ))
}

pub(crate) fn dataset_from_rows(
    rows: Vec<ParsedRow>,
    options: &HfReadOptions,
) -> Result<Dataset, PanlabelError> {
    let mut by_file: HashMap<String, ParsedRow> = HashMap::new();
    for row in rows {
        if by_file.contains_key(&row.file_name) {
            return Err(PanlabelError::HfLayoutInvalid {
                path: PathBuf::from("metadata.jsonl"),
                message: format!("duplicate file_name '{}' in HF metadata", row.file_name),
            });
        }
        by_file.insert(row.file_name.clone(), row);
    }

    let mut file_names: Vec<String> = by_file.keys().cloned().collect();
    file_names.sort();

    let mut category_names_set: BTreeSet<String> = BTreeSet::new();
    let mut file_ann_categories: HashMap<String, Vec<String>> = HashMap::new();

    for file_name in &file_names {
        let row = by_file
            .get(file_name)
            .expect("file name list is derived from map keys");
        let mut ann_categories = Vec::with_capacity(row.anns.len());
        for ann in &row.anns {
            let name = match &ann.category {
                ParsedCategory::Name(name) => name.clone(),
                ParsedCategory::Id(id) => options
                    .category_map
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| id.to_string()),
            };
            category_names_set.insert(name.clone());
            ann_categories.push(name);
        }
        file_ann_categories.insert(file_name.clone(), ann_categories);
    }

    let categories: Vec<Category> = category_names_set
        .iter()
        .enumerate()
        .map(|(idx, name)| Category::new((idx + 1) as u64, name.clone()))
        .collect();

    let category_id_by_name: BTreeMap<String, CategoryId> = categories
        .iter()
        .map(|category| (category.name.clone(), category.id))
        .collect();

    let mut images = Vec::with_capacity(file_names.len());
    let mut image_id_by_file_name = BTreeMap::new();

    for (idx, file_name) in file_names.iter().enumerate() {
        let row = by_file
            .get(file_name)
            .expect("file name list is derived from map keys");
        let image_id = ImageId::new((idx + 1) as u64);
        images.push(Image::new(
            image_id,
            file_name.clone(),
            row.width.expect("width is filled during parse"),
            row.height.expect("height is filled during parse"),
        ));
        image_id_by_file_name.insert(file_name.clone(), image_id);
    }

    let mut annotations = Vec::new();
    let mut next_ann_id: u64 = 1;

    for file_name in &file_names {
        let row = by_file
            .get(file_name)
            .expect("file name list is derived from map keys");
        let image_id = image_id_by_file_name[file_name];
        let category_names = file_ann_categories
            .get(file_name)
            .expect("category names were computed for each file");

        for (ann_idx, ann) in row.anns.iter().enumerate() {
            let bbox = match options.bbox_format {
                HfBboxFormat::Xywh => {
                    BBoxXYXY::from_xywh(ann.bbox[0], ann.bbox[1], ann.bbox[2], ann.bbox[3])
                }
                HfBboxFormat::Xyxy => {
                    BBoxXYXY::from_xyxy(ann.bbox[0], ann.bbox[1], ann.bbox[2], ann.bbox[3])
                }
            };

            let category_name = &category_names[ann_idx];
            let category_id = category_id_by_name[category_name];
            annotations.push(Annotation::new(next_ann_id, image_id, category_id, bbox));
            next_ann_id += 1;
        }
    }

    let mut attributes = options.provenance.clone();
    attributes
        .entry("hf_bbox_format".to_string())
        .or_insert_with(|| options.bbox_format.as_str().to_string());

    Ok(Dataset {
        info: DatasetInfo {
            attributes,
            ..Default::default()
        },
        licenses: vec![],
        images,
        categories,
        annotations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_jsonl(root: &Path, rows: &[&str]) {
        fs::create_dir_all(root).expect("create root");
        let path = root.join("metadata.jsonl");
        let mut file = fs::File::create(path).expect("create metadata");
        for row in rows {
            writeln!(file, "{row}").expect("write row");
        }
    }

    fn write_dummy_bmp(path: &Path, width: u32, height: u32) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }

        let row_stride = (width * 3).div_ceil(4) * 4;
        let pixel_array_size = row_stride * height;
        let file_size = 54 + pixel_array_size;

        let mut bytes = Vec::with_capacity(file_size as usize);
        bytes.extend_from_slice(b"BM");
        bytes.extend_from_slice(&file_size.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes.extend_from_slice(&54u32.to_le_bytes());
        bytes.extend_from_slice(&40u32.to_le_bytes());
        bytes.extend_from_slice(&(width as i32).to_le_bytes());
        bytes.extend_from_slice(&(height as i32).to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&24u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&pixel_array_size.to_le_bytes());
        bytes.extend_from_slice(&2835u32.to_le_bytes());
        bytes.extend_from_slice(&2835u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.resize(file_size as usize, 0);

        fs::write(path, bytes).expect("write bmp");
    }

    #[test]
    fn read_xywh_jsonl() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_jsonl(
            temp.path(),
            &[
                r#"{"file_name":"img1.bmp","width":20,"height":10,"objects":{"bbox":[[2,1,5,4]],"categories":["person"]}}"#,
            ],
        );

        let dataset = read_hf_imagefolder(temp.path()).expect("read dataset");
        assert_eq!(dataset.images.len(), 1);
        assert_eq!(dataset.categories[0].name, "person");
        assert_eq!(dataset.annotations.len(), 1);

        let bbox = &dataset.annotations[0].bbox;
        assert!((bbox.xmin() - 2.0).abs() < 1e-9);
        assert!((bbox.ymin() - 1.0).abs() < 1e-9);
        assert!((bbox.xmax() - 7.0).abs() < 1e-9);
        assert!((bbox.ymax() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn read_xyxy_jsonl() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_jsonl(
            temp.path(),
            &[
                r#"{"file_name":"img1.bmp","width":20,"height":10,"objects":{"bbox":[[2,1,7,5]],"categories":["person"]}}"#,
            ],
        );

        let options = HfReadOptions {
            bbox_format: HfBboxFormat::Xyxy,
            ..Default::default()
        };
        let dataset =
            read_hf_imagefolder_with_options(temp.path(), &options).expect("read dataset");

        let bbox = &dataset.annotations[0].bbox;
        assert!((bbox.xmin() - 2.0).abs() < 1e-9);
        assert!((bbox.ymin() - 1.0).abs() < 1e-9);
        assert!((bbox.xmax() - 7.0).abs() < 1e-9);
        assert!((bbox.ymax() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn read_uses_imagesize_when_dimensions_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_dummy_bmp(&temp.path().join("img1.bmp"), 32, 16);
        write_jsonl(
            temp.path(),
            &[r#"{"file_name":"img1.bmp","objects":{"bbox":[[1,2,3,4]],"categories":["person"]}}"#],
        );

        let dataset = read_hf_imagefolder(temp.path()).expect("read dataset");
        assert_eq!(dataset.images[0].width, 32);
        assert_eq!(dataset.images[0].height, 16);
    }

    #[test]
    fn read_errors_on_missing_objects_column() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_jsonl(
            temp.path(),
            &[
                r#"{"file_name":"img1.bmp","annotations":{"bbox":[[1,2,3,4]],"categories":["person"]}}"#,
            ],
        );

        let err = read_hf_imagefolder(temp.path()).expect_err("should fail");
        match err {
            PanlabelError::HfJsonlParse { message, .. } => {
                assert!(message.contains("objects"));
                assert!(message.contains("faces"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn write_then_read_roundtrip_counts() {
        let temp = tempfile::tempdir().expect("tempdir");
        let out_dir = temp.path().join("out");

        let dataset = Dataset {
            info: DatasetInfo::default(),
            licenses: vec![],
            images: vec![
                Image::new(1u64, "img_a.bmp", 20, 10),
                Image::new(2u64, "img_b.bmp", 30, 15),
            ],
            categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
            annotations: vec![
                Annotation::new(1u64, 1u64, 1u64, BBoxXYXY::from_xyxy(1.0, 2.0, 4.0, 6.0)),
                Annotation::new(2u64, 1u64, 2u64, BBoxXYXY::from_xyxy(3.0, 1.0, 8.0, 5.0)),
            ],
        };

        write_hf_imagefolder(&out_dir, &dataset).expect("write dataset");
        let restored = read_hf_imagefolder(&out_dir).expect("read dataset");

        assert_eq!(restored.images.len(), dataset.images.len());
        assert_eq!(restored.categories.len(), dataset.categories.len());
        assert_eq!(restored.annotations.len(), dataset.annotations.len());
    }
}
