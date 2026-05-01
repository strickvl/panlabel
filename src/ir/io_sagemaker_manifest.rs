//! SageMaker Ground Truth object-detection manifest reader and writer.
//!
//! This adapter supports annotated JSON Lines manifests where each row has a
//! `source-ref`, one object-detection label attribute, and the paired
//! `<label>-metadata` object used by SageMaker Ground Truth output manifests.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId};
use crate::error::PanlabelError;

pub const ATTR_LABEL_ATTRIBUTE_NAME: &str = "sagemaker_label_attribute_name";
pub const ATTR_SOURCE_REF: &str = "sagemaker_source_ref";
pub const ATTR_IMAGE_DEPTH: &str = "sagemaker_image_depth";
pub const ATTR_METADATA_TYPE: &str = "sagemaker_metadata_type";
pub const ATTR_HUMAN_ANNOTATED: &str = "sagemaker_human_annotated";
pub const ATTR_CREATION_DATE: &str = "sagemaker_creation_date";
pub const ATTR_JOB_NAME: &str = "sagemaker_job_name";
pub const ATTR_CLASS_ID: &str = "sagemaker_class_id";

const DEFAULT_LABEL_ATTRIBUTE: &str = "bounding-box";
const OBJECT_DETECTION_TYPE: &str = "groundtruth/object-detection";
const STRING_HELPER_PATH: &str = "<manifest string>";

#[derive(Debug)]
struct ParsedRow {
    source_ref: String,
    file_name: String,
    width: u32,
    height: u32,
    depth: Option<u32>,
    label_attribute_name: String,
    metadata: ParsedMetadata,
    annotations: Vec<ParsedAnnotation>,
    line: usize,
}

#[derive(Debug, Default)]
struct ParsedMetadata {
    class_map: BTreeMap<i64, String>,
    metadata_type: Option<String>,
    human_annotated: Option<String>,
    creation_date: Option<String>,
    job_name: Option<String>,
}

#[derive(Debug)]
struct ParsedAnnotation {
    source_class_id: i64,
    bbox_xywh: [f64; 4],
    confidence: Option<f64>,
}

/// Read a SageMaker Ground Truth object-detection manifest from a JSONL file.
pub fn read_sagemaker_manifest(path: &Path) -> Result<Dataset, PanlabelError> {
    let manifest = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    from_sagemaker_manifest_str_with_path(&manifest, path)
}

/// Write a SageMaker Ground Truth object-detection manifest JSONL file.
pub fn write_sagemaker_manifest(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let manifest = to_sagemaker_manifest_string_with_path(dataset, path)?;
    fs::write(path, manifest).map_err(PanlabelError::Io)
}

/// Parse a SageMaker manifest from a string.
pub fn from_sagemaker_manifest_str(manifest: &str) -> Result<Dataset, PanlabelError> {
    from_sagemaker_manifest_str_with_path(manifest, Path::new(STRING_HELPER_PATH))
}

/// Serialize a dataset as a SageMaker manifest string.
pub fn to_sagemaker_manifest_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    to_sagemaker_manifest_string_with_path(dataset, Path::new(STRING_HELPER_PATH))
}

fn from_sagemaker_manifest_str_with_path(
    manifest: &str,
    path: &Path,
) -> Result<Dataset, PanlabelError> {
    let mut rows = Vec::new();

    for (idx, line) in manifest.lines().enumerate() {
        let line_num = idx + 1;
        if line.trim().is_empty() {
            continue;
        }

        let value: Value =
            serde_json::from_str(line).map_err(|source| PanlabelError::SageMakerManifestParse {
                path: path.to_path_buf(),
                line: line_num,
                message: source.to_string(),
            })?;
        rows.push(parse_manifest_row(path, line_num, &value)?);
    }

    dataset_from_rows(rows, path)
}

fn parse_manifest_row(path: &Path, line: usize, value: &Value) -> Result<ParsedRow, PanlabelError> {
    let row = value
        .as_object()
        .ok_or_else(|| parse_error(path, line, "line is not a JSON object"))?;

    reject_unsupported_metadata_types(path, line, row)?;

    let source_ref = row
        .get("source-ref")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            parse_error(
                path,
                line,
                "missing required non-empty string field 'source-ref'",
            )
        })?;

    let label_attribute_name = detect_label_attribute(path, line, row)?;
    let label = row
        .get(&label_attribute_name)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            parse_error(
                path,
                line,
                format!("field '{label_attribute_name}' must be a JSON object"),
            )
        })?;

    let image_size = label
        .get("image_size")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            parse_error(
                path,
                line,
                format!("missing required array field '{label_attribute_name}.image_size'"),
            )
        })?;
    let first_size =
        image_size
            .first()
            .and_then(Value::as_object)
            .ok_or_else(|| {
                parse_error(
            path,
            line,
            format!("field '{label_attribute_name}.image_size' must contain at least one object"),
        )
            })?;
    let width = parse_required_u32(first_size.get("width"), path, line, "image_size[0].width")?;
    let height = parse_required_u32(first_size.get("height"), path, line, "image_size[0].height")?;
    let depth = parse_optional_u32(first_size.get("depth"), path, line, "image_size[0].depth")?;

    let annotations = label
        .get("annotations")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            parse_error(
                path,
                line,
                format!("missing required array field '{label_attribute_name}.annotations'"),
            )
        })?;

    let metadata_key = format!("{label_attribute_name}-metadata");
    let metadata = match row.get(&metadata_key) {
        Some(value) => Some(value.as_object().ok_or_else(|| {
            parse_error(
                path,
                line,
                format!("field '{metadata_key}' must be a JSON object"),
            )
        })?),
        None => None,
    };
    let parsed_metadata = parse_metadata(path, line, metadata)?;
    let confidences = parse_confidences(path, line, metadata, annotations.len())?;

    let mut parsed_annotations = Vec::with_capacity(annotations.len());
    for (idx, ann_value) in annotations.iter().enumerate() {
        parsed_annotations.push(parse_annotation(
            path,
            line,
            &label_attribute_name,
            idx,
            ann_value,
            confidences.get(idx).copied().flatten(),
        )?);
    }

    Ok(ParsedRow {
        source_ref: source_ref.clone(),
        file_name: derive_file_name(&source_ref),
        width,
        height,
        depth,
        label_attribute_name,
        metadata: parsed_metadata,
        annotations: parsed_annotations,
        line,
    })
}

fn reject_unsupported_metadata_types(
    path: &Path,
    line: usize,
    row: &Map<String, Value>,
) -> Result<(), PanlabelError> {
    for (key, value) in row {
        if !key.ends_with("-metadata") {
            continue;
        }
        let Some(metadata) = value.as_object() else {
            continue;
        };
        let Some(metadata_type) = metadata.get("type").and_then(Value::as_str) else {
            continue;
        };
        if metadata_type != OBJECT_DETECTION_TYPE {
            return Err(parse_error(
                path,
                line,
                format!(
                    "unsupported SageMaker metadata type '{metadata_type}' in '{key}' (only '{OBJECT_DETECTION_TYPE}' is supported)"
                ),
            ));
        }
    }
    Ok(())
}

fn detect_label_attribute(
    path: &Path,
    line: usize,
    row: &Map<String, Value>,
) -> Result<String, PanlabelError> {
    let mut candidates = Vec::new();
    for (key, value) in row {
        if key == "source-ref" || key.ends_with("-metadata") {
            continue;
        }
        let Some(object) = value.as_object() else {
            continue;
        };
        let has_object_detection_shape = object.get("annotations").is_some()
            && object.get("image_size").is_some()
            && row
                .get(&format!("{key}-metadata"))
                .and_then(Value::as_object)
                .and_then(|metadata| metadata.get("type"))
                .and_then(Value::as_str)
                .map(|metadata_type| metadata_type == OBJECT_DETECTION_TYPE)
                .unwrap_or(true);
        if has_object_detection_shape {
            candidates.push(key.clone());
        }
    }

    match candidates.len() {
        1 => Ok(candidates.remove(0)),
        0 => Err(parse_error(
            path,
            line,
            "expected exactly one object-detection label attribute with 'annotations' and 'image_size'",
        )),
        _ => Err(parse_error(
            path,
            line,
            format!(
                "expected exactly one object-detection label attribute, found {}: {}",
                candidates.len(),
                candidates.join(", ")
            ),
        )),
    }
}

fn parse_metadata(
    path: &Path,
    line: usize,
    metadata: Option<&Map<String, Value>>,
) -> Result<ParsedMetadata, PanlabelError> {
    let Some(metadata) = metadata else {
        return Ok(ParsedMetadata::default());
    };

    let class_map = match metadata.get("class-map") {
        Some(Value::Object(map)) => {
            let mut out = BTreeMap::new();
            for (key, value) in map {
                let class_id = key.parse::<i64>().map_err(|_| {
                    parse_error(
                        path,
                        line,
                        format!("class-map key '{key}' is not an integer"),
                    )
                })?;
                let name = value.as_str().ok_or_else(|| {
                    parse_error(
                        path,
                        line,
                        format!("class-map value for class id {class_id} must be a string"),
                    )
                })?;
                out.insert(class_id, name.to_string());
            }
            out
        }
        Some(_) => {
            return Err(parse_error(
                path,
                line,
                "metadata field 'class-map' must be a JSON object",
            ));
        }
        None => BTreeMap::new(),
    };

    Ok(ParsedMetadata {
        class_map,
        metadata_type: optional_string(metadata, "type"),
        human_annotated: optional_string(metadata, "human-annotated"),
        creation_date: optional_string(metadata, "creation-date"),
        job_name: optional_string(metadata, "job-name"),
    })
}

fn parse_confidences(
    path: &Path,
    line: usize,
    metadata: Option<&Map<String, Value>>,
    annotation_count: usize,
) -> Result<Vec<Option<f64>>, PanlabelError> {
    let Some(objects_value) = metadata.and_then(|metadata| metadata.get("objects")) else {
        return Ok(vec![None; annotation_count]);
    };

    let objects = objects_value
        .as_array()
        .ok_or_else(|| parse_error(path, line, "metadata field 'objects' must be a JSON array"))?;

    if !objects.is_empty() && objects.len() != annotation_count {
        return Err(parse_error(
            path,
            line,
            format!(
                "metadata.objects length {} does not match annotations length {}",
                objects.len(),
                annotation_count
            ),
        ));
    }

    let mut confidences = Vec::with_capacity(annotation_count);
    for (idx, object) in objects.iter().enumerate() {
        let object = object.as_object().ok_or_else(|| {
            parse_error(
                path,
                line,
                format!("metadata.objects[{}] must be a JSON object", idx),
            )
        })?;
        let confidence = match object.get("confidence") {
            Some(value) => Some(parse_finite_f64(
                value,
                path,
                line,
                &format!("metadata.objects[{idx}].confidence"),
            )?),
            None => None,
        };
        confidences.push(confidence);
    }

    if confidences.is_empty() {
        Ok(vec![None; annotation_count])
    } else {
        Ok(confidences)
    }
}

fn parse_annotation(
    path: &Path,
    line: usize,
    label_attribute_name: &str,
    idx: usize,
    value: &Value,
    confidence: Option<f64>,
) -> Result<ParsedAnnotation, PanlabelError> {
    let object = value.as_object().ok_or_else(|| {
        parse_error(
            path,
            line,
            format!("{label_attribute_name}.annotations[{idx}] must be a JSON object"),
        )
    })?;

    let source_class_id = parse_required_i64(
        object.get("class_id"),
        path,
        line,
        &format!("{label_attribute_name}.annotations[{idx}].class_id"),
    )?;
    let left = parse_finite_f64(
        required_value(object.get("left"), path, line, "annotation.left")?,
        path,
        line,
        &format!("{label_attribute_name}.annotations[{idx}].left"),
    )?;
    let top = parse_finite_f64(
        required_value(object.get("top"), path, line, "annotation.top")?,
        path,
        line,
        &format!("{label_attribute_name}.annotations[{idx}].top"),
    )?;
    let width = parse_finite_f64(
        required_value(object.get("width"), path, line, "annotation.width")?,
        path,
        line,
        &format!("{label_attribute_name}.annotations[{idx}].width"),
    )?;
    let height = parse_finite_f64(
        required_value(object.get("height"), path, line, "annotation.height")?,
        path,
        line,
        &format!("{label_attribute_name}.annotations[{idx}].height"),
    )?;

    if width < 0.0 || height < 0.0 {
        return Err(parse_error(
            path,
            line,
            format!(
                "{label_attribute_name}.annotations[{idx}] has negative width/height ({width}, {height})"
            ),
        ));
    }

    Ok(ParsedAnnotation {
        source_class_id,
        bbox_xywh: [left, top, width, height],
        confidence,
    })
}

fn dataset_from_rows(rows: Vec<ParsedRow>, path: &Path) -> Result<Dataset, PanlabelError> {
    let mut by_file_name: HashMap<String, ParsedRow> = HashMap::new();
    let mut label_attribute_name: Option<String> = None;

    for row in rows {
        if let Some(existing_label) = &label_attribute_name {
            if existing_label != &row.label_attribute_name {
                return Err(PanlabelError::SageMakerManifestInvalid {
                    path: path.to_path_buf(),
                    message: format!(
                        "mixed label attribute names: first row used '{existing_label}', line {} used '{}'",
                        row.line, row.label_attribute_name
                    ),
                });
            }
        } else {
            label_attribute_name = Some(row.label_attribute_name.clone());
        }

        if by_file_name.contains_key(&row.file_name) {
            return Err(PanlabelError::SageMakerManifestInvalid {
                path: path.to_path_buf(),
                message: format!("duplicate derived image file_name '{}'", row.file_name),
            });
        }
        by_file_name.insert(row.file_name.clone(), row);
    }

    let mut file_names: Vec<String> = by_file_name.keys().cloned().collect();
    file_names.sort();

    let mut source_class_ids = BTreeSet::new();
    let mut class_names_by_id = BTreeMap::new();
    for file_name in &file_names {
        let row = by_file_name
            .get(file_name)
            .expect("file name list is derived from map keys");
        for (class_id, name) in &row.metadata.class_map {
            source_class_ids.insert(*class_id);
            if let Some(existing) = class_names_by_id.get(class_id) {
                if existing != name {
                    return Err(PanlabelError::SageMakerManifestInvalid {
                        path: path.to_path_buf(),
                        message: format!(
                            "conflicting class-map names for class_id {class_id}: '{existing}' vs '{name}'"
                        ),
                    });
                }
            } else {
                class_names_by_id.insert(*class_id, name.clone());
            }
        }
        for ann in &row.annotations {
            source_class_ids.insert(ann.source_class_id);
        }
    }

    let categories: Vec<Category> = source_class_ids
        .iter()
        .enumerate()
        .map(|(idx, source_class_id)| {
            Category::new(
                (idx + 1) as u64,
                class_names_by_id
                    .get(source_class_id)
                    .cloned()
                    .unwrap_or_else(|| source_class_id.to_string()),
            )
        })
        .collect();
    let category_id_by_source_id: BTreeMap<i64, CategoryId> = source_class_ids
        .iter()
        .enumerate()
        .map(|(idx, source_class_id)| (*source_class_id, CategoryId::new((idx + 1) as u64)))
        .collect();

    let mut images = Vec::with_capacity(file_names.len());
    let mut image_id_by_file_name = BTreeMap::new();
    for (idx, file_name) in file_names.iter().enumerate() {
        let row = by_file_name
            .get(file_name)
            .expect("file name list is derived from map keys");
        let image_id = ImageId::new((idx + 1) as u64);
        let mut image = Image::new(image_id, file_name.clone(), row.width, row.height);
        image
            .attributes
            .insert(ATTR_SOURCE_REF.to_string(), row.source_ref.clone());
        if let Some(depth) = row.depth {
            image
                .attributes
                .insert(ATTR_IMAGE_DEPTH.to_string(), depth.to_string());
        }
        if let Some(value) = &row.metadata.metadata_type {
            image
                .attributes
                .insert(ATTR_METADATA_TYPE.to_string(), value.clone());
        }
        if let Some(value) = &row.metadata.human_annotated {
            image
                .attributes
                .insert(ATTR_HUMAN_ANNOTATED.to_string(), value.clone());
        }
        if let Some(value) = &row.metadata.creation_date {
            image
                .attributes
                .insert(ATTR_CREATION_DATE.to_string(), value.clone());
        }
        if let Some(value) = &row.metadata.job_name {
            image
                .attributes
                .insert(ATTR_JOB_NAME.to_string(), value.clone());
        }
        images.push(image);
        image_id_by_file_name.insert(file_name.clone(), image_id);
    }

    let mut annotations = Vec::new();
    let mut next_ann_id = 1u64;
    for file_name in &file_names {
        let row = by_file_name
            .get(file_name)
            .expect("file name list is derived from map keys");
        let image_id = image_id_by_file_name[file_name];
        for parsed_ann in &row.annotations {
            let category_id = category_id_by_source_id[&parsed_ann.source_class_id];
            let bbox = BBoxXYXY::from_xywh(
                parsed_ann.bbox_xywh[0],
                parsed_ann.bbox_xywh[1],
                parsed_ann.bbox_xywh[2],
                parsed_ann.bbox_xywh[3],
            );
            let mut ann =
                Annotation::new(AnnotationId::new(next_ann_id), image_id, category_id, bbox);
            ann.confidence = parsed_ann.confidence;
            ann.attributes.insert(
                ATTR_CLASS_ID.to_string(),
                parsed_ann.source_class_id.to_string(),
            );
            annotations.push(ann);
            next_ann_id += 1;
        }
    }

    let mut attributes = BTreeMap::new();
    if let Some(label) = label_attribute_name {
        attributes.insert(ATTR_LABEL_ATTRIBUTE_NAME.to_string(), label);
    }

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

fn to_sagemaker_manifest_string_with_path(
    dataset: &Dataset,
    path: &Path,
) -> Result<String, PanlabelError> {
    let label_attribute = dataset
        .info
        .attributes
        .get(ATTR_LABEL_ATTRIBUTE_NAME)
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .unwrap_or_else(|| DEFAULT_LABEL_ATTRIBUTE.to_string());
    let metadata_attribute = format!("{label_attribute}-metadata");

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
    let mut categories_sorted: Vec<&Category> = dataset.categories.iter().collect();
    categories_sorted.sort_by_key(|category| category.id);
    let class_id_by_category_id: BTreeMap<CategoryId, i64> = categories_sorted
        .iter()
        .enumerate()
        .map(|(idx, category)| (category.id, idx as i64))
        .collect();

    let mut anns_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        if !image_lookup.contains_key(&ann.image_id) {
            return Err(write_error(
                path,
                format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            ));
        }
        if !category_lookup.contains_key(&ann.category_id) {
            return Err(write_error(
                path,
                format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            ));
        }
        if !ann.bbox.is_finite() {
            return Err(write_error(
                path,
                format!(
                    "annotation {} has non-finite bbox coordinates",
                    ann.id.as_u64()
                ),
            ));
        }
        if let Some(confidence) = ann.confidence {
            if !confidence.is_finite() {
                return Err(write_error(
                    path,
                    format!("annotation {} has non-finite confidence", ann.id.as_u64()),
                ));
            }
        }
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }

    let mut class_map = Map::new();
    for category in categories_sorted {
        let class_id = class_id_by_category_id[&category.id];
        class_map.insert(class_id.to_string(), Value::String(category.name.clone()));
    }

    let mut images_sorted: Vec<&Image> = dataset.images.iter().collect();
    images_sorted.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    let mut out = Vec::new();
    for image in images_sorted {
        if image.width == 0 || image.height == 0 {
            return Err(write_error(
                path,
                format!(
                    "image {} ('{}') has zero width/height ({}x{})",
                    image.id.as_u64(),
                    image.file_name,
                    image.width,
                    image.height
                ),
            ));
        }

        let depth = parse_writer_depth(path, image)?;
        let mut anns = anns_by_image.remove(&image.id).unwrap_or_default();
        anns.sort_by_key(|ann| ann.id);

        let mut annotation_values = Vec::with_capacity(anns.len());
        let mut object_values = Vec::with_capacity(anns.len());
        for ann in anns {
            let class_id = class_id_by_category_id[&ann.category_id];
            let (left, top, width, height) = ann.bbox.to_xywh();
            if width < 0.0 || height < 0.0 {
                return Err(write_error(
                    path,
                    format!(
                        "annotation {} has negative bbox width/height after XYWH conversion ({width}, {height})",
                        ann.id.as_u64()
                    ),
                ));
            }
            let mut annotation = Map::new();
            annotation.insert("class_id".to_string(), Value::from(class_id));
            annotation.insert("left".to_string(), Value::from(left));
            annotation.insert("top".to_string(), Value::from(top));
            annotation.insert("width".to_string(), Value::from(width));
            annotation.insert("height".to_string(), Value::from(height));
            annotation_values.push(Value::Object(annotation));

            let mut object = Map::new();
            if let Some(confidence) = ann.confidence {
                object.insert("confidence".to_string(), Value::from(confidence));
            }
            object_values.push(Value::Object(object));
        }

        let mut image_size = Map::new();
        image_size.insert("width".to_string(), Value::from(image.width));
        image_size.insert("height".to_string(), Value::from(image.height));
        image_size.insert("depth".to_string(), Value::from(depth));

        let mut label = Map::new();
        label.insert("annotations".to_string(), Value::Array(annotation_values));
        label.insert(
            "image_size".to_string(),
            Value::Array(vec![Value::Object(image_size)]),
        );

        let mut metadata = Map::new();
        metadata.insert("objects".to_string(), Value::Array(object_values));
        metadata.insert("class-map".to_string(), Value::Object(class_map.clone()));
        metadata.insert(
            "type".to_string(),
            Value::String(OBJECT_DETECTION_TYPE.to_string()),
        );
        metadata.insert(
            "human-annotated".to_string(),
            Value::String("yes".to_string()),
        );
        metadata.insert(
            "job-name".to_string(),
            Value::String(
                image
                    .attributes
                    .get(ATTR_JOB_NAME)
                    .cloned()
                    .unwrap_or_else(|| "panlabel-export".to_string()),
            ),
        );
        if let Some(creation_date) = image.attributes.get(ATTR_CREATION_DATE) {
            metadata.insert(
                "creation-date".to_string(),
                Value::String(creation_date.clone()),
            );
        } else if let Some(creation_date) = dataset.info.attributes.get(ATTR_CREATION_DATE) {
            metadata.insert(
                "creation-date".to_string(),
                Value::String(creation_date.clone()),
            );
        }

        let mut row = Map::new();
        row.insert(
            "source-ref".to_string(),
            Value::String(
                image
                    .attributes
                    .get(ATTR_SOURCE_REF)
                    .cloned()
                    .unwrap_or_else(|| image.file_name.clone()),
            ),
        );
        row.insert(label_attribute.clone(), Value::Object(label));
        row.insert(metadata_attribute.clone(), Value::Object(metadata));

        serde_json::to_writer(&mut out, &Value::Object(row))
            .map_err(|source| write_error(path, format!("JSON serialization failed: {source}")))?;
        writeln!(&mut out).map_err(PanlabelError::Io)?;
    }

    String::from_utf8(out).map_err(|source| write_error(path, source.to_string()))
}

fn parse_writer_depth(path: &Path, image: &Image) -> Result<u32, PanlabelError> {
    match image.attributes.get(ATTR_IMAGE_DEPTH) {
        Some(raw) => raw.parse::<u32>().map_err(|_| {
            write_error(
                path,
                format!(
                    "image {} ('{}') has invalid {ATTR_IMAGE_DEPTH} value '{}'",
                    image.id.as_u64(),
                    image.file_name,
                    raw
                ),
            )
        }),
        None => Ok(3),
    }
}

fn required_value<'a>(
    value: Option<&'a Value>,
    path: &Path,
    line: usize,
    field_name: &str,
) -> Result<&'a Value, PanlabelError> {
    value.ok_or_else(|| parse_error(path, line, format!("missing required field '{field_name}'")))
}

fn parse_required_i64(
    value: Option<&Value>,
    path: &Path,
    line: usize,
    field_name: &str,
) -> Result<i64, PanlabelError> {
    let value = required_value(value, path, line, field_name)?;
    if let Some(integer) = value.as_i64() {
        return Ok(integer);
    }
    if let Some(unsigned) = value.as_u64() {
        return i64::try_from(unsigned)
            .map_err(|_| parse_error(path, line, format!("field '{field_name}' is out of range")));
    }
    Err(parse_error(
        path,
        line,
        format!("field '{field_name}' must be an integer"),
    ))
}

fn parse_required_u32(
    value: Option<&Value>,
    path: &Path,
    line: usize,
    field_name: &str,
) -> Result<u32, PanlabelError> {
    let value = required_value(value, path, line, field_name)?;
    let unsigned = value.as_u64().ok_or_else(|| {
        parse_error(
            path,
            line,
            format!("field '{field_name}' must be an unsigned integer"),
        )
    })?;
    u32::try_from(unsigned)
        .map_err(|_| parse_error(path, line, format!("field '{field_name}' is out of range")))
}

fn parse_optional_u32(
    value: Option<&Value>,
    path: &Path,
    line: usize,
    field_name: &str,
) -> Result<Option<u32>, PanlabelError> {
    let Some(value) = value else {
        return Ok(None);
    };
    parse_required_u32(Some(value), path, line, field_name).map(Some)
}

fn parse_finite_f64(
    value: &Value,
    path: &Path,
    line: usize,
    field_name: &str,
) -> Result<f64, PanlabelError> {
    let number = value
        .as_f64()
        .ok_or_else(|| parse_error(path, line, format!("field '{field_name}' must be a number")))?;
    if !number.is_finite() {
        return Err(parse_error(
            path,
            line,
            format!("field '{field_name}' must be finite"),
        ));
    }
    Ok(number)
}

fn optional_string(metadata: &Map<String, Value>, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn derive_file_name(source_ref: &str) -> String {
    let without_fragment = source_ref.split('#').next().unwrap_or(source_ref);
    let clean = without_fragment
        .split('?')
        .next()
        .unwrap_or(without_fragment);
    if let Some(rest) = clean.strip_prefix("s3://") {
        if let Some((_, key)) = rest.split_once('/') {
            return key.to_string();
        }
    }
    clean.to_string()
}

fn parse_error(path: &Path, line: usize, message: impl Into<String>) -> PanlabelError {
    PanlabelError::SageMakerManifestParse {
        path: path.to_path_buf(),
        line,
        message: message.into(),
    }
}

fn write_error(path: &Path, message: impl Into<String>) -> PanlabelError {
    PanlabelError::SageMakerManifestWrite {
        path: PathBuf::from(path),
        message: message.into(),
    }
}
