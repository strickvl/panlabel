//! SuperAnnotate JSON reader and writer.
//!
//! This adapter supports SuperAnnotate-style per-image annotation JSON files
//! with a top-level `metadata` object and `instances` array. The IR only stores
//! axis-aligned detection boxes, so polygon/oriented-box instances are converted
//! to their axis-aligned bbox envelope and marked with attributes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use super::model::{Annotation, Category, Dataset, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

use super::io_super_json_common::{
    envelope, has_json_extension, optional_finite_f64, parse_point_pair,
    reject_unsafe_relative_path, required_f64, required_u32, scalar_to_string,
};

pub const ATTR_IMAGE_NAME: &str = "superannotate_image_name";
pub const ATTR_GEOMETRY_TYPE: &str = "superannotate_geometry_type";
pub const ATTR_INSTANCE_ID: &str = "superannotate_instance_id";

const IMAGES_README: &str = "This directory is a placeholder. Panlabel does not copy image files during conversion.\nPlace your original images here to complete the SuperAnnotate dataset layout.\n";
const MEMORY_PATH: &str = "<memory>";

#[derive(Debug)]
struct ParsedFile {
    source_path: PathBuf,
    image_name: String,
    width: u32,
    height: u32,
    annotations: Vec<ParsedAnnotation>,
}

#[derive(Debug)]
struct ParsedAnnotation {
    label: String,
    bbox: BBoxXYXY<Pixel>,
    confidence: Option<f64>,
    attributes: BTreeMap<String, String>,
}

/// Reads a SuperAnnotate annotation JSON file or directory into panlabel IR.
pub fn read_superannotate_json(path: &Path) -> Result<Dataset, PanlabelError> {
    if path.is_file() {
        read_single_file(path)
    } else if path.is_dir() {
        read_directory(path)
    } else {
        Err(invalid(
            path,
            "path must be a SuperAnnotate JSON file or directory",
        ))
    }
}

/// Writes panlabel IR as SuperAnnotate JSON.
///
/// A `.json` output path writes one annotation file and requires exactly one
/// image. Any other output path writes a canonical directory layout:
/// `annotations/<image-stem>.json`, `classes/classes.json`, and an
/// `images/README.txt` placeholder.
pub fn write_superannotate_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let is_json_ext = has_json_extension(path);
    if is_json_ext && dataset.images.len() == 1 {
        write_single_file(path, dataset)
    } else if is_json_ext {
        Err(invalid(
            path,
            format!(
                "single-file SuperAnnotate output requires exactly 1 image, got {}; use a directory path instead",
                dataset.images.len()
            ),
        ))
    } else {
        write_directory(path, dataset)
    }
}

/// Parses a single SuperAnnotate annotation JSON string into IR.
pub fn from_superannotate_str(json: &str) -> Result<Dataset, PanlabelError> {
    let path = Path::new(MEMORY_PATH);
    let value: Value =
        serde_json::from_str(json).map_err(|source| PanlabelError::SuperAnnotateJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let parsed = parse_annotation_file(&value, path)?;
    dataset_from_parsed(vec![parsed], BTreeSet::new(), path)
}

/// Parses a single SuperAnnotate annotation JSON byte slice into IR.
pub fn from_superannotate_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    let path = Path::new(MEMORY_PATH);
    let value: Value =
        serde_json::from_slice(bytes).map_err(|source| PanlabelError::SuperAnnotateJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let parsed = parse_annotation_file(&value, path)?;
    dataset_from_parsed(vec![parsed], BTreeSet::new(), path)
}

/// Serializes a single-image dataset to a SuperAnnotate annotation JSON string.
pub fn to_superannotate_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let memory_path = Path::new(MEMORY_PATH);
    if dataset.images.len() != 1 {
        return Err(invalid(
            memory_path,
            format!(
                "single-file SuperAnnotate output requires exactly 1 image, got {}",
                dataset.images.len()
            ),
        ));
    }
    validate_dataset_for_write(dataset, memory_path)?;
    let value = image_to_superannotate_value(dataset, &dataset.images[0], memory_path)?;
    serde_json::to_string_pretty(&value).map_err(|source| PanlabelError::SuperAnnotateJsonWrite {
        path: PathBuf::from(MEMORY_PATH),
        source,
    })
}

fn read_single_file(path: &Path) -> Result<Dataset, PanlabelError> {
    let contents = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let value: Value = serde_json::from_str(&contents).map_err(|source| {
        PanlabelError::SuperAnnotateJsonParse {
            path: path.to_path_buf(),
            source,
        }
    })?;
    let parsed = parse_annotation_file(&value, path)?;
    dataset_from_parsed(vec![parsed], BTreeSet::new(), path)
}

fn read_directory(path: &Path) -> Result<Dataset, PanlabelError> {
    let class_names = read_optional_classes(path)?;
    let annotation_dir = path.join("annotations");
    let base_dir = if annotation_dir.is_dir() {
        annotation_dir.as_path()
    } else {
        path
    };

    let mut parsed_files = Vec::new();
    for entry in walkdir::WalkDir::new(base_dir).follow_links(true) {
        let entry = entry.map_err(|source| {
            invalid(path, format!("failed while traversing directory: {source}"))
        })?;
        let json_path = entry.path();
        if !json_path.is_file() || !has_json_extension(json_path) || is_classes_json(json_path) {
            continue;
        }
        let contents = fs::read_to_string(json_path).map_err(PanlabelError::Io)?;
        let value: Value = match serde_json::from_str(&contents) {
            Ok(value) => value,
            Err(source) => {
                return Err(PanlabelError::SuperAnnotateJsonParse {
                    path: json_path.to_path_buf(),
                    source,
                });
            }
        };
        if !looks_like_superannotate_annotation(&value) {
            continue;
        }
        parsed_files.push(parse_annotation_file(&value, json_path)?);
    }

    if parsed_files.is_empty() {
        return Err(invalid(
            path,
            "no SuperAnnotate annotation JSON files found in directory",
        ));
    }

    dataset_from_parsed(parsed_files, class_names, path)
}

fn parse_annotation_file(value: &Value, path: &Path) -> Result<ParsedFile, PanlabelError> {
    let root = value
        .as_object()
        .ok_or_else(|| invalid(path, "SuperAnnotate annotation must be a JSON object"))?;
    let metadata = root
        .get("metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(path, "missing required object field 'metadata'"))?;
    let width = required_u32(
        metadata.get("width"),
        path,
        "metadata.width",
        invalid_common,
    )?;
    let height = required_u32(
        metadata.get("height"),
        path,
        "metadata.height",
        invalid_common,
    )?;
    let image_name = metadata
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| derive_superannotate_image_name(path));

    let instances = root
        .get("instances")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(path, "missing required array field 'instances'"))?;

    let mut annotations = Vec::with_capacity(instances.len());
    for (idx, instance) in instances.iter().enumerate() {
        annotations.push(parse_instance(instance, idx, path)?);
    }

    Ok(ParsedFile {
        source_path: path.to_path_buf(),
        image_name,
        width,
        height,
        annotations,
    })
}

fn parse_instance(
    value: &Value,
    idx: usize,
    path: &Path,
) -> Result<ParsedAnnotation, PanlabelError> {
    let obj = value
        .as_object()
        .ok_or_else(|| invalid(path, format!("instances[{idx}] must be an object")))?;
    let label = obj
        .get("className")
        .or_else(|| obj.get("class_name"))
        .and_then(Value::as_str)
        .filter(|label| !label.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            invalid(
                path,
                format!("instances[{idx}] missing non-empty 'className'"),
            )
        })?;
    let geometry_type = obj
        .get("type")
        .or_else(|| obj.get("geometryType"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| infer_instance_type(obj));
    let (bbox, stored_geometry_type) = match geometry_type {
        "bbox" | "bounding_box" | "rectangle" => {
            let points = obj
                .get("points")
                .ok_or_else(|| invalid(path, format!("instances[{idx}] missing 'points'")))?;
            (bbox_from_bbox_points(points, path, idx)?, "bbox")
        }
        "polygon" | "rotated_bbox" | "rotated_box" | "oriented_bbox" | "oriented_box" => {
            let points = obj
                .get("points")
                .ok_or_else(|| invalid(path, format!("instances[{idx}] missing 'points'")))?;
            (bbox_envelope_from_points(points, path, idx)?, geometry_type)
        }
        other => {
            return Err(invalid(
                path,
                format!(
                    "instances[{idx}] unsupported SuperAnnotate geometry type '{other}'; supported: bbox, polygon, rotated_bbox/oriented_bbox"
                ),
            ));
        }
    };

    let mut attributes = BTreeMap::new();
    attributes.insert(
        ATTR_GEOMETRY_TYPE.to_string(),
        stored_geometry_type.to_string(),
    );
    if let Some(id) = scalar_to_string(obj.get("id").or_else(|| obj.get("uuid"))) {
        attributes.insert(ATTR_INSTANCE_ID.to_string(), id);
    }
    preserve_superannotate_attributes(obj.get("attributes"), &mut attributes);

    let confidence = optional_finite_f64(
        obj.get("probability").or_else(|| obj.get("confidence")),
        path,
        format!("instances[{idx}].probability"),
        invalid_common,
    )?;

    Ok(ParsedAnnotation {
        label,
        bbox,
        confidence,
        attributes,
    })
}

fn dataset_from_parsed(
    mut parsed_files: Vec<ParsedFile>,
    mut class_names: BTreeSet<String>,
    path: &Path,
) -> Result<Dataset, PanlabelError> {
    parsed_files.sort_by(|a, b| a.image_name.cmp(&b.image_name));

    let mut seen = BTreeSet::new();
    for parsed in &parsed_files {
        if !seen.insert(parsed.image_name.clone()) {
            return Err(invalid(
                &parsed.source_path,
                format!("duplicate derived image name: '{}'", parsed.image_name),
            ));
        }
        for ann in &parsed.annotations {
            if ann.label.trim().is_empty() {
                return Err(invalid(path, "empty SuperAnnotate className"));
            }
            class_names.insert(ann.label.clone());
        }
    }

    let categories: Vec<Category> = class_names
        .iter()
        .enumerate()
        .map(|(idx, name)| Category::new((idx + 1) as u64, name.clone()))
        .collect();
    let label_to_category: BTreeMap<String, CategoryId> = categories
        .iter()
        .map(|category| (category.name.clone(), category.id))
        .collect();

    let mut images = Vec::with_capacity(parsed_files.len());
    let mut annotations = Vec::new();
    let mut next_ann_id = 1u64;

    for (image_idx, parsed) in parsed_files.iter().enumerate() {
        let image_id = ImageId::new((image_idx + 1) as u64);
        let mut image = Image::new(
            image_id,
            parsed.image_name.clone(),
            parsed.width,
            parsed.height,
        );
        image
            .attributes
            .insert(ATTR_IMAGE_NAME.to_string(), parsed.image_name.clone());
        images.push(image);

        for parsed_ann in &parsed.annotations {
            let category_id = label_to_category[&parsed_ann.label];
            let mut ann = Annotation::new(
                AnnotationId::new(next_ann_id),
                image_id,
                category_id,
                parsed_ann.bbox,
            );
            ann.confidence = parsed_ann.confidence;
            ann.attributes = parsed_ann.attributes.clone();
            annotations.push(ann);
            next_ann_id += 1;
        }
    }

    Ok(Dataset {
        images,
        categories,
        annotations,
        ..Default::default()
    })
}

fn write_single_file(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    validate_dataset_for_write(dataset, path)?;
    let value = image_to_superannotate_value(dataset, &dataset.images[0], path)?;
    let file = fs::File::create(path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &value).map_err(|source| {
        PanlabelError::SuperAnnotateJsonWrite {
            path: path.to_path_buf(),
            source,
        }
    })
}

fn write_directory(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    validate_dataset_for_write(dataset, path)?;
    fs::create_dir_all(path).map_err(PanlabelError::Io)?;
    let annotations_dir = path.join("annotations");
    let images_dir = path.join("images");
    let classes_dir = path.join("classes");
    fs::create_dir_all(&annotations_dir).map_err(PanlabelError::Io)?;
    fs::create_dir_all(&images_dir).map_err(PanlabelError::Io)?;
    fs::create_dir_all(&classes_dir).map_err(PanlabelError::Io)?;
    fs::write(images_dir.join("README.txt"), IMAGES_README).map_err(PanlabelError::Io)?;

    let classes = superannotate_classes_value(dataset);
    let classes_file =
        fs::File::create(classes_dir.join("classes.json")).map_err(PanlabelError::Io)?;
    serde_json::to_writer_pretty(BufWriter::new(classes_file), &classes).map_err(|source| {
        PanlabelError::SuperAnnotateJsonWrite {
            path: classes_dir.join("classes.json"),
            source,
        }
    })?;

    let mut images: Vec<&Image> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    for image in images {
        let json_path = annotations_dir.join(superannotate_annotation_rel_path(
            &image.file_name,
            &annotations_dir,
        )?);
        if let Some(parent) = json_path.parent() {
            fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
        }
        let value = image_to_superannotate_value(dataset, image, &json_path)?;
        let file = fs::File::create(&json_path).map_err(PanlabelError::Io)?;
        serde_json::to_writer_pretty(BufWriter::new(file), &value).map_err(|source| {
            PanlabelError::SuperAnnotateJsonWrite {
                path: json_path.clone(),
                source,
            }
        })?;
    }
    Ok(())
}

fn image_to_superannotate_value(
    dataset: &Dataset,
    image: &Image,
    path: &Path,
) -> Result<Value, PanlabelError> {
    if image.width == 0 || image.height == 0 {
        return Err(invalid(
            path,
            format!("image '{}' has zero width or height", image.file_name),
        ));
    }
    let category_names: BTreeMap<CategoryId, &str> = dataset
        .categories
        .iter()
        .map(|category| (category.id, category.name.as_str()))
        .collect();
    let mut anns: Vec<&Annotation> = dataset
        .annotations
        .iter()
        .filter(|ann| ann.image_id == image.id)
        .collect();
    anns.sort_by_key(|ann| ann.id);

    let mut instances = Vec::with_capacity(anns.len());
    for ann in anns {
        if !ann.bbox.is_finite() || !ann.bbox.is_ordered() {
            return Err(invalid(
                path,
                format!(
                    "annotation {} has a non-finite or unordered bbox",
                    ann.id.as_u64()
                ),
            ));
        }
        if let Some(confidence) = ann.confidence {
            if !confidence.is_finite() {
                return Err(invalid(
                    path,
                    format!("annotation {} has non-finite confidence", ann.id.as_u64()),
                ));
            }
        }
        let class_name = category_names.get(&ann.category_id).ok_or_else(|| {
            invalid(
                path,
                format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            )
        })?;
        let mut instance = Map::new();
        instance.insert("id".to_string(), json!(ann.id.as_u64()));
        instance.insert("type".to_string(), json!("bbox"));
        instance.insert("className".to_string(), json!(class_name));
        instance.insert(
            "points".to_string(),
            json!({
                "x1": ann.bbox.xmin(),
                "y1": ann.bbox.ymin(),
                "x2": ann.bbox.xmax(),
                "y2": ann.bbox.ymax()
            }),
        );
        if let Some(confidence) = ann.confidence {
            instance.insert("probability".to_string(), json!(confidence));
        }
        instances.push(Value::Object(instance));
    }

    Ok(json!({
        "metadata": {
            "name": image.file_name,
            "width": image.width,
            "height": image.height
        },
        "instances": instances
    }))
}

fn validate_dataset_for_write(dataset: &Dataset, path: &Path) -> Result<(), PanlabelError> {
    let image_ids: BTreeSet<ImageId> = dataset.images.iter().map(|image| image.id).collect();
    let category_ids: BTreeSet<CategoryId> = dataset
        .categories
        .iter()
        .map(|category| category.id)
        .collect();

    for ann in &dataset.annotations {
        if !image_ids.contains(&ann.image_id) {
            return Err(invalid(
                path,
                format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            ));
        }
        if !category_ids.contains(&ann.category_id) {
            return Err(invalid(
                path,
                format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            ));
        }
    }

    let mut seen_paths = BTreeSet::new();
    for image in &dataset.images {
        let rel_path = superannotate_annotation_rel_path(&image.file_name, path)?;
        if !seen_paths.insert(rel_path.clone()) {
            return Err(invalid(
                path,
                format!(
                    "multiple images would write to annotation path '{}'",
                    rel_path.display()
                ),
            ));
        }
    }

    Ok(())
}

fn superannotate_annotation_rel_path(
    file_name: &str,
    path: &Path,
) -> Result<PathBuf, PanlabelError> {
    reject_unsafe_relative_path(file_name, path, invalid_common)?;
    let mut rel_path = PathBuf::from(file_name);
    rel_path.set_extension("json");
    Ok(rel_path)
}

fn superannotate_classes_value(dataset: &Dataset) -> Value {
    let mut categories: Vec<&Category> = dataset.categories.iter().collect();
    categories.sort_by(|a, b| a.name.cmp(&b.name));
    let classes: Vec<Value> = categories
        .into_iter()
        .map(|category| json!({ "name": category.name }))
        .collect();
    Value::Array(classes)
}

fn read_optional_classes(root: &Path) -> Result<BTreeSet<String>, PanlabelError> {
    let mut class_names = BTreeSet::new();
    for path in [root.join("classes/classes.json"), root.join("classes.json")] {
        if !path.is_file() {
            continue;
        }
        let contents = fs::read_to_string(&path).map_err(PanlabelError::Io)?;
        let value: Value = serde_json::from_str(&contents).map_err(|source| {
            PanlabelError::SuperAnnotateJsonParse {
                path: path.clone(),
                source,
            }
        })?;
        class_names.extend(parse_class_names(&value));
    }
    Ok(class_names)
}

fn parse_class_names(value: &Value) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let class_values: Vec<&Value> = if let Some(array) = value.as_array() {
        array.iter().collect()
    } else if let Some(array) = value.get("classes").and_then(Value::as_array) {
        array.iter().collect()
    } else {
        Vec::new()
    };
    for class_value in class_values {
        if let Some(name) = class_value
            .get("name")
            .or_else(|| class_value.get("title"))
            .or_else(|| class_value.get("className"))
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
        {
            names.insert(name.to_string());
        }
    }
    names
}

fn looks_like_superannotate_annotation(value: &Value) -> bool {
    value.get("metadata").is_some_and(Value::is_object)
        && value.get("instances").is_some_and(Value::is_array)
}

fn bbox_from_bbox_points(
    points: &Value,
    path: &Path,
    idx: usize,
) -> Result<BBoxXYXY<Pixel>, PanlabelError> {
    let obj = points.as_object().ok_or_else(|| {
        invalid(
            path,
            format!("instances[{idx}].points for bbox must be an object"),
        )
    })?;
    let x1 = required_f64(
        obj.get("x1"),
        path,
        format!("instances[{idx}].points.x1"),
        invalid_common,
    )?;
    let y1 = required_f64(
        obj.get("y1"),
        path,
        format!("instances[{idx}].points.y1"),
        invalid_common,
    )?;
    let x2 = required_f64(
        obj.get("x2"),
        path,
        format!("instances[{idx}].points.x2"),
        invalid_common,
    )?;
    let y2 = required_f64(
        obj.get("y2"),
        path,
        format!("instances[{idx}].points.y2"),
        invalid_common,
    )?;
    Ok(BBoxXYXY::<Pixel>::from_xyxy(
        x1.min(x2),
        y1.min(y2),
        x1.max(x2),
        y1.max(y2),
    ))
}

fn bbox_envelope_from_points(
    points: &Value,
    path: &Path,
    idx: usize,
) -> Result<BBoxXYXY<Pixel>, PanlabelError> {
    let pairs = point_pairs(points, path, format!("instances[{idx}].points"))?;
    if pairs.len() < 2 {
        return Err(invalid(
            path,
            format!("instances[{idx}].points must contain at least 2 points"),
        ));
    }
    Ok(envelope(&pairs))
}

fn point_pairs(points: &Value, path: &Path, field: String) -> Result<Vec<[f64; 2]>, PanlabelError> {
    if let Some(array) = points.as_array() {
        return array
            .iter()
            .enumerate()
            .map(|(idx, point)| {
                parse_point_pair(point, path, format!("{field}[{idx}]"), invalid_common)
            })
            .collect();
    }

    let obj = points.as_object().ok_or_else(|| {
        invalid(
            path,
            format!("{field} must be an array of [x,y] pairs or an object with x/y arrays"),
        )
    })?;
    let xs = obj
        .get("x")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(path, format!("{field}.x must be an array")))?;
    let ys = obj
        .get("y")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(path, format!("{field}.y must be an array")))?;
    if xs.len() != ys.len() {
        return Err(invalid(
            path,
            format!("{field}.x and {field}.y must have the same length"),
        ));
    }
    xs.iter()
        .zip(ys.iter())
        .enumerate()
        .map(|(idx, (x, y))| {
            Ok([
                required_f64(Some(x), path, format!("{field}.x[{idx}]"), invalid_common)?,
                required_f64(Some(y), path, format!("{field}.y[{idx}]"), invalid_common)?,
            ])
        })
        .collect()
}

fn infer_instance_type(obj: &Map<String, Value>) -> &'static str {
    if obj
        .get("points")
        .and_then(Value::as_object)
        .is_some_and(|points| {
            points.contains_key("x1")
                && points.contains_key("y1")
                && points.contains_key("x2")
                && points.contains_key("y2")
        })
    {
        "bbox"
    } else {
        "polygon"
    }
}

fn preserve_superannotate_attributes(
    value: Option<&Value>,
    attributes: &mut BTreeMap<String, String>,
) {
    let Some(array) = value.and_then(Value::as_array) else {
        return;
    };
    for item in array {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let name = obj
            .get("name")
            .or_else(|| obj.get("value"))
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty());
        let group = obj
            .get("groupName")
            .or_else(|| obj.get("group"))
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty());
        if let Some(name) = name {
            let key_part = group
                .unwrap_or(name)
                .replace(|ch: char| !ch.is_ascii_alphanumeric(), "_");
            attributes.insert(format!("superannotate_attr_{key_part}"), name.to_string());
        }
    }
}

fn derive_superannotate_image_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && *name != MEMORY_PATH)
        .unwrap_or("image")
        .to_string()
}

fn is_classes_json(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("classes.json"))
}

fn invalid_common(path: &Path, message: String) -> PanlabelError {
    invalid(path, message)
}

fn invalid(path: &Path, message: impl Into<String>) -> PanlabelError {
    PanlabelError::SuperAnnotateLayoutInvalid {
        path: path.to_path_buf(),
        message: message.into(),
    }
}
