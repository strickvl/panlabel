//! Supervisely JSON reader and writer.
//!
//! This adapter supports Supervisely per-image annotation JSON files plus the
//! common dataset/project layouts (`ann/` directories and project `meta.json`).
//! Since panlabel IR stores detection bboxes, polygon geometry is converted to
//! an axis-aligned bbox envelope and marked with attributes.

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
    reject_unsafe_relative_path, required_u32, scalar_to_string,
};

pub const ATTR_DATASET: &str = "supervisely_dataset";
pub const ATTR_ANN_PATH: &str = "supervisely_ann_path";
pub const ATTR_GEOMETRY_TYPE: &str = "supervisely_geometry_type";
pub const ATTR_OBJECT_ID: &str = "supervisely_object_id";

const IMG_README: &str = "This directory is a placeholder. Panlabel does not copy image files during conversion.\nPlace your original images here to complete the Supervisely dataset layout.\n";
const MEMORY_PATH: &str = "<memory>";

#[derive(Debug)]
struct ParsedFile {
    source_path: PathBuf,
    ann_rel_path: Option<String>,
    dataset_name: Option<String>,
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

/// Reads Supervisely annotation JSON, dataset directory, or project directory into IR.
pub fn read_supervisely_json(path: &Path) -> Result<Dataset, PanlabelError> {
    if path.is_file() {
        read_single_file(path)
    } else if path.is_dir() {
        read_directory(path)
    } else {
        Err(invalid(
            path,
            "path must be a Supervisely JSON file or directory",
        ))
    }
}

/// Writes panlabel IR as Supervisely JSON.
///
/// A `.json` output path writes one annotation JSON and requires exactly one
/// image. Any other output path writes a canonical minimal Supervisely project:
/// `meta.json`, `dataset/ann/<image-file-name>.json`, and `dataset/img/README.txt`.
pub fn write_supervisely_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let is_json_ext = has_json_extension(path);
    if is_json_ext && dataset.images.len() == 1 {
        write_single_file(path, dataset)
    } else if is_json_ext {
        Err(invalid(
            path,
            format!(
                "single-file Supervisely output requires exactly 1 image, got {}; use a directory path instead",
                dataset.images.len()
            ),
        ))
    } else {
        write_project_directory(path, dataset)
    }
}

/// Parses a single Supervisely annotation JSON string into IR.
pub fn from_supervisely_str(json: &str) -> Result<Dataset, PanlabelError> {
    let path = Path::new(MEMORY_PATH);
    let value: Value =
        serde_json::from_str(json).map_err(|source| PanlabelError::SuperviselyJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let parsed = parse_annotation_file(&value, path, None, None, None)?;
    dataset_from_parsed(vec![parsed], BTreeSet::new())
}

/// Parses a single Supervisely annotation JSON byte slice into IR.
pub fn from_supervisely_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    let path = Path::new(MEMORY_PATH);
    let value: Value =
        serde_json::from_slice(bytes).map_err(|source| PanlabelError::SuperviselyJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let parsed = parse_annotation_file(&value, path, None, None, None)?;
    dataset_from_parsed(vec![parsed], BTreeSet::new())
}

/// Serializes a single-image dataset to a Supervisely annotation JSON string.
pub fn to_supervisely_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let memory_path = Path::new(MEMORY_PATH);
    if dataset.images.len() != 1 {
        return Err(invalid(
            memory_path,
            format!(
                "single-file Supervisely output requires exactly 1 image, got {}",
                dataset.images.len()
            ),
        ));
    }
    validate_dataset_for_write(dataset, memory_path)?;
    let value = image_to_supervisely_value(dataset, &dataset.images[0], memory_path)?;
    serde_json::to_string_pretty(&value).map_err(|source| PanlabelError::SuperviselyJsonWrite {
        path: PathBuf::from(MEMORY_PATH),
        source,
    })
}

fn read_single_file(path: &Path) -> Result<Dataset, PanlabelError> {
    let contents = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let value: Value =
        serde_json::from_str(&contents).map_err(|source| PanlabelError::SuperviselyJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let parsed = parse_annotation_file(&value, path, None, None, None)?;
    dataset_from_parsed(vec![parsed], BTreeSet::new())
}

fn read_directory(path: &Path) -> Result<Dataset, PanlabelError> {
    let class_names = read_meta_classes(path)?;
    let mut parsed_files = Vec::new();

    if path.join("ann").is_dir() {
        let dataset_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string);
        parsed_files.extend(read_ann_directory(
            &path.join("ann"),
            dataset_name,
            path,
            false,
        )?);
    } else if path.join("meta.json").is_file() {
        for entry in fs::read_dir(path).map_err(PanlabelError::Io)? {
            let entry = entry.map_err(PanlabelError::Io)?;
            let dataset_path = entry.path();
            if !dataset_path.is_dir() || !dataset_path.join("ann").is_dir() {
                continue;
            }
            let dataset_name = dataset_path
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string);
            parsed_files.extend(read_ann_directory(
                &dataset_path.join("ann"),
                dataset_name,
                path,
                true,
            )?);
        }
    } else {
        return Err(invalid(
            path,
            "Supervisely directory must contain an ann/ directory or project meta.json with dataset ann/ directories",
        ));
    }

    if parsed_files.is_empty() {
        return Err(invalid(
            path,
            "no Supervisely annotation JSON files found in directory",
        ));
    }

    dataset_from_parsed(parsed_files, class_names)
}

fn read_ann_directory(
    ann_dir: &Path,
    dataset_name: Option<String>,
    project_root: &Path,
    include_dataset_in_image_name: bool,
) -> Result<Vec<ParsedFile>, PanlabelError> {
    let mut parsed = Vec::new();
    for entry in walkdir::WalkDir::new(ann_dir).follow_links(true) {
        let entry = entry.map_err(|source| {
            invalid(
                project_root,
                format!("failed while traversing directory: {source}"),
            )
        })?;
        let path = entry.path();
        if !path.is_file() || !has_json_extension(path) {
            continue;
        }
        let contents = fs::read_to_string(path).map_err(PanlabelError::Io)?;
        let value: Value = serde_json::from_str(&contents).map_err(|source| {
            PanlabelError::SuperviselyJsonParse {
                path: path.to_path_buf(),
                source,
            }
        })?;
        if !looks_like_supervisely_annotation(&value) {
            continue;
        }
        let ann_rel_path = path
            .strip_prefix(project_root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let image_name = derive_image_name_from_ann_relative_path(
            path.strip_prefix(ann_dir).unwrap_or(path),
            dataset_name.as_deref(),
            include_dataset_in_image_name,
        );
        parsed.push(parse_annotation_file(
            &value,
            path,
            dataset_name.clone(),
            Some(ann_rel_path),
            Some(image_name),
        )?);
    }
    parsed.sort_by(|a, b| a.image_name.cmp(&b.image_name));
    Ok(parsed)
}

fn parse_annotation_file(
    value: &Value,
    path: &Path,
    dataset_name: Option<String>,
    ann_rel_path: Option<String>,
    image_name: Option<String>,
) -> Result<ParsedFile, PanlabelError> {
    let root = value
        .as_object()
        .ok_or_else(|| invalid(path, "Supervisely annotation must be a JSON object"))?;
    let size = root
        .get("size")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(path, "missing required object field 'size'"))?;
    let width = required_u32(size.get("width"), path, "size.width", invalid_common)?;
    let height = required_u32(size.get("height"), path, "size.height", invalid_common)?;
    let objects = root
        .get("objects")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(path, "missing required array field 'objects'"))?;

    let mut annotations = Vec::with_capacity(objects.len());
    for (idx, object) in objects.iter().enumerate() {
        annotations.push(parse_object(object, idx, path)?);
    }

    Ok(ParsedFile {
        source_path: path.to_path_buf(),
        ann_rel_path,
        dataset_name,
        image_name: image_name.unwrap_or_else(|| derive_image_name_from_ann_path(path)),
        width,
        height,
        annotations,
    })
}

fn parse_object(value: &Value, idx: usize, path: &Path) -> Result<ParsedAnnotation, PanlabelError> {
    let obj = value
        .as_object()
        .ok_or_else(|| invalid(path, format!("objects[{idx}] must be an object")))?;
    let label = obj
        .get("classTitle")
        .and_then(Value::as_str)
        .filter(|label| !label.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            invalid(
                path,
                format!("objects[{idx}] missing non-empty 'classTitle'"),
            )
        })?;
    let geometry_type = obj
        .get("geometryType")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(path, format!("objects[{idx}] missing 'geometryType'")))?;

    let bbox = match geometry_type {
        "rectangle" => {
            let points = object_exterior_points(obj, idx, path)?;
            if points.len() != 2 {
                return Err(invalid(
                    path,
                    format!("objects[{idx}] rectangle must contain exactly 2 exterior points"),
                ));
            }
            envelope(&points)
        }
        "polygon" => {
            let points = object_exterior_points(obj, idx, path)?;
            if points.len() < 3 {
                return Err(invalid(
                    path,
                    format!("objects[{idx}] polygon must contain at least 3 exterior points"),
                ));
            }
            envelope(&points)
        }
        other => {
            return Err(invalid(
                path,
                format!(
                    "objects[{idx}] unsupported Supervisely geometryType '{other}'; supported: rectangle, polygon"
                ),
            ));
        }
    };

    let mut attributes = BTreeMap::new();
    attributes.insert(ATTR_GEOMETRY_TYPE.to_string(), geometry_type.to_string());
    if let Some(id) = scalar_to_string(obj.get("id")) {
        attributes.insert(ATTR_OBJECT_ID.to_string(), id);
    }

    let confidence = optional_finite_f64(
        obj.get("confidence").or_else(|| obj.get("score")),
        path,
        format!("objects[{idx}].confidence"),
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
        if let Some(dataset_name) = &parsed.dataset_name {
            image
                .attributes
                .insert(ATTR_DATASET.to_string(), dataset_name.clone());
        }
        if let Some(ann_rel_path) = &parsed.ann_rel_path {
            image
                .attributes
                .insert(ATTR_ANN_PATH.to_string(), ann_rel_path.clone());
        }
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
    let value = image_to_supervisely_value(dataset, &dataset.images[0], path)?;
    let file = fs::File::create(path).map_err(PanlabelError::Io)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &value).map_err(|source| {
        PanlabelError::SuperviselyJsonWrite {
            path: path.to_path_buf(),
            source,
        }
    })
}

fn write_project_directory(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    validate_dataset_for_write(dataset, path)?;
    fs::create_dir_all(path).map_err(PanlabelError::Io)?;
    let dataset_dir = path.join("dataset");
    let ann_dir = dataset_dir.join("ann");
    let img_dir = dataset_dir.join("img");
    fs::create_dir_all(&ann_dir).map_err(PanlabelError::Io)?;
    fs::create_dir_all(&img_dir).map_err(PanlabelError::Io)?;
    fs::write(img_dir.join("README.txt"), IMG_README).map_err(PanlabelError::Io)?;

    let meta = supervisely_meta_value(dataset);
    let meta_path = path.join("meta.json");
    let meta_file = fs::File::create(&meta_path).map_err(PanlabelError::Io)?;
    serde_json::to_writer_pretty(BufWriter::new(meta_file), &meta).map_err(|source| {
        PanlabelError::SuperviselyJsonWrite {
            path: meta_path,
            source,
        }
    })?;

    let mut images: Vec<&Image> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    for image in images {
        let ann_path = ann_dir.join(supervisely_annotation_rel_path(&image.file_name, &ann_dir)?);
        if let Some(parent) = ann_path.parent() {
            fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
        }
        let value = image_to_supervisely_value(dataset, image, &ann_path)?;
        let file = fs::File::create(&ann_path).map_err(PanlabelError::Io)?;
        serde_json::to_writer_pretty(BufWriter::new(file), &value).map_err(|source| {
            PanlabelError::SuperviselyJsonWrite {
                path: ann_path.clone(),
                source,
            }
        })?;
    }
    Ok(())
}

fn image_to_supervisely_value(
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

    let mut objects = Vec::with_capacity(anns.len());
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
        let class_title = category_names.get(&ann.category_id).ok_or_else(|| {
            invalid(
                path,
                format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            )
        })?;
        let mut object = Map::new();
        object.insert("id".to_string(), json!(ann.id.as_u64()));
        object.insert("classTitle".to_string(), json!(class_title));
        object.insert("geometryType".to_string(), json!("rectangle"));
        object.insert(
            "geometry".to_string(),
            json!({
                "points": {
                    "exterior": [[ann.bbox.xmin(), ann.bbox.ymin()], [ann.bbox.xmax(), ann.bbox.ymax()]],
                    "interior": []
                }
            }),
        );
        objects.push(Value::Object(object));
    }

    Ok(json!({
        "description": "",
        "tags": [],
        "size": {
            "width": image.width,
            "height": image.height
        },
        "objects": objects
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
        let rel_path = supervisely_annotation_rel_path(&image.file_name, path)?;
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

fn supervisely_annotation_rel_path(file_name: &str, path: &Path) -> Result<PathBuf, PanlabelError> {
    reject_unsafe_relative_path(file_name, path, invalid_common)?;
    Ok(PathBuf::from(format!("{file_name}.json")))
}

fn supervisely_meta_value(dataset: &Dataset) -> Value {
    let mut categories: Vec<&Category> = dataset.categories.iter().collect();
    categories.sort_by(|a, b| a.name.cmp(&b.name));
    let classes: Vec<Value> = categories
        .into_iter()
        .enumerate()
        .map(|(idx, category)| {
            json!({
                "title": category.name,
                "shape": "rectangle",
                "color": deterministic_color(idx)
            })
        })
        .collect();
    json!({
        "classes": classes,
        "tags": []
    })
}

fn deterministic_color(idx: usize) -> String {
    const COLORS: [&str; 8] = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ];
    COLORS[idx % COLORS.len()].to_string()
}

fn read_meta_classes(root: &Path) -> Result<BTreeSet<String>, PanlabelError> {
    let meta_path = root.join("meta.json");
    if !meta_path.is_file() {
        return Ok(BTreeSet::new());
    }
    let contents = fs::read_to_string(&meta_path).map_err(PanlabelError::Io)?;
    let value: Value =
        serde_json::from_str(&contents).map_err(|source| PanlabelError::SuperviselyJsonParse {
            path: meta_path.clone(),
            source,
        })?;
    Ok(parse_meta_class_names(&value))
}

fn parse_meta_class_names(value: &Value) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let Some(classes) = value.get("classes").and_then(Value::as_array) else {
        return names;
    };
    for class_value in classes {
        if let Some(title) = class_value
            .get("title")
            .and_then(Value::as_str)
            .filter(|title| !title.trim().is_empty())
        {
            names.insert(title.to_string());
        }
    }
    names
}

fn looks_like_supervisely_annotation(value: &Value) -> bool {
    value.get("size").is_some_and(Value::is_object)
        && value.get("objects").is_some_and(Value::is_array)
}

fn object_exterior_points(
    obj: &Map<String, Value>,
    idx: usize,
    path: &Path,
) -> Result<Vec<[f64; 2]>, PanlabelError> {
    let exterior = obj
        .get("geometry")
        .and_then(|geometry| geometry.get("points"))
        .and_then(|points| points.get("exterior"))
        .ok_or_else(|| {
            invalid(
                path,
                format!("objects[{idx}] missing geometry.points.exterior"),
            )
        })?;
    point_pairs(
        exterior,
        path,
        format!("objects[{idx}].geometry.points.exterior"),
    )
}

fn point_pairs(value: &Value, path: &Path, field: String) -> Result<Vec<[f64; 2]>, PanlabelError> {
    let array = value
        .as_array()
        .ok_or_else(|| invalid(path, format!("{field} must be an array of [x, y] pairs")))?;
    array
        .iter()
        .enumerate()
        .map(|(idx, point)| {
            parse_point_pair(point, path, format!("{field}[{idx}]"), invalid_common)
        })
        .collect()
}

fn derive_image_name_from_ann_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && *name != MEMORY_PATH)
        .unwrap_or("image")
        .to_string()
}

fn derive_image_name_from_ann_relative_path(
    relative_ann_path: &Path,
    dataset_name: Option<&str>,
    include_dataset: bool,
) -> String {
    let mut without_json = relative_ann_path.to_path_buf();
    without_json.set_extension("");
    let rel = without_json.to_string_lossy().replace('\\', "/");
    if include_dataset {
        if let Some(dataset_name) = dataset_name.filter(|name| !name.is_empty()) {
            return format!("{dataset_name}/{rel}");
        }
    }
    rel
}

fn invalid_common(path: &Path, message: String) -> PanlabelError {
    invalid(path, message)
}

fn invalid(path: &Path, message: impl Into<String>) -> PanlabelError {
    PanlabelError::SuperviselyLayoutInvalid {
        path: path.to_path_buf(),
        message: message.into(),
    }
}
