//! Cityscapes polygon JSON reader and writer.
//!
//! Cityscapes stores one polygon JSON file per image, typically under
//! `gtFine/<split>/<city>/*_gtFine_polygons.json`. Panlabel's IR is
//! detection-oriented, so this adapter converts each kept polygon into its
//! axis-aligned bbox envelope and records Cityscapes provenance in attributes.
//!
//! The reader supports:
//! - a single `*_gtFine_polygons.json` file,
//! - a directory containing Cityscapes polygon JSON files,
//! - a dataset root containing `gtFine/`, and
//! - a `gtFine/` root directly.
//!
//! The writer emits minimal deterministic rectangle-polygon JSON files and does
//! not copy image binaries.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::BufWriter;
use std::path::{Component, Path, PathBuf};

use serde_json::{json, Map, Value};

use super::io_super_json_common::{
    envelope, has_json_extension, parse_point_pair, reject_unsafe_relative_path, required_u32,
};
use super::model::{Annotation, Category, Dataset, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

pub const ATTR_ANN_PATH: &str = "cityscapes_ann_path";
pub const ATTR_SPLIT: &str = "cityscapes_split";
pub const ATTR_CITY: &str = "cityscapes_city";
pub const ATTR_ORIGINAL_LABEL: &str = "cityscapes_original_label";
pub const ATTR_BBOX_SOURCE: &str = "cityscapes_bbox_source";
pub const ATTR_IS_GROUP: &str = "cityscapes_is_group";
pub const ATTR_LABEL_STATUS: &str = "cityscapes_label_status";
pub const BBOX_SOURCE_POLYGON_ENVELOPE: &str = "polygon_envelope";

const MEMORY_PATH: &str = "<memory>";
const IMAGES_README: &str = "This directory is a placeholder. Panlabel does not copy image files during conversion.\nPlace your original Cityscapes leftImg8bit images here if you need a complete dataset tree.\n";

const INSTANCE_LABELS: &[&str] = &[
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
];

const SKIPPED_LABELS: &[&str] = &[
    "unlabeled",
    "ego vehicle",
    "rectification border",
    "out of roi",
    "static",
    "dynamic",
    "ground",
    "road",
    "sidewalk",
    "parking",
    "rail track",
    "building",
    "wall",
    "fence",
    "guard rail",
    "bridge",
    "tunnel",
    "pole",
    "polegroup",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "caravan",
    "trailer",
    "license plate",
];

#[derive(Debug)]
struct ParsedFile {
    source_path: PathBuf,
    ann_rel_path: Option<String>,
    split: Option<String>,
    city: Option<String>,
    image_name: String,
    width: u32,
    height: u32,
    annotations: Vec<ParsedAnnotation>,
}

#[derive(Debug)]
struct ParsedAnnotation {
    label: String,
    bbox: BBoxXYXY<Pixel>,
    attributes: BTreeMap<String, String>,
}

#[derive(Debug)]
struct LabelDecision {
    label: String,
    original_label: String,
    is_group: bool,
    is_unknown: bool,
}

/// Reads Cityscapes polygon JSON from a file, directory, dataset root, or gtFine root.
pub fn read_cityscapes_json(path: &Path) -> Result<Dataset, PanlabelError> {
    if path.is_file() {
        read_single_file(path)
    } else if path.is_dir() {
        read_directory(path)
    } else {
        Err(invalid(
            path,
            "path must be a Cityscapes polygon JSON file or directory",
        ))
    }
}

/// Writes panlabel IR as minimal Cityscapes polygon JSON.
///
/// A `.json` output path writes one annotation file and requires exactly one
/// image. Any other path writes `gtFine/<split>/<city>/*_gtFine_polygons.json`
/// files plus a placeholder `leftImg8bit/README.txt`; image binaries are not
/// copied.
pub fn write_cityscapes_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let is_json_ext = has_json_extension(path);
    if is_json_ext && dataset.images.len() == 1 {
        write_single_file(path, dataset)
    } else if is_json_ext {
        Err(invalid(
            path,
            format!(
                "single-file Cityscapes output requires exactly 1 image, got {}; use a directory path instead",
                dataset.images.len()
            ),
        ))
    } else {
        write_directory(path, dataset)
    }
}

/// Parses a single Cityscapes polygon JSON string into IR.
pub fn from_cityscapes_str(json: &str) -> Result<Dataset, PanlabelError> {
    let path = Path::new(MEMORY_PATH);
    let value: Value =
        serde_json::from_str(json).map_err(|source| PanlabelError::CityscapesJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let parsed = parse_annotation_file(&value, path, None, None, None, None)?;
    dataset_from_parsed(vec![parsed])
}

/// Parses a single Cityscapes polygon JSON byte slice into IR.
pub fn from_cityscapes_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    let path = Path::new(MEMORY_PATH);
    let value: Value =
        serde_json::from_slice(bytes).map_err(|source| PanlabelError::CityscapesJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let parsed = parse_annotation_file(&value, path, None, None, None, None)?;
    dataset_from_parsed(vec![parsed])
}

/// Serializes a single-image dataset to a Cityscapes polygon JSON string.
pub fn to_cityscapes_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let path = Path::new(MEMORY_PATH);
    if dataset.images.len() != 1 {
        return Err(invalid(
            path,
            format!(
                "single-file Cityscapes output requires exactly 1 image, got {}",
                dataset.images.len()
            ),
        ));
    }
    validate_dataset_for_write(dataset, path)?;
    let value = image_to_cityscapes_value(dataset, &dataset.images[0], path)?;
    serde_json::to_string_pretty(&value).map_err(|source| PanlabelError::CityscapesJsonWrite {
        path: PathBuf::from(MEMORY_PATH),
        source,
    })
}

/// Quick structural check used by CLI autodetection.
pub fn is_likely_cityscapes_file(value: &Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    obj.get("imgWidth").and_then(Value::as_u64).is_some()
        && obj.get("imgHeight").and_then(Value::as_u64).is_some()
        && obj.get("objects").and_then(Value::as_array).is_some()
}

fn read_single_file(path: &Path) -> Result<Dataset, PanlabelError> {
    let contents = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let value: Value =
        serde_json::from_str(&contents).map_err(|source| PanlabelError::CityscapesJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let parsed = parse_annotation_file(&value, path, None, None, None, None)?;
    dataset_from_parsed(vec![parsed])
}

fn read_directory(path: &Path) -> Result<Dataset, PanlabelError> {
    let has_gt_fine_dir = path.join("gtFine").is_dir();
    let search_root = if has_gt_fine_dir {
        path.join("gtFine")
    } else {
        path.to_path_buf()
    };
    let dataset_root = if has_gt_fine_dir {
        path.to_path_buf()
    } else {
        search_root.clone()
    };

    let mut parsed_files = Vec::new();
    for ann_path in collect_cityscapes_files(&search_root)? {
        let contents = fs::read_to_string(&ann_path).map_err(PanlabelError::Io)?;
        let value: Value = serde_json::from_str(&contents).map_err(|source| {
            PanlabelError::CityscapesJsonParse {
                path: ann_path.clone(),
                source,
            }
        })?;
        if !is_likely_cityscapes_file(&value) {
            continue;
        }
        let rel_to_dataset = ann_path
            .strip_prefix(&dataset_root)
            .unwrap_or(&ann_path)
            .to_string_lossy()
            .replace('\\', "/");
        let rel_to_search = ann_path.strip_prefix(&search_root).unwrap_or(&ann_path);
        let (split, city) = split_city_from_rel_path(rel_to_search);
        let image_name = derive_image_name(&ann_path, rel_to_search, has_gt_fine_dir);
        parsed_files.push(parse_annotation_file(
            &value,
            &ann_path,
            Some(rel_to_dataset),
            split,
            city,
            Some(image_name),
        )?);
    }

    if parsed_files.is_empty() {
        return Err(invalid(
            path,
            "no Cityscapes *_gtFine_polygons.json files found",
        ));
    }

    dataset_from_parsed(parsed_files)
}

fn collect_cityscapes_files(root: &Path) -> Result<Vec<PathBuf>, PanlabelError> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root).follow_links(true) {
        let entry = entry.map_err(|source| {
            invalid(root, format!("failed while traversing directory: {source}"))
        })?;
        let path = entry.path();
        if entry.file_type().is_file()
            && has_json_extension(path)
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with("_gtFine_polygons.json"))
                .unwrap_or(false)
        {
            files.push(path.to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

fn parse_annotation_file(
    value: &Value,
    path: &Path,
    ann_rel_path: Option<String>,
    split: Option<String>,
    city: Option<String>,
    image_name: Option<String>,
) -> Result<ParsedFile, PanlabelError> {
    let root = value
        .as_object()
        .ok_or_else(|| invalid(path, "Cityscapes annotation must be a JSON object"))?;
    let width = required_u32(root.get("imgWidth"), path, "imgWidth", invalid_common)?;
    let height = required_u32(root.get("imgHeight"), path, "imgHeight", invalid_common)?;
    let objects = root
        .get("objects")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(path, "missing required array field 'objects'"))?;

    let mut annotations = Vec::new();
    for (idx, object) in objects.iter().enumerate() {
        if let Some(annotation) = parse_object(object, idx, path)? {
            annotations.push(annotation);
        }
    }

    Ok(ParsedFile {
        source_path: path.to_path_buf(),
        ann_rel_path,
        split,
        city,
        image_name: image_name.unwrap_or_else(|| derive_single_file_image_name(path)),
        width,
        height,
        annotations,
    })
}

fn parse_object(
    value: &Value,
    idx: usize,
    path: &Path,
) -> Result<Option<ParsedAnnotation>, PanlabelError> {
    let obj = value
        .as_object()
        .ok_or_else(|| invalid(path, format!("objects[{idx}] must be an object")))?;
    let original_label = obj
        .get("label")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .ok_or_else(|| invalid(path, format!("objects[{idx}] missing non-empty 'label'")))?;

    if object_is_deleted(obj, original_label) {
        return Ok(None);
    }

    let Some(label_decision) = decide_label(original_label) else {
        return Ok(None);
    };

    let polygon = obj
        .get("polygon")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            invalid(
                path,
                format!("objects[{idx}] missing array field 'polygon'"),
            )
        })?;
    if polygon.len() < 3 {
        return Err(invalid(
            path,
            format!("objects[{idx}] polygon must contain at least 3 points"),
        ));
    }

    let mut points = Vec::with_capacity(polygon.len());
    for (point_idx, point) in polygon.iter().enumerate() {
        points.push(parse_point_pair(
            point,
            path,
            format!("objects[{idx}].polygon[{point_idx}]"),
            invalid_common,
        )?);
    }

    let bbox = envelope(&points);
    let mut attributes = BTreeMap::new();
    attributes.insert(
        ATTR_ORIGINAL_LABEL.to_string(),
        label_decision.original_label.clone(),
    );
    attributes.insert(
        ATTR_BBOX_SOURCE.to_string(),
        BBOX_SOURCE_POLYGON_ENVELOPE.to_string(),
    );
    if label_decision.is_group {
        attributes.insert(ATTR_IS_GROUP.to_string(), "true".to_string());
    }
    if label_decision.is_unknown {
        attributes.insert(ATTR_LABEL_STATUS.to_string(), "unknown".to_string());
    }

    Ok(Some(ParsedAnnotation {
        label: label_decision.label,
        bbox,
        attributes,
    }))
}

fn object_is_deleted(obj: &Map<String, Value>, label: &str) -> bool {
    label.eq_ignore_ascii_case("deleted")
        || obj.get("deleted").and_then(Value::as_bool).unwrap_or(false)
}

fn decide_label(label: &str) -> Option<LabelDecision> {
    let normalized = label.trim();
    let lower = normalized.to_ascii_lowercase();

    if SKIPPED_LABELS.contains(&lower.as_str()) {
        return None;
    }

    if INSTANCE_LABELS.contains(&lower.as_str()) {
        return Some(LabelDecision {
            label: normalized.to_string(),
            original_label: normalized.to_string(),
            is_group: false,
            is_unknown: false,
        });
    }

    if let Some(base) = lower.strip_suffix("group") {
        if INSTANCE_LABELS.contains(&base) {
            return Some(LabelDecision {
                label: base.to_string(),
                original_label: normalized.to_string(),
                is_group: true,
                is_unknown: false,
            });
        }
    }

    Some(LabelDecision {
        label: normalized.to_string(),
        original_label: normalized.to_string(),
        is_group: lower.ends_with("group"),
        is_unknown: true,
    })
}

fn dataset_from_parsed(mut parsed_files: Vec<ParsedFile>) -> Result<Dataset, PanlabelError> {
    parsed_files.sort_by(|a, b| a.image_name.cmp(&b.image_name));

    let mut seen_images = BTreeSet::new();
    let mut label_set = BTreeSet::new();
    for parsed in &parsed_files {
        if !seen_images.insert(parsed.image_name.clone()) {
            return Err(invalid(
                &parsed.source_path,
                format!("duplicate derived image name: '{}'", parsed.image_name),
            ));
        }
        for ann in &parsed.annotations {
            label_set.insert(ann.label.clone());
        }
    }

    let categories: Vec<Category> = label_set
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
        if let Some(ann_rel_path) = &parsed.ann_rel_path {
            image
                .attributes
                .insert(ATTR_ANN_PATH.to_string(), ann_rel_path.clone());
        }
        if let Some(split) = &parsed.split {
            image
                .attributes
                .insert(ATTR_SPLIT.to_string(), split.clone());
        }
        if let Some(city) = &parsed.city {
            image.attributes.insert(ATTR_CITY.to_string(), city.clone());
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
    let value = image_to_cityscapes_value(dataset, &dataset.images[0], path)?;
    let file = fs::File::create(path).map_err(PanlabelError::Io)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &value).map_err(|source| {
        PanlabelError::CityscapesJsonWrite {
            path: path.to_path_buf(),
            source,
        }
    })
}

fn write_directory(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    validate_dataset_for_write(dataset, path)?;
    let gt_fine_dir = path.join("gtFine");
    fs::create_dir_all(&gt_fine_dir).map_err(PanlabelError::Io)?;
    fs::create_dir_all(path.join("leftImg8bit")).map_err(PanlabelError::Io)?;
    fs::write(path.join("leftImg8bit/README.txt"), IMAGES_README).map_err(PanlabelError::Io)?;

    let mut images: Vec<&Image> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    for image in images {
        let ann_rel_path = cityscapes_annotation_rel_path(image, path)?;
        let ann_path = gt_fine_dir.join(&ann_rel_path);
        if let Some(parent) = ann_path.parent() {
            fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
        }
        let value = image_to_cityscapes_value(dataset, image, &ann_path)?;
        let file = fs::File::create(&ann_path).map_err(PanlabelError::Io)?;
        serde_json::to_writer_pretty(BufWriter::new(file), &value).map_err(|source| {
            PanlabelError::CityscapesJsonWrite {
                path: ann_path.clone(),
                source,
            }
        })?;
    }
    Ok(())
}

fn image_to_cityscapes_value(
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
        let label = category_names.get(&ann.category_id).ok_or_else(|| {
            invalid(
                path,
                format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            )
        })?;
        objects.push(json!({
            "label": label,
            "polygon": rectangle_polygon(&ann.bbox),
            "deleted": false
        }));
    }

    Ok(json!({
        "imgHeight": image.height,
        "imgWidth": image.width,
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
        let rel_path = cityscapes_annotation_rel_path(image, path)?;
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

fn cityscapes_annotation_rel_path(image: &Image, path: &Path) -> Result<PathBuf, PanlabelError> {
    reject_unsafe_relative_path(&image.file_name, path, invalid_common)?;
    let inferred = split_city_from_image_name(&image.file_name);
    let split_raw = image
        .attributes
        .get(ATTR_SPLIT)
        .map(String::as_str)
        .or(inferred.0)
        .unwrap_or("train");
    let city_raw = image
        .attributes
        .get(ATTR_CITY)
        .map(String::as_str)
        .or(inferred.1)
        .unwrap_or("panlabel");
    let split = safe_cityscapes_component(split_raw, ATTR_SPLIT, path)?;
    let city = safe_cityscapes_component(city_raw, ATTR_CITY, path)?;
    let base = image_base_name(&image.file_name);
    Ok(PathBuf::from(split)
        .join(city)
        .join(format!("{base}_gtFine_polygons.json")))
}

fn safe_cityscapes_component(
    value: &str,
    field: &str,
    path: &Path,
) -> Result<String, PanlabelError> {
    let candidate = Path::new(value);
    let mut components = candidate.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(part)), None) if !value.trim().is_empty() => {
            Ok(part.to_string_lossy().to_string())
        }
        _ => Err(invalid(
            path,
            format!("unsafe Cityscapes {field} attribute '{value}'"),
        )),
    }
}

fn rectangle_polygon(bbox: &BBoxXYXY<Pixel>) -> Vec<Vec<f64>> {
    vec![
        vec![bbox.xmin(), bbox.ymin()],
        vec![bbox.xmax(), bbox.ymin()],
        vec![bbox.xmax(), bbox.ymax()],
        vec![bbox.xmin(), bbox.ymax()],
    ]
}

fn split_city_from_rel_path(rel_path: &Path) -> (Option<String>, Option<String>) {
    let components: Vec<String> = rel_path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str().map(str::to_string),
            _ => None,
        })
        .collect();
    if components.len() >= 3 {
        (
            components.get(components.len() - 3).cloned(),
            components.get(components.len() - 2).cloned(),
        )
    } else {
        (None, None)
    }
}

fn split_city_from_image_name(file_name: &str) -> (Option<&str>, Option<&str>) {
    let components: Vec<&str> = file_name
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    if components.len() >= 4 && components[0] == "leftImg8bit" {
        (Some(components[1]), Some(components[2]))
    } else if components.len() >= 3 {
        (
            Some(components[components.len() - 3]),
            Some(components[components.len() - 2]),
        )
    } else {
        (None, None)
    }
}

fn derive_image_name(ann_path: &Path, rel_path: &Path, dataset_root_input: bool) -> String {
    let image_file = format!("{}_leftImg8bit.png", annotation_base_name(ann_path));
    let components: Vec<String> = rel_path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str().map(str::to_string),
            _ => None,
        })
        .collect();
    if components.len() >= 3 {
        let split = &components[components.len() - 3];
        let city = &components[components.len() - 2];
        if dataset_root_input {
            format!("leftImg8bit/{split}/{city}/{image_file}")
        } else {
            format!("{split}/{city}/{image_file}")
        }
    } else {
        image_file
    }
}

fn derive_single_file_image_name(path: &Path) -> String {
    format!("{}_leftImg8bit.png", annotation_base_name(path))
}

fn annotation_base_name(path: &Path) -> String {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    file_name
        .strip_suffix("_gtFine_polygons.json")
        .or_else(|| file_name.strip_suffix(".json"))
        .unwrap_or(file_name)
        .to_string()
}

fn image_base_name(file_name: &str) -> String {
    let basename = Path::new(file_name)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(file_name);
    let stem = Path::new(basename)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(basename);
    stem.strip_suffix("_leftImg8bit")
        .unwrap_or(stem)
        .to_string()
}

fn invalid(path: &Path, message: impl Into<String>) -> PanlabelError {
    PanlabelError::CityscapesLayoutInvalid {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn invalid_common(path: &Path, message: String) -> PanlabelError {
    invalid(path, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_cityscapes_stuff_and_maps_groups() {
        assert!(decide_label("road").is_none());
        let car_group = decide_label("cargroup").expect("group label kept");
        assert_eq!(car_group.label, "car");
        assert!(car_group.is_group);
        let odd = decide_label("weird-instance").expect("unknown label kept");
        assert_eq!(odd.label, "weird-instance");
        assert!(odd.is_unknown);
    }
}
