//! Scale AI image annotation JSON reader and writer.
//!
//! The reader accepts Scale task objects, callback/response objects, arrays of
//! those objects, and directory exports containing JSON files under
//! `annotations/` or at the directory root. Plain `type: "box"` annotations
//! import directly from `left`, `top`, `width`, and `height`. Polygon
//! annotations and rotated boxes with `vertices` are flattened to the
//! axis-aligned envelope of those vertices with provenance attributes. Other
//! geometry types are rejected clearly because panlabel's IR is object
//! detection-only.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use super::io_adapter_common::{
    basename_from_uri_or_path, has_json_extension, is_safe_relative_image_ref, write_images_readme,
};
use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

pub const ATTR_TASK_ID: &str = "scale_ai_task_id";
pub const ATTR_ATTACHMENT: &str = "scale_ai_attachment";
pub const ATTR_UUID: &str = "scale_ai_uuid";
pub const ATTR_GEOMETRY_TYPE: &str = "scale_ai_geometry_type";
pub const ATTR_ENVELOPED: &str = "scale_ai_enveloped";
pub const ATTR_ROTATION_RAD: &str = "scale_ai_rotation_rad";
pub const ATTR_POLYGON_ENVELOPES: &str = "scale_ai_polygon_envelopes";
pub const ATTR_ROTATED_BOX_ENVELOPES: &str = "scale_ai_rotated_box_envelopes";
pub const ATTR_PREFIX_ATTRIBUTE: &str = "scale_ai_attribute_";

const STRING_HELPER_PATH: &str = "<scale-ai string>";
const IMAGES_README: &str = "Panlabel wrote Scale AI annotation JSON only. Copy your image files here if a downstream tool expects a self-contained export directory.\n";

#[derive(Debug)]
struct ParsedItem {
    source_index: usize,
    file_name: String,
    width: u32,
    height: u32,
    task_id: Option<String>,
    attachment: Option<String>,
    category_order: Vec<String>,
    objects: Vec<ParsedObject>,
}

#[derive(Debug)]
struct ParsedObject {
    source_index: usize,
    category_name: String,
    bbox: BBoxXYXY<Pixel>,
    geometry_type: GeometryType,
    uuid: Option<String>,
    rotation: Option<f64>,
    attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeometryType {
    Box,
    PolygonEnvelope,
    RotatedBoxEnvelope,
}

#[derive(Debug, Default)]
struct ReaderStats {
    polygon_envelopes: usize,
    rotated_box_envelopes: usize,
}

pub fn read_scale_ai_json(path: &Path) -> Result<Dataset, PanlabelError> {
    if path.is_file() {
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        read_scale_ai_json_file(path, base_dir)
    } else if path.is_dir() {
        read_scale_ai_json_dir(path)
    } else {
        Err(invalid(
            path,
            "path must be a Scale AI JSON file or directory",
        ))
    }
}

pub fn write_scale_ai_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    if is_json_file_path(path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
        }
        let value = dataset_to_scale_value(dataset, path)?;
        let file = File::create(path).map_err(PanlabelError::Io)?;
        serde_json::to_writer_pretty(file, &value).map_err(|source| {
            PanlabelError::ScaleAiJsonWrite {
                path: path.to_path_buf(),
                source,
            }
        })?;
        return Ok(());
    }

    let annotations_dir = path.join("annotations");
    fs::create_dir_all(&annotations_dir).map_err(PanlabelError::Io)?;
    write_images_readme(path, IMAGES_README)?;

    let category_lookup = category_lookup(dataset);
    let mut anns_by_image = annotations_by_image(dataset, path)?;
    let mut images = sorted_images(dataset);
    for image in images.drain(..) {
        let anns = anns_by_image.remove(&image.id).unwrap_or_default();
        let task = image_to_scale_task(image, anns, &category_lookup);
        let output_path =
            annotations_dir.join(format!("{}.json", safe_json_stem(&image.file_name)));
        let file = File::create(&output_path).map_err(PanlabelError::Io)?;
        serde_json::to_writer_pretty(file, &task).map_err(|source| {
            PanlabelError::ScaleAiJsonWrite {
                path: output_path,
                source,
            }
        })?;
    }
    Ok(())
}

pub fn from_scale_ai_json_str(json: &str) -> Result<Dataset, PanlabelError> {
    let value: Value =
        serde_json::from_str(json).map_err(|source| PanlabelError::ScaleAiJsonParse {
            path: PathBuf::from(STRING_HELPER_PATH),
            source,
        })?;
    values_to_dataset(vec![value], Path::new("."), Path::new(STRING_HELPER_PATH))
}

pub fn to_scale_ai_json_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let path = Path::new(STRING_HELPER_PATH);
    let value = dataset_to_scale_value(dataset, path)?;
    serde_json::to_string_pretty(&value).map_err(|source| PanlabelError::ScaleAiJsonWrite {
        path: PathBuf::from(STRING_HELPER_PATH),
        source,
    })
}

fn read_scale_ai_json_file(path: &Path, base_dir: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let value: Value =
        serde_json::from_reader(reader).map_err(|source| PanlabelError::ScaleAiJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    values_to_dataset(vec![value], base_dir, path)
}

fn read_scale_ai_json_dir(path: &Path) -> Result<Dataset, PanlabelError> {
    let json_paths = collect_scale_json_paths(path)?;
    if json_paths.is_empty() {
        return Err(invalid(
            path,
            "directory does not contain Scale AI JSON files under annotations/ or at the root",
        ));
    }

    let mut values = Vec::new();
    for json_path in json_paths {
        let file = File::open(&json_path).map_err(PanlabelError::Io)?;
        let reader = BufReader::new(file);
        let value: Value =
            serde_json::from_reader(reader).map_err(|source| PanlabelError::ScaleAiJsonParse {
                path: json_path,
                source,
            })?;
        values.push(value);
    }
    values_to_dataset(values, path, path)
}

fn collect_scale_json_paths(path: &Path) -> Result<Vec<PathBuf>, PanlabelError> {
    let mut candidates = Vec::new();
    let annotations_dir = path.join("annotations");
    if annotations_dir.is_dir() {
        for entry in walkdir::WalkDir::new(&annotations_dir).follow_links(true) {
            let entry = entry.map_err(|source| {
                invalid(
                    path,
                    format!("failed while inspecting annotations directory: {source}"),
                )
            })?;
            if entry.file_type().is_file() && is_json_file_path(entry.path()) {
                candidates.push(entry.path().to_path_buf());
            }
        }
    } else {
        for entry in fs::read_dir(path).map_err(PanlabelError::Io)? {
            let entry = entry.map_err(PanlabelError::Io)?;
            let entry_path = entry.path();
            if entry_path.is_file() && is_json_file_path(&entry_path) {
                candidates.push(entry_path);
            }
        }
    }
    candidates.sort();

    let mut matches = Vec::new();
    for candidate in candidates {
        if let Ok(contents) = fs::read_to_string(&candidate) {
            if let Ok(value) = serde_json::from_str::<Value>(&contents) {
                if is_likely_scale_ai_file(&value) {
                    matches.push(candidate);
                }
            }
        }
    }
    Ok(matches)
}

fn values_to_dataset(
    values: Vec<Value>,
    base_dir: &Path,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    let mut stats = ReaderStats::default();
    let mut parsed = Vec::new();
    for value in values {
        let items = normalize_items(value, source_path)?;
        for item in items {
            let source_index = parsed.len();
            parsed.push(parse_item(
                source_index,
                &item,
                base_dir,
                source_path,
                &mut stats,
            )?);
        }
    }

    if parsed.is_empty() {
        return Err(invalid(
            source_path,
            "Scale AI JSON contains no task/response objects",
        ));
    }

    parsed.sort_by(|a, b| {
        a.file_name
            .cmp(&b.file_name)
            .then_with(|| a.source_index.cmp(&b.source_index))
    });
    ensure_unique_file_names(&parsed, source_path)?;

    let category_names = category_names_in_order(&parsed);
    let categories: Vec<Category> = category_names
        .iter()
        .enumerate()
        .map(|(idx, name)| Category::new((idx + 1) as u64, name.clone()))
        .collect();
    let category_id_by_name: BTreeMap<String, CategoryId> = categories
        .iter()
        .map(|category| (category.name.clone(), category.id))
        .collect();

    let mut images = Vec::with_capacity(parsed.len());
    let mut annotations = Vec::new();
    let mut ann_id = 1u64;
    for (image_idx, item) in parsed.iter().enumerate() {
        let image_id = ImageId::new((image_idx + 1) as u64);
        let mut image = Image::new(image_id, item.file_name.clone(), item.width, item.height);
        if let Some(task_id) = item.task_id.as_ref().filter(|value| !value.is_empty()) {
            image
                .attributes
                .insert(ATTR_TASK_ID.to_string(), task_id.clone());
        }
        if let Some(attachment) = item.attachment.as_ref().filter(|value| !value.is_empty()) {
            image
                .attributes
                .insert(ATTR_ATTACHMENT.to_string(), attachment.clone());
        }
        images.push(image);

        let mut objects = item.objects.iter().collect::<Vec<_>>();
        objects.sort_by_key(|object| object.source_index);
        for object in objects {
            let Some(category_id) = category_id_by_name.get(&object.category_name).copied() else {
                continue;
            };
            let mut annotation = Annotation::new(
                AnnotationId::new(ann_id),
                image_id,
                category_id,
                object.bbox,
            );
            if let Some(uuid) = object.uuid.as_ref().filter(|value| !value.is_empty()) {
                annotation
                    .attributes
                    .insert(ATTR_UUID.to_string(), uuid.clone());
            }
            annotation.attributes.insert(
                ATTR_GEOMETRY_TYPE.to_string(),
                match object.geometry_type {
                    GeometryType::Box => "box",
                    GeometryType::PolygonEnvelope => "polygon",
                    GeometryType::RotatedBoxEnvelope => "rotated_box",
                }
                .to_string(),
            );
            if object.geometry_type != GeometryType::Box {
                annotation
                    .attributes
                    .insert(ATTR_ENVELOPED.to_string(), "true".to_string());
            }
            if let Some(rotation) = object.rotation {
                annotation
                    .attributes
                    .insert(ATTR_ROTATION_RAD.to_string(), rotation.to_string());
            }
            for (key, value) in &object.attributes {
                annotation
                    .attributes
                    .insert(format!("{ATTR_PREFIX_ATTRIBUTE}{key}"), value.clone());
            }
            annotations.push(annotation);
            ann_id += 1;
        }
    }

    let mut info_attributes = BTreeMap::new();
    if stats.polygon_envelopes > 0 {
        info_attributes.insert(
            ATTR_POLYGON_ENVELOPES.to_string(),
            stats.polygon_envelopes.to_string(),
        );
    }
    if stats.rotated_box_envelopes > 0 {
        info_attributes.insert(
            ATTR_ROTATED_BOX_ENVELOPES.to_string(),
            stats.rotated_box_envelopes.to_string(),
        );
    }

    Ok(Dataset {
        info: DatasetInfo {
            attributes: info_attributes,
            ..Default::default()
        },
        licenses: vec![],
        images,
        categories,
        annotations,
    })
}

fn normalize_items(value: Value, path: &Path) -> Result<Vec<Value>, PanlabelError> {
    match value {
        Value::Array(items) => {
            if items.is_empty() {
                Err(invalid(path, "Scale AI JSON array contains no objects"))
            } else {
                Ok(items)
            }
        }
        Value::Object(_) => Ok(vec![value]),
        other => Err(invalid(
            path,
            format!(
                "expected Scale AI task/response object or array, got {}",
                value_type_name(&other)
            ),
        )),
    }
}

fn parse_item(
    source_index: usize,
    value: &Value,
    base_dir: &Path,
    path: &Path,
    stats: &mut ReaderStats,
) -> Result<ParsedItem, PanlabelError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid(path, "Scale AI item must be an object"))?;

    let task_obj = object
        .get("task")
        .and_then(Value::as_object)
        .unwrap_or(object);
    let params_obj = task_obj
        .get("params")
        .and_then(Value::as_object)
        .or_else(|| object.get("params").and_then(Value::as_object));
    let metadata_obj = object
        .get("metadata")
        .and_then(Value::as_object)
        .or_else(|| task_obj.get("metadata").and_then(Value::as_object))
        .or_else(|| {
            params_obj.and_then(|params| params.get("metadata").and_then(Value::as_object))
        });
    let response_obj = object
        .get("response")
        .and_then(Value::as_object)
        .or_else(|| task_obj.get("response").and_then(Value::as_object));

    let annotations = response_obj
        .and_then(|response| response.get("annotations"))
        .and_then(Value::as_array)
        .or_else(|| object.get("annotations").and_then(Value::as_array))
        .cloned()
        .unwrap_or_default();

    if annotations.is_empty() && params_obj.is_none() && !object.contains_key("annotations") {
        return Err(invalid(
            path,
            "Scale AI object must contain response.annotations, root annotations, or params for an empty task",
        ));
    }

    let task_id = optional_string(object, "task_id")
        .or_else(|| optional_string(task_obj, "task_id"))
        .or_else(|| optional_string(task_obj, "id"))
        .or_else(|| optional_string(object, "id"));
    let attachment = params_obj
        .and_then(|params| optional_string(params, "attachment"))
        .or_else(|| optional_string(object, "attachment"))
        .or_else(|| optional_string(task_obj, "attachment"))
        .or_else(|| optional_string(object, "image"))
        .or_else(|| optional_string(object, "image_url"));
    let category_order = params_obj
        .and_then(|params| params.get("geometries"))
        .map(scale_ontology_order)
        .unwrap_or_default();

    let file_name = derive_file_name(metadata_obj, attachment.as_deref())
        .or_else(|| task_id.as_ref().map(|task_id| format!("{task_id}.jpg")))
        .unwrap_or_else(|| format!("scale-ai-response-{}.jpg", source_index + 1));
    validate_relative_image_ref(&file_name, path)?;

    let mut objects = Vec::new();
    for (ann_idx, annotation_value) in annotations.iter().enumerate() {
        let parsed = parse_annotation(ann_idx, annotation_value, path)?;
        match parsed.geometry_type {
            GeometryType::PolygonEnvelope => stats.polygon_envelopes += 1,
            GeometryType::RotatedBoxEnvelope => stats.rotated_box_envelopes += 1,
            GeometryType::Box => {}
        }
        objects.push(parsed);
    }

    let (width, height) = resolve_dimensions(
        object,
        task_obj,
        params_obj,
        metadata_obj,
        response_obj,
        attachment.as_deref(),
        &file_name,
        base_dir,
        &objects,
        path,
    )?;

    Ok(ParsedItem {
        source_index,
        file_name,
        width,
        height,
        task_id,
        attachment,
        category_order,
        objects,
    })
}

fn parse_annotation(
    source_index: usize,
    value: &Value,
    path: &Path,
) -> Result<ParsedObject, PanlabelError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid(path, "Scale AI annotation must be an object"))?;
    let geometry_type = optional_string(object, "type")
        .or_else(|| optional_string(object, "geometry"))
        .unwrap_or_else(|| {
            if object.contains_key("vertices") {
                "polygon".to_string()
            } else {
                "box".to_string()
            }
        });
    let normalized_geometry = geometry_type.to_ascii_lowercase();
    let category_name = optional_string(object, "label")
        .or_else(|| optional_string(object, "name"))
        .or_else(|| optional_string(object, "class"))
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid(path, "Scale AI annotation is missing non-empty label"))?;
    let uuid = optional_string(object, "uuid");
    let rotation = object.get("rotation").and_then(Value::as_f64);

    let (bbox, parsed_geometry_type) = match normalized_geometry.as_str() {
        "box" | "bbox" | "bounding_box" | "boundingbox" => {
            if rotation.is_some() || object.contains_key("vertices") {
                let vertices = object.get("vertices").ok_or_else(|| {
                    invalid(
                        path,
                        "Scale AI rotated box has rotation but no vertices; cannot safely flatten",
                    )
                })?;
                (
                    vertices_to_envelope(vertices, path, "annotation.vertices")?,
                    GeometryType::RotatedBoxEnvelope,
                )
            } else {
                (
                    BBoxXYXY::<Pixel>::from_xywh(
                        required_f64(object.get("left"), path, "annotation.left")?,
                        required_f64(object.get("top"), path, "annotation.top")?,
                        required_f64(object.get("width"), path, "annotation.width")?,
                        required_f64(object.get("height"), path, "annotation.height")?,
                    ),
                    GeometryType::Box,
                )
            }
        }
        "polygon" => (
            vertices_to_envelope(
                object.get("vertices").ok_or_else(|| {
                    invalid(path, "Scale AI polygon annotation is missing vertices")
                })?,
                path,
                "annotation.vertices",
            )?,
            GeometryType::PolygonEnvelope,
        ),
        other => {
            return Err(invalid(
                path,
                format!(
                    "unsupported Scale AI geometry type '{other}'; panlabel supports only box, rotated box with vertices, and polygon envelopes"
                ),
            ));
        }
    };

    let attributes = object
        .get("attributes")
        .and_then(Value::as_object)
        .map(|attrs| {
            attrs
                .iter()
                .map(|(key, value)| (key.clone(), value_to_attribute_string(value)))
                .collect()
        })
        .unwrap_or_default();

    Ok(ParsedObject {
        source_index,
        category_name,
        bbox,
        geometry_type: parsed_geometry_type,
        uuid,
        rotation,
        attributes,
    })
}

fn vertices_to_envelope(
    value: &Value,
    path: &Path,
    field_name: &str,
) -> Result<BBoxXYXY<Pixel>, PanlabelError> {
    let vertices = value
        .as_array()
        .ok_or_else(|| invalid(path, format!("{field_name} must be an array")))?;
    if vertices.is_empty() {
        return Err(invalid(path, format!("{field_name} must not be empty")));
    }

    let mut xmin = f64::INFINITY;
    let mut ymin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for (idx, vertex) in vertices.iter().enumerate() {
        let vertex_obj = vertex.as_object().ok_or_else(|| {
            invalid(
                path,
                format!("{field_name}[{idx}] must be an object with x/y"),
            )
        })?;
        let x = required_f64(vertex_obj.get("x"), path, &format!("{field_name}[{idx}].x"))?;
        let y = required_f64(vertex_obj.get("y"), path, &format!("{field_name}[{idx}].y"))?;
        xmin = xmin.min(x);
        ymin = ymin.min(y);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
    }
    Ok(BBoxXYXY::<Pixel>::from_xyxy(xmin, ymin, xmax, ymax))
}

#[allow(clippy::too_many_arguments)]
fn resolve_dimensions(
    object: &Map<String, Value>,
    task_obj: &Map<String, Value>,
    params_obj: Option<&Map<String, Value>>,
    metadata_obj: Option<&Map<String, Value>>,
    response_obj: Option<&Map<String, Value>>,
    attachment: Option<&str>,
    file_name: &str,
    base_dir: &Path,
    objects: &[ParsedObject],
    path: &Path,
) -> Result<(u32, u32), PanlabelError> {
    for candidate in [
        Some(object),
        Some(task_obj),
        params_obj,
        metadata_obj,
        response_obj,
    ]
    .into_iter()
    .flatten()
    {
        if let Some(dimensions) = dimensions_from_object(candidate, path)? {
            return Ok(dimensions);
        }
    }

    for image_ref in [attachment, Some(file_name)].into_iter().flatten() {
        if let Some(dimensions) = probe_image_dimensions(base_dir, image_ref) {
            return Ok(dimensions);
        }
    }

    if let Some(dimensions) = dimensions_from_geometry(objects) {
        return Ok(dimensions);
    }

    Err(invalid(
        path,
        "could not determine image width/height from Scale AI JSON metadata, local image file, or annotation geometry",
    ))
}

fn dimensions_from_object(
    object: &Map<String, Value>,
    path: &Path,
) -> Result<Option<(u32, u32)>, PanlabelError> {
    let width_value = object
        .get("width")
        .or_else(|| object.get("image_width"))
        .or_else(|| object.get("attachment_width"));
    let height_value = object
        .get("height")
        .or_else(|| object.get("image_height"))
        .or_else(|| object.get("attachment_height"));
    match (width_value, height_value) {
        (Some(width), Some(height)) => Ok(Some((
            required_u32(Some(width), path, "width")?,
            required_u32(Some(height), path, "height")?,
        ))),
        _ => Ok(None),
    }
}

fn probe_image_dimensions(base_dir: &Path, image_ref: &str) -> Option<(u32, u32)> {
    let file_name = basename_from_uri_or_path(image_ref)?;
    for candidate in [
        base_dir.join(&file_name),
        base_dir.join("images").join(&file_name),
    ] {
        if let Ok(size) = imagesize::size(&candidate) {
            return Some((size.width as u32, size.height as u32));
        }
    }
    None
}

fn dimensions_from_geometry(objects: &[ParsedObject]) -> Option<(u32, u32)> {
    let xmax = objects
        .iter()
        .map(|object| object.bbox.xmax())
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max);
    let ymax = objects
        .iter()
        .map(|object| object.bbox.ymax())
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max);
    if xmax > 0.0 && ymax > 0.0 {
        Some((xmax.ceil().max(1.0) as u32, ymax.ceil().max(1.0) as u32))
    } else {
        None
    }
}

fn derive_file_name(
    metadata_obj: Option<&Map<String, Value>>,
    attachment: Option<&str>,
) -> Option<String> {
    metadata_obj
        .and_then(|metadata| {
            optional_string(metadata, "file_name")
                .or_else(|| optional_string(metadata, "filename"))
                .or_else(|| optional_string(metadata, "image"))
        })
        .or_else(|| attachment.and_then(basename_from_uri_or_path))
}

fn category_names_in_order(items: &[ParsedItem]) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = BTreeSet::new();
    for item in items {
        for name in &item.category_order {
            if !name.trim().is_empty() && seen.insert(name.clone()) {
                names.push(name.clone());
            }
        }
    }

    let mut extra = BTreeSet::new();
    for item in items {
        for object in &item.objects {
            if !seen.contains(&object.category_name) {
                extra.insert(object.category_name.clone());
            }
        }
    }
    names.extend(extra);
    names
}

fn scale_ontology_order(value: &Value) -> Vec<String> {
    let Some(geometries) = value.as_object() else {
        return Vec::new();
    };

    let mut names = Vec::new();
    let mut seen = BTreeSet::new();
    for geometry in geometries.values() {
        let Some(objects) = geometry
            .as_object()
            .and_then(|geometry| geometry.get("objects_to_annotate"))
            .and_then(Value::as_array)
        else {
            continue;
        };
        for object in objects {
            collect_ontology_label(object, &mut names, &mut seen);
        }
    }
    names
}

fn collect_ontology_label(value: &Value, names: &mut Vec<String>, seen: &mut BTreeSet<String>) {
    if let Some(name) = value.as_str().filter(|name| !name.trim().is_empty()) {
        if seen.insert(name.to_string()) {
            names.push(name.to_string());
        }
        return;
    }

    let Some(object) = value.as_object() else {
        return;
    };
    for key in ["choice", "label", "value", "display"] {
        if let Some(name) = optional_string(object, key).filter(|name| !name.trim().is_empty()) {
            if seen.insert(name.clone()) {
                names.push(name);
            }
            break;
        }
    }
    if let Some(subchoices) = object.get("subchoices").and_then(Value::as_array) {
        for subchoice in subchoices {
            collect_ontology_label(subchoice, names, seen);
        }
    }
}

fn ensure_unique_file_names(items: &[ParsedItem], path: &Path) -> Result<(), PanlabelError> {
    let mut seen = BTreeSet::new();
    for item in items {
        if !seen.insert(item.file_name.clone()) {
            return Err(invalid(
                path,
                format!("duplicate derived image file_name '{}'", item.file_name),
            ));
        }
    }
    Ok(())
}

fn dataset_to_scale_value(dataset: &Dataset, path: &Path) -> Result<Value, PanlabelError> {
    let category_lookup = category_lookup(dataset);
    let mut anns_by_image = annotations_by_image(dataset, path)?;
    let images = sorted_images(dataset);
    let mut tasks = Vec::with_capacity(images.len());
    for image in images {
        let anns = anns_by_image.remove(&image.id).unwrap_or_default();
        tasks.push(image_to_scale_task(image, anns, &category_lookup));
    }
    Ok(if tasks.len() == 1 {
        tasks.remove(0)
    } else {
        Value::Array(tasks)
    })
}

fn image_to_scale_task(
    image: &Image,
    annotations: Vec<&Annotation>,
    category_lookup: &BTreeMap<CategoryId, &Category>,
) -> Value {
    let mut sorted_annotations = annotations;
    sorted_annotations.sort_by_key(|ann| ann.id);

    let response_annotations = sorted_annotations
        .into_iter()
        .filter_map(|ann| {
            category_lookup
                .get(&ann.category_id)
                .map(|category| annotation_to_scale_value(ann, category))
        })
        .collect::<Vec<_>>();

    let mut category_names = category_lookup
        .values()
        .map(|category| category.name.clone())
        .collect::<Vec<_>>();
    category_names.sort();
    category_names.dedup();

    let task_id = image
        .attributes
        .get(ATTR_TASK_ID)
        .cloned()
        .unwrap_or_else(|| format!("panlabel-image-{}", image.id.as_u64()));
    let attachment = image
        .attributes
        .get(ATTR_ATTACHMENT)
        .cloned()
        .unwrap_or_else(|| image.file_name.clone());

    json!({
        "task_id": task_id,
        "type": "imageannotation",
        "status": "completed",
        "params": {
            "attachment_type": "image",
            "attachment": attachment,
            "geometries": {
                "box": {
                    "objects_to_annotate": category_names,
                    "integer_pixels": false,
                    "can_rotate": false
                }
            },
            "metadata": {
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height
            }
        },
        "response": {
            "annotations": response_annotations
        }
    })
}

fn annotation_to_scale_value(annotation: &Annotation, category: &Category) -> Value {
    let (left, top, width, height) = annotation.bbox.to_xywh();
    let mut object = Map::new();
    object.insert("type".to_string(), Value::String("box".to_string()));
    object.insert("label".to_string(), Value::String(category.name.clone()));
    object.insert("left".to_string(), Value::from(left));
    object.insert("top".to_string(), Value::from(top));
    object.insert("width".to_string(), Value::from(width));
    object.insert("height".to_string(), Value::from(height));
    object.insert(
        "uuid".to_string(),
        Value::String(
            annotation
                .attributes
                .get(ATTR_UUID)
                .cloned()
                .unwrap_or_else(|| format!("panlabel-ann-{}", annotation.id.as_u64())),
        ),
    );

    let mut attributes = Map::new();
    for (key, value) in &annotation.attributes {
        if let Some(stripped) = key.strip_prefix(ATTR_PREFIX_ATTRIBUTE) {
            attributes.insert(stripped.to_string(), Value::String(value.clone()));
        }
    }
    object.insert("attributes".to_string(), Value::Object(attributes));
    Value::Object(object)
}

fn category_lookup(dataset: &Dataset) -> BTreeMap<CategoryId, &Category> {
    dataset
        .categories
        .iter()
        .map(|category| (category.id, category))
        .collect()
}

fn annotations_by_image<'a>(
    dataset: &'a Dataset,
    path: &Path,
) -> Result<BTreeMap<ImageId, Vec<&'a Annotation>>, PanlabelError> {
    let image_lookup: BTreeSet<ImageId> = dataset.images.iter().map(|image| image.id).collect();
    let category_lookup: BTreeSet<CategoryId> = dataset
        .categories
        .iter()
        .map(|category| category.id)
        .collect();
    let mut anns_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        if !image_lookup.contains(&ann.image_id) {
            return Err(invalid(
                path,
                format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            ));
        }
        if !category_lookup.contains(&ann.category_id) {
            return Err(invalid(
                path,
                format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            ));
        }
        if !ann.bbox.is_finite() {
            return Err(invalid(
                path,
                format!("annotation {} has non-finite bbox", ann.id.as_u64()),
            ));
        }
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }
    Ok(anns_by_image)
}

fn sorted_images(dataset: &Dataset) -> Vec<&Image> {
    let mut images = dataset.images.iter().collect::<Vec<_>>();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name).then_with(|| a.id.cmp(&b.id)));
    images
}

fn required_f64(
    value: Option<&Value>,
    path: &Path,
    field_name: &str,
) -> Result<f64, PanlabelError> {
    let value =
        value.ok_or_else(|| invalid(path, format!("missing required field '{field_name}'")))?;
    let number = value
        .as_f64()
        .ok_or_else(|| invalid(path, format!("field '{field_name}' must be a number")))?;
    if !number.is_finite() {
        return Err(invalid(
            path,
            format!("field '{field_name}' must be finite"),
        ));
    }
    Ok(number)
}

fn required_u32(
    value: Option<&Value>,
    path: &Path,
    field_name: &str,
) -> Result<u32, PanlabelError> {
    let value =
        value.ok_or_else(|| invalid(path, format!("missing required field '{field_name}'")))?;
    if let Some(unsigned) = value.as_u64() {
        if unsigned == 0 {
            return Err(invalid(
                path,
                format!("field '{field_name}' must be positive"),
            ));
        }
        return u32::try_from(unsigned)
            .map_err(|_| invalid(path, format!("field '{field_name}' is out of range")));
    }
    let number = value.as_f64().ok_or_else(|| {
        invalid(
            path,
            format!("field '{field_name}' must be a positive integer"),
        )
    })?;
    if !number.is_finite() || number <= 0.0 || number.fract() != 0.0 || number > u32::MAX as f64 {
        return Err(invalid(
            path,
            format!("field '{field_name}' must be a positive integer"),
        ));
    }
    Ok(number as u32)
}

fn optional_string(object: &Map<String, Value>, key: &str) -> Option<String> {
    object.get(key).and_then(Value::as_str).map(str::to_string)
}

fn value_to_attribute_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Null => "null".to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn validate_relative_image_ref(image_ref: &str, path: &Path) -> Result<(), PanlabelError> {
    if !is_safe_relative_image_ref(image_ref) {
        return Err(invalid(
            path,
            format!(
                "image reference '{}' must be a relative path without parent-directory components",
                image_ref
            ),
        ));
    }
    Ok(())
}

fn safe_json_stem(file_name: &str) -> String {
    Path::new(file_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.trim().is_empty())
        .unwrap_or("image")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn is_json_file_path(path: &Path) -> bool {
    has_json_extension(path)
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn invalid(path: &Path, message: impl Into<String>) -> PanlabelError {
    PanlabelError::ScaleAiJsonInvalid {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

/// Lightweight structural probe used by CLI auto-detection.
pub(crate) fn is_likely_scale_ai_file(value: &Value) -> bool {
    if let Some(items) = value.as_array() {
        return items.iter().any(is_likely_scale_ai_file);
    }
    let Some(object) = value.as_object() else {
        return false;
    };

    let has_response_annotations = object
        .get("response")
        .and_then(Value::as_object)
        .and_then(|response| response.get("annotations"))
        .map(is_likely_scale_annotations_array)
        .unwrap_or(false);
    let has_root_annotations = object
        .get("annotations")
        .map(is_likely_scale_annotations_array)
        .unwrap_or(false);
    let has_scale_task_shape = object
        .get("params")
        .and_then(Value::as_object)
        .map(|params| {
            params
                .get("attachment")
                .map(Value::is_string)
                .unwrap_or(false)
                && params
                    .get("geometries")
                    .map(Value::is_object)
                    .unwrap_or(false)
        })
        .unwrap_or(false);
    let nested_task_has_params = object
        .get("task")
        .and_then(Value::as_object)
        .and_then(|task| task.get("params"))
        .and_then(Value::as_object)
        .map(|params| {
            params
                .get("attachment")
                .map(Value::is_string)
                .unwrap_or(false)
        })
        .unwrap_or(false);

    has_response_annotations
        || has_root_annotations
        || has_scale_task_shape
        || nested_task_has_params
}

fn is_likely_scale_annotations_array(value: &Value) -> bool {
    let Some(annotations) = value.as_array() else {
        return false;
    };
    annotations.iter().any(is_likely_scale_annotation)
}

fn is_likely_scale_annotation(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let has_label = object.get("label").map(Value::is_string).unwrap_or(false)
        || object.get("name").map(Value::is_string).unwrap_or(false)
        || object.get("class").map(Value::is_string).unwrap_or(false);
    let has_scale_box = ["left", "top", "width", "height"]
        .iter()
        .all(|key| object.get(*key).map(Value::is_number).unwrap_or(false));
    let has_vertices = object.get("vertices").map(Value::is_array).unwrap_or(false);
    let geometry_type = object
        .get("type")
        .or_else(|| object.get("geometry"))
        .and_then(Value::as_str)
        .map(|value| value.to_ascii_lowercase());
    let known_scale_geometry = geometry_type
        .as_deref()
        .map(|kind| {
            matches!(
                kind,
                "box" | "bbox" | "bounding_box" | "boundingbox" | "polygon"
            )
        })
        .unwrap_or(false);

    has_label && (has_scale_box || (known_scale_geometry && has_vertices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_box_polygon_and_rotated_box() {
        let json = r#"[
            {
                "task_id": "task-1",
                "params": {"attachment": "https://example.com/img1.jpg", "metadata": {"width": 100, "height": 80}},
                "response": {"annotations": [
                    {"type": "box", "label": "car", "left": 10, "top": 20, "width": 30, "height": 40, "uuid": "box-1"},
                    {"type": "polygon", "label": "person", "vertices": [{"x": 1, "y": 2}, {"x": 11, "y": 4}, {"x": 5, "y": 12}], "uuid": "poly-1"},
                    {"type": "box", "label": "truck", "left": 0, "top": 0, "width": 5, "height": 5, "rotation": 0.5, "vertices": [{"x": 50, "y": 10}, {"x": 70, "y": 20}, {"x": 60, "y": 40}, {"x": 40, "y": 30}], "uuid": "rot-1"}
                ]}
            }
        ]"#;
        let dataset = from_scale_ai_json_str(json).expect("parse Scale AI JSON");
        assert_eq!(dataset.images.len(), 1);
        assert_eq!(dataset.categories.len(), 3);
        assert_eq!(dataset.annotations.len(), 3);
        assert_eq!(dataset.images[0].file_name, "img1.jpg");
        assert_eq!(dataset.info.attributes[ATTR_POLYGON_ENVELOPES], "1");
        assert_eq!(dataset.info.attributes[ATTR_ROTATED_BOX_ENVELOPES], "1");
        let rotated = dataset
            .annotations
            .iter()
            .find(|ann| ann.attributes.get(ATTR_UUID).map(String::as_str) == Some("rot-1"))
            .expect("rotated box");
        assert_eq!(rotated.bbox.xmin(), 40.0);
        assert_eq!(rotated.bbox.ymax(), 40.0);
    }

    #[test]
    fn rejects_unsupported_geometry() {
        let err = from_scale_ai_json_str(
            r#"{"annotations":[{"type":"line","label":"lane","vertices":[{"x":1,"y":2},{"x":3,"y":4}]}]}"#,
        )
        .expect_err("unsupported geometry should fail");
        assert!(err
            .to_string()
            .contains("unsupported Scale AI geometry type 'line'"));
    }
}
