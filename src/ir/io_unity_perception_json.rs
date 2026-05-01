//! Unity Perception / SOLO JSON reader and writer.
//!
//! The reader accepts SOLO-style per-frame JSON files with a `captures` array,
//! directories containing those frame files, and the older narrow
//! `captures_*.json` shape where captures are collected in one file. 2D
//! bounding-box annotations import from `values` entries using either
//! `x`/`y`/`width`/`height` or `origin`/`dimension`. Non-bbox annotations are
//! skipped with a dataset-level warning count while their captures/images are
//! still preserved in the IR.
//!
//! The writer emits a minimal bbox-only SOLO-like directory. It intentionally
//! rejects `.json` file outputs because one JSON file is ambiguous for multiple
//! frames/images and easy to confuse with legacy captures shards.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use super::io_adapter_common::{has_json_extension, write_images_readme};
use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

pub const ATTR_CAPTURE_ID: &str = "unity_perception_capture_id";
pub const ATTR_SEQUENCE_ID: &str = "unity_perception_sequence_id";
pub const ATTR_SEQUENCE: &str = "unity_perception_sequence";
pub const ATTR_STEP: &str = "unity_perception_step";
pub const ATTR_FRAME: &str = "unity_perception_frame";
pub const ATTR_TIMESTAMP: &str = "unity_perception_timestamp";
pub const ATTR_SENSOR_ID: &str = "unity_perception_sensor_id";
pub const ATTR_ANNOTATION_ID: &str = "unity_perception_annotation_id";
pub const ATTR_ANNOTATION_DEFINITION: &str = "unity_perception_annotation_definition";
pub const ATTR_LABEL_ID: &str = "unity_perception_label_id";
pub const ATTR_INSTANCE_ID: &str = "unity_perception_instance_id";
pub const ATTR_SKIPPED_ANNOTATIONS: &str = "unity_perception_skipped_annotations";

const STRING_HELPER_PATH: &str = "<unity-perception string>";
const BBOX_ANNOTATION_TYPE: &str = "type.unity.com/unity.solo.BoundingBox2DAnnotation";
const BBOX_DEFINITION_TYPE: &str = "type.unity.com/unity.solo.BoundingBox2DAnnotationDefinition";
const DEFAULT_ANNOTATION_DEFINITION_ID: &str = "bounding_box_2d";
const DEFAULT_SENSOR_ID: &str = "camera";
const IMAGES_README: &str = "Panlabel wrote Unity Perception/SOLO annotation JSON only. Copy your image files into this dataset if a downstream tool expects a self-contained export directory.\n";

#[derive(Debug, Default)]
struct ReaderStats {
    skipped_annotations: usize,
}

#[derive(Debug, Default)]
struct Definitions {
    labels_by_id: BTreeMap<String, String>,
    label_order: Vec<String>,
}

#[derive(Debug)]
struct ParsedCapture {
    source_index: usize,
    file_name: String,
    width: u32,
    height: u32,
    capture_id: Option<String>,
    sequence_id: Option<String>,
    sequence: Option<String>,
    step: Option<String>,
    frame: Option<String>,
    timestamp: Option<String>,
    sensor_id: Option<String>,
    objects: Vec<ParsedObject>,
}

#[derive(Debug)]
struct ParsedObject {
    source_index: usize,
    annotation_id: Option<String>,
    annotation_definition: Option<String>,
    category_name: String,
    label_id: Option<String>,
    instance_id: Option<String>,
    bbox: BBoxXYXY<Pixel>,
}

pub fn read_unity_perception_json(path: &Path) -> Result<Dataset, PanlabelError> {
    if path.is_file() {
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let definitions = read_definitions_near_file(path)?;
        read_unity_file(path, base_dir, &definitions)
    } else if path.is_dir() {
        read_unity_dir(path)
    } else {
        Err(invalid(
            path,
            "path must be a Unity Perception JSON file or directory",
        ))
    }
}

pub fn write_unity_perception_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    if is_json_file_path(path) || path.is_file() {
        return Err(PanlabelError::UnityPerceptionWriteError {
            path: path.to_path_buf(),
            message: "Unity Perception writer emits directory datasets only; choose an output directory instead of a .json file".to_string(),
        });
    }

    fs::create_dir_all(path).map_err(PanlabelError::Io)?;
    write_images_readme(path, IMAGES_README)?;
    write_annotation_definitions(path, dataset)?;

    let sequence_dir = path.join("sequence.0");
    clear_panlabel_frame_files(&sequence_dir)?;
    fs::create_dir_all(&sequence_dir).map_err(PanlabelError::Io)?;

    let category_label_ids = writer_category_label_ids(dataset);
    let category_lookup = category_lookup(dataset);
    let mut anns_by_image = annotations_by_image(dataset);

    for (idx, image) in sorted_images(dataset).into_iter().enumerate() {
        let anns = anns_by_image.remove(&image.id).unwrap_or_default();
        let frame = image_to_frame_json(image, anns, &category_lookup, &category_label_ids, idx);
        let output_path = sequence_dir.join(format!("step{idx}.frame_data.json"));
        let file = File::create(&output_path).map_err(PanlabelError::Io)?;
        serde_json::to_writer_pretty(file, &frame).map_err(|source| {
            PanlabelError::UnityPerceptionJsonWrite {
                path: output_path,
                source,
            }
        })?;
    }

    Ok(())
}

pub fn from_unity_perception_json_str(json: &str) -> Result<Dataset, PanlabelError> {
    let value: Value =
        serde_json::from_str(json).map_err(|source| PanlabelError::UnityPerceptionJsonParse {
            path: PathBuf::from(STRING_HELPER_PATH),
            source,
        })?;
    let definitions = Definitions::default();
    values_to_dataset(
        vec![(PathBuf::from(STRING_HELPER_PATH), value)],
        Path::new("."),
        &definitions,
    )
}

fn read_unity_file(
    path: &Path,
    base_dir: &Path,
    definitions: &Definitions,
) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let value: Value = serde_json::from_reader(reader).map_err(|source| {
        PanlabelError::UnityPerceptionJsonParse {
            path: path.to_path_buf(),
            source,
        }
    })?;
    values_to_dataset(vec![(path.to_path_buf(), value)], base_dir, definitions)
}

fn read_unity_dir(path: &Path) -> Result<Dataset, PanlabelError> {
    let definitions = read_definitions_in_dir(path)?;
    let json_paths = collect_unity_json_paths(path)?;
    if json_paths.is_empty() {
        return Err(invalid(
            path,
            "directory does not contain Unity Perception/SOLO frame or captures JSON files",
        ));
    }

    let mut values = Vec::new();
    for json_path in json_paths {
        let file = File::open(&json_path).map_err(PanlabelError::Io)?;
        let reader = BufReader::new(file);
        let value: Value = serde_json::from_reader(reader).map_err(|source| {
            PanlabelError::UnityPerceptionJsonParse {
                path: json_path.clone(),
                source,
            }
        })?;
        values.push((json_path, value));
    }
    values_to_dataset(values, path, &definitions)
}

fn values_to_dataset(
    values: Vec<(PathBuf, Value)>,
    base_dir: &Path,
    definitions: &Definitions,
) -> Result<Dataset, PanlabelError> {
    let mut stats = ReaderStats::default();
    let mut parsed = Vec::new();
    for (source_path, value) in values {
        parse_value(
            &source_path,
            &value,
            base_dir,
            definitions,
            &mut parsed,
            &mut stats,
        )?;
    }

    if parsed.is_empty() {
        return Err(invalid(
            base_dir,
            "Unity Perception JSON contains no captures",
        ));
    }

    parsed.sort_by(|a, b| {
        a.file_name
            .cmp(&b.file_name)
            .then_with(|| a.source_index.cmp(&b.source_index))
    });
    ensure_unique_file_names(&parsed, base_dir)?;

    let category_names = category_names_in_order(&parsed, definitions);
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
    let mut next_ann_id = 1u64;
    for (image_idx, capture) in parsed.into_iter().enumerate() {
        let image_id = ImageId::new((image_idx + 1) as u64);
        let mut image = Image::new(
            image_id,
            capture.file_name.clone(),
            capture.width,
            capture.height,
        );
        insert_opt(&mut image.attributes, ATTR_CAPTURE_ID, capture.capture_id);
        insert_opt(&mut image.attributes, ATTR_SEQUENCE_ID, capture.sequence_id);
        insert_opt(&mut image.attributes, ATTR_SEQUENCE, capture.sequence);
        insert_opt(&mut image.attributes, ATTR_STEP, capture.step);
        insert_opt(&mut image.attributes, ATTR_FRAME, capture.frame);
        insert_opt(&mut image.attributes, ATTR_TIMESTAMP, capture.timestamp);
        insert_opt(&mut image.attributes, ATTR_SENSOR_ID, capture.sensor_id);

        let mut objects = capture.objects;
        objects.sort_by_key(|object| object.source_index);
        for object in objects {
            let Some(category_id) = category_id_by_name.get(&object.category_name).copied() else {
                continue;
            };
            let mut ann = Annotation::new(
                AnnotationId::new(next_ann_id),
                image_id,
                category_id,
                object.bbox,
            );
            insert_opt(
                &mut ann.attributes,
                ATTR_ANNOTATION_ID,
                object.annotation_id,
            );
            insert_opt(
                &mut ann.attributes,
                ATTR_ANNOTATION_DEFINITION,
                object.annotation_definition,
            );
            insert_opt(&mut ann.attributes, ATTR_LABEL_ID, object.label_id);
            insert_opt(&mut ann.attributes, ATTR_INSTANCE_ID, object.instance_id);
            annotations.push(ann);
            next_ann_id += 1;
        }
        images.push(image);
    }

    let mut info = DatasetInfo {
        name: Some("Unity Perception dataset".to_string()),
        ..Default::default()
    };
    if stats.skipped_annotations > 0 {
        info.attributes.insert(
            ATTR_SKIPPED_ANNOTATIONS.to_string(),
            stats.skipped_annotations.to_string(),
        );
    }

    Ok(Dataset {
        info,
        licenses: Vec::new(),
        images,
        categories,
        annotations,
    })
}

fn parse_value(
    source_path: &Path,
    value: &Value,
    base_dir: &Path,
    definitions: &Definitions,
    parsed: &mut Vec<ParsedCapture>,
    stats: &mut ReaderStats,
) -> Result<(), PanlabelError> {
    if let Some(items) = value.as_array() {
        for item in items {
            parse_value(source_path, item, base_dir, definitions, parsed, stats)?;
        }
        return Ok(());
    }

    let Some(obj) = value.as_object() else {
        return Err(invalid(
            source_path,
            format!(
                "expected Unity Perception JSON object, got {}",
                value_type_name(value)
            ),
        ));
    };

    if obj.contains_key("filename") && obj.contains_key("annotations") {
        parse_capture_object(
            source_path,
            obj,
            FrameContext::from_object(obj),
            base_dir,
            definitions,
            parsed,
            stats,
        )?;
        return Ok(());
    }

    let captures = obj
        .get("captures")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(source_path, "missing or invalid 'captures' array"))?;

    let frame_context = FrameContext::from_object(obj);
    for capture in captures {
        let Some(capture_obj) = capture.as_object() else {
            return Err(invalid(source_path, "capture entry must be a JSON object"));
        };
        parse_capture_object(
            source_path,
            capture_obj,
            frame_context.clone(),
            base_dir,
            definitions,
            parsed,
            stats,
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
struct FrameContext {
    sequence_id: Option<String>,
    sequence: Option<String>,
    step: Option<String>,
    frame: Option<String>,
    timestamp: Option<String>,
}

impl FrameContext {
    fn from_object(obj: &Map<String, Value>) -> Self {
        Self {
            sequence_id: stringish_field(obj, &["sequence_id", "sequenceId"]),
            sequence: stringish_field(obj, &["sequence"]),
            step: stringish_field(obj, &["step"]),
            frame: stringish_field(obj, &["frame"]),
            timestamp: stringish_field(obj, &["timestamp"]),
        }
    }

    fn merge_capture(mut self, capture: &Map<String, Value>) -> Self {
        if self.sequence_id.is_none() {
            self.sequence_id = stringish_field(capture, &["sequence_id", "sequenceId"]);
        }
        if self.sequence.is_none() {
            self.sequence = stringish_field(capture, &["sequence"]);
        }
        if self.step.is_none() {
            self.step = stringish_field(capture, &["step"]);
        }
        if self.frame.is_none() {
            self.frame = stringish_field(capture, &["frame"]);
        }
        if self.timestamp.is_none() {
            self.timestamp = stringish_field(capture, &["timestamp"]);
        }
        self
    }
}

fn parse_capture_object(
    source_path: &Path,
    capture: &Map<String, Value>,
    frame_context: FrameContext,
    base_dir: &Path,
    definitions: &Definitions,
    parsed: &mut Vec<ParsedCapture>,
    stats: &mut ReaderStats,
) -> Result<(), PanlabelError> {
    let file_name = string_field(capture, &["filename", "file_name"])
        .ok_or_else(|| invalid(source_path, "capture is missing filename"))?;
    let source_index = parsed.len();
    let mut objects = Vec::new();
    let annotations = capture
        .get("annotations")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);

    for (ann_idx, annotation) in annotations.iter().enumerate() {
        let Some(annotation_obj) = annotation.as_object() else {
            return Err(invalid(
                source_path,
                "annotation entry must be a JSON object",
            ));
        };
        if !is_bbox_annotation(annotation_obj) {
            warn_skip_non_bbox(source_path, &file_name, ann_idx, annotation_obj);
            stats.skipped_annotations += 1;
            continue;
        }
        let values = annotation_obj
            .get("values")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                invalid(
                    source_path,
                    "BoundingBox2D annotation is missing values array",
                )
            })?;
        for (value_idx, item) in values.iter().enumerate() {
            let Some(value_obj) = item.as_object() else {
                return Err(invalid(
                    source_path,
                    "BoundingBox2D value must be a JSON object",
                ));
            };
            let bbox = bbox_from_value(value_obj).ok_or_else(|| {
                invalid(
                    source_path,
                    "BoundingBox2D value must contain x/y/width/height or origin/dimension",
                )
            })?;
            let label_id = stringish_field(value_obj, &["label_id", "labelId"]);
            let category_name = label_name_for_value(value_obj, label_id.as_deref(), definitions)
                .ok_or_else(|| {
                invalid(
                    source_path,
                    "BoundingBox2D value is missing label_name/label_id",
                )
            })?;
            let annotation_id = stringish_field(annotation_obj, &["id"]);
            let annotation_definition = stringish_field(
                annotation_obj,
                &["annotation_definition", "annotationDefinition"],
            );
            let instance_id = stringish_field(value_obj, &["instance_id", "instanceId"]);
            objects.push(ParsedObject {
                source_index: ann_idx * 1_000_000 + value_idx,
                annotation_id,
                annotation_definition,
                category_name,
                label_id,
                instance_id,
                bbox,
            });
        }
    }

    let (width, height) = resolve_dimensions(capture, base_dir, &file_name, &objects);
    let context = frame_context.merge_capture(capture);
    parsed.push(ParsedCapture {
        source_index,
        file_name,
        width,
        height,
        capture_id: stringish_field(capture, &["id", "capture_id", "captureId"]),
        sequence_id: context.sequence_id,
        sequence: context.sequence,
        step: context.step,
        frame: context.frame,
        timestamp: context.timestamp,
        sensor_id: stringish_field(capture, &["sensor_id", "sensorId", "id"]),
        objects,
    });
    Ok(())
}

fn is_bbox_annotation(annotation: &Map<String, Value>) -> bool {
    for key in ["@type", "type", "name"] {
        if annotation
            .get(key)
            .and_then(Value::as_str)
            .map(|value| {
                let lower = value.to_ascii_lowercase();
                lower.contains("boundingbox2d") || lower.contains("bounding_box_2d")
            })
            .unwrap_or(false)
        {
            return true;
        }
    }

    annotation
        .get("values")
        .and_then(Value::as_array)
        .and_then(|values| values.first())
        .and_then(Value::as_object)
        .map(|value| bbox_from_value(value).is_some())
        .unwrap_or(false)
}

fn bbox_from_value(value: &Map<String, Value>) -> Option<BBoxXYXY<Pixel>> {
    let (x, y, width, height) = if let (Some(x), Some(y), Some(width), Some(height)) = (
        number_field(value, &["x"]),
        number_field(value, &["y"]),
        number_field(value, &["width"]),
        number_field(value, &["height"]),
    ) {
        (x, y, width, height)
    } else {
        let origin = value.get("origin").and_then(Value::as_array)?;
        let dimension = value.get("dimension").and_then(Value::as_array)?;
        if origin.len() < 2 || dimension.len() < 2 {
            return None;
        }
        let x = origin.first()?.as_f64()?;
        let y = origin.get(1)?.as_f64()?;
        let width = dimension.first()?.as_f64()?;
        let height = dimension.get(1)?.as_f64()?;
        (x, y, width, height)
    };

    if !(x.is_finite() && y.is_finite() && width.is_finite() && height.is_finite()) {
        return None;
    }
    Some(BBoxXYXY::from_xyxy(x, y, x + width, y + height))
}

fn resolve_dimensions(
    capture: &Map<String, Value>,
    base_dir: &Path,
    file_name: &str,
    objects: &[ParsedObject],
) -> (u32, u32) {
    if let Some((width, height)) = dimension_from_object(capture) {
        return (width, height);
    }

    for candidate in [
        base_dir.join(file_name),
        base_dir.join("images").join(file_name),
    ] {
        if let Ok(size) = imagesize::size(candidate) {
            return (size.width as u32, size.height as u32);
        }
    }

    let max_x = objects
        .iter()
        .map(|object| object.bbox.xmax())
        .fold(0.0, f64::max)
        .ceil()
        .max(1.0) as u32;
    let max_y = objects
        .iter()
        .map(|object| object.bbox.ymax())
        .fold(0.0, f64::max)
        .ceil()
        .max(1.0) as u32;
    (max_x, max_y)
}

fn dimension_from_object(obj: &Map<String, Value>) -> Option<(u32, u32)> {
    if let (Some(width), Some(height)) = (
        number_field(obj, &["width", "image_width", "imageWidth"]),
        number_field(obj, &["height", "image_height", "imageHeight"]),
    ) {
        if width > 0.0 && height > 0.0 {
            return Some((width.round() as u32, height.round() as u32));
        }
    }

    let dimension = obj.get("dimension").and_then(Value::as_array)?;
    if dimension.len() < 2 {
        return None;
    }
    let width = dimension.first()?.as_f64()?;
    let height = dimension.get(1)?.as_f64()?;
    if width > 0.0 && height > 0.0 {
        Some((width.round() as u32, height.round() as u32))
    } else {
        None
    }
}

fn label_name_for_value(
    value: &Map<String, Value>,
    label_id: Option<&str>,
    definitions: &Definitions,
) -> Option<String> {
    if let Some(name) = string_field(value, &["label_name", "labelName", "label", "name"]) {
        return Some(name);
    }
    if let Some(label_id) = label_id {
        if let Some(name) = definitions.labels_by_id.get(label_id) {
            return Some(name.clone());
        }
        return Some(format!("label_{label_id}"));
    }
    None
}

fn read_definitions_near_file(path: &Path) -> Result<Definitions, PanlabelError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut definitions = read_definitions_in_dir(parent)?;
    if definitions.labels_by_id.is_empty() {
        if let Some(grandparent) = parent.parent() {
            definitions = read_definitions_in_dir(grandparent)?;
        }
    }
    Ok(definitions)
}

fn read_definitions_in_dir(path: &Path) -> Result<Definitions, PanlabelError> {
    let mut definitions = Definitions::default();
    let candidate = path.join("annotation_definitions.json");
    if candidate.is_file() {
        let file = File::open(&candidate).map_err(PanlabelError::Io)?;
        let reader = BufReader::new(file);
        let value: Value = serde_json::from_reader(reader).map_err(|source| {
            PanlabelError::UnityPerceptionJsonParse {
                path: candidate.clone(),
                source,
            }
        })?;
        collect_definition_labels(&value, &mut definitions);
    }
    Ok(definitions)
}

fn collect_definition_labels(value: &Value, definitions: &mut Definitions) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_definition_labels(item, definitions);
            }
        }
        Value::Object(obj) => {
            if let Some(spec) = obj.get("spec").and_then(Value::as_array) {
                let looks_bbox = obj.values().any(|v| {
                    v.as_str()
                        .map(|s| {
                            let lower = s.to_ascii_lowercase();
                            lower.contains("boundingbox2d") || lower.contains("bounding_box_2d")
                        })
                        .unwrap_or(false)
                });
                if looks_bbox || !spec.is_empty() {
                    for label in spec {
                        if let Some(label_obj) = label.as_object() {
                            let label_id = stringish_field(label_obj, &["label_id", "labelId"]);
                            let label_name = string_field(label_obj, &["label_name", "labelName"]);
                            if let (Some(label_id), Some(label_name)) = (label_id, label_name) {
                                definitions
                                    .labels_by_id
                                    .insert(label_id, label_name.clone());
                                if !definitions.label_order.contains(&label_name) {
                                    definitions.label_order.push(label_name);
                                }
                            }
                        }
                    }
                }
            }
            for key in [
                "annotationDefinitions",
                "annotation_definitions",
                "definitions",
            ] {
                if let Some(child) = obj.get(key) {
                    collect_definition_labels(child, definitions);
                }
            }
        }
        _ => {}
    }
}

fn collect_unity_json_paths(path: &Path) -> Result<Vec<PathBuf>, PanlabelError> {
    let mut paths = Vec::new();
    for entry in walkdir::WalkDir::new(path).follow_links(true) {
        let entry = entry.map_err(|source| {
            invalid(
                path,
                format!("failed while scanning Unity Perception directory: {source}"),
            )
        })?;
        if !entry.file_type().is_file() || !is_json_file_path(entry.path()) {
            continue;
        }
        let candidate = entry.path();
        if candidate
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.eq_ignore_ascii_case("annotation_definitions.json"))
            .unwrap_or(false)
        {
            continue;
        }
        let contents = match fs::read_to_string(candidate) {
            Ok(contents) => contents,
            Err(_) => continue,
        };
        match serde_json::from_str::<Value>(&contents) {
            Ok(value) => {
                if is_likely_unity_perception_file(&value) {
                    paths.push(candidate.to_path_buf());
                }
            }
            Err(source) if is_plausible_unity_json_path(candidate) => {
                return Err(PanlabelError::UnityPerceptionJsonParse {
                    path: candidate.to_path_buf(),
                    source,
                });
            }
            Err(_) => {}
        }
    }
    paths.sort();
    Ok(paths)
}

fn is_plausible_unity_json_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            name.ends_with(".frame_data.json")
                || name.starts_with("captures") && name.ends_with(".json")
        })
        .unwrap_or(false)
}

pub fn is_likely_unity_perception_file(value: &Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    let Some(captures) = obj.get("captures").and_then(Value::as_array) else {
        return false;
    };
    captures.iter().any(|capture| {
        let Some(capture_obj) = capture.as_object() else {
            return false;
        };
        capture_obj
            .get("filename")
            .and_then(Value::as_str)
            .is_some()
            && capture_obj
                .get("annotations")
                .and_then(Value::as_array)
                .is_some()
    })
}

fn warn_skip_non_bbox(
    path: &Path,
    file_name: &str,
    annotation_index: usize,
    annotation: &Map<String, Value>,
) {
    let kind = string_field(annotation, &["@type", "type", "name"])
        .unwrap_or_else(|| "unknown".to_string());
    eprintln!(
        "Warning: skipped Unity Perception non-bbox annotation {annotation_index} ({kind}) for capture '{file_name}' in {}",
        path.display()
    );
}

fn category_names_in_order(parsed: &[ParsedCapture], definitions: &Definitions) -> Vec<String> {
    let observed: BTreeSet<String> = parsed
        .iter()
        .flat_map(|capture| {
            capture
                .objects
                .iter()
                .map(|object| object.category_name.clone())
        })
        .collect();
    let mut names = Vec::new();
    for name in &definitions.label_order {
        if observed.contains(name) && !names.contains(name) {
            names.push(name.clone());
        }
    }
    for name in observed {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn ensure_unique_file_names(captures: &[ParsedCapture], path: &Path) -> Result<(), PanlabelError> {
    let mut seen = BTreeSet::new();
    for capture in captures {
        if !seen.insert(capture.file_name.clone()) {
            return Err(invalid(
                path,
                format!("duplicate derived image file_name '{}'", capture.file_name),
            ));
        }
    }
    Ok(())
}

fn image_to_frame_json(
    image: &Image,
    annotations: Vec<&Annotation>,
    category_lookup: &BTreeMap<CategoryId, String>,
    category_label_ids: &BTreeMap<CategoryId, u64>,
    frame_idx: usize,
) -> Value {
    let sensor_id = image
        .attributes
        .get(ATTR_SENSOR_ID)
        .map(String::as_str)
        .unwrap_or(DEFAULT_SENSOR_ID);
    let capture_id = image
        .attributes
        .get(ATTR_CAPTURE_ID)
        .map(String::as_str)
        .unwrap_or(sensor_id);
    let frame = attr_json_or_default(image, ATTR_FRAME, json!(frame_idx));
    let sequence = attr_json_or_default(image, ATTR_SEQUENCE, json!(0));
    let step = attr_json_or_default(image, ATTR_STEP, json!(frame_idx));
    let timestamp = attr_json_or_default(image, ATTR_TIMESTAMP, json!(frame_idx as f64));
    let values: Vec<Value> = annotations
        .into_iter()
        .filter_map(|ann| {
            let label_name = category_lookup.get(&ann.category_id)?;
            let label_id = category_label_ids
                .get(&ann.category_id)
                .copied()
                .unwrap_or(0);
            Some(json!({
                "label_id": label_id,
                "label_name": label_name,
                "instance_id": ann.id.as_u64(),
                "x": ann.bbox.xmin(),
                "y": ann.bbox.ymin(),
                "width": ann.bbox.width(),
                "height": ann.bbox.height()
            }))
        })
        .collect();

    let mut frame_json = json!({
        "frame": frame,
        "sequence": sequence,
        "step": step,
        "timestamp": timestamp,
        "captures": [{
            "@type": "type.unity.com/unity.solo.RGBCamera",
            "id": capture_id,
            "description": "",
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "velocity": [0.0, 0.0, 0.0],
            "acceleration": [0.0, 0.0, 0.0],
            "filename": image.file_name,
            "imageFormat": image_format(&image.file_name),
            "dimension": [image.width, image.height],
            "projection": "",
            "matrix": [],
            "annotations": [{
                "@type": BBOX_ANNOTATION_TYPE,
                "id": DEFAULT_ANNOTATION_DEFINITION_ID,
                "sensorId": sensor_id,
                "description": "2D bounding boxes exported by panlabel",
                "annotation_definition": DEFAULT_ANNOTATION_DEFINITION_ID,
                "values": values
            }]
        }]
    });
    if let Some(sequence_id) = image.attributes.get(ATTR_SEQUENCE_ID) {
        if let Some(capture) = frame_json
            .get_mut("captures")
            .and_then(Value::as_array_mut)
            .and_then(|captures| captures.first_mut())
            .and_then(Value::as_object_mut)
        {
            capture.insert(
                "sequence_id".to_string(),
                Value::String(sequence_id.clone()),
            );
        }
    }
    frame_json
}

fn clear_panlabel_frame_files(sequence_dir: &Path) -> Result<(), PanlabelError> {
    if !sequence_dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(sequence_dir).map_err(PanlabelError::Io)? {
        let entry = entry.map_err(PanlabelError::Io)?;
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with(".frame_data.json"))
                .unwrap_or(false)
        {
            fs::remove_file(path).map_err(PanlabelError::Io)?;
        }
    }
    Ok(())
}

fn write_annotation_definitions(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let category_label_ids = writer_category_label_ids(dataset);
    let mut categories = dataset.categories.iter().collect::<Vec<_>>();
    categories.sort_by_key(|category| category.id);
    let spec: Vec<Value> = categories
        .into_iter()
        .filter_map(|category| {
            category_label_ids.get(&category.id).map(|label_id| {
                json!({
                    "label_id": label_id,
                    "label_name": category.name
                })
            })
        })
        .collect();
    let value = json!({
        "version": "1.0.0",
        "annotationDefinitions": [{
            "@type": BBOX_DEFINITION_TYPE,
            "id": DEFAULT_ANNOTATION_DEFINITION_ID,
            "description": "2D bounding boxes exported by panlabel",
            "format": "json",
            "spec": spec
        }]
    });
    let output_path = path.join("annotation_definitions.json");
    let file = File::create(&output_path).map_err(PanlabelError::Io)?;
    serde_json::to_writer_pretty(file, &value).map_err(|source| {
        PanlabelError::UnityPerceptionJsonWrite {
            path: output_path,
            source,
        }
    })
}

fn sorted_images(dataset: &Dataset) -> Vec<&Image> {
    let mut images = dataset.images.iter().collect::<Vec<_>>();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name).then_with(|| a.id.cmp(&b.id)));
    images
}

fn annotations_by_image(dataset: &Dataset) -> BTreeMap<ImageId, Vec<&Annotation>> {
    let mut map: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        map.entry(ann.image_id).or_default().push(ann);
    }
    for anns in map.values_mut() {
        anns.sort_by_key(|ann| ann.id);
    }
    map
}

fn category_lookup(dataset: &Dataset) -> BTreeMap<CategoryId, String> {
    dataset
        .categories
        .iter()
        .map(|category| (category.id, category.name.clone()))
        .collect()
}

fn writer_category_label_ids(dataset: &Dataset) -> BTreeMap<CategoryId, u64> {
    let mut categories = dataset.categories.iter().collect::<Vec<_>>();
    categories.sort_by_key(|category| category.id);
    categories
        .into_iter()
        .enumerate()
        .map(|(idx, category)| (category.id, (idx + 1) as u64))
        .collect()
}

fn attr_json_or_default(image: &Image, key: &str, default: Value) -> Value {
    image
        .attributes
        .get(key)
        .and_then(|value| serde_json::from_str::<Value>(value).ok())
        .unwrap_or(default)
}

fn image_format(file_name: &str) -> String {
    Path::new(file_name)
        .extension()
        .and_then(|ext| ext.to_str())
        .filter(|ext| !ext.is_empty())
        .unwrap_or("png")
        .to_ascii_uppercase()
}

fn string_field(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| obj.get(*key).and_then(Value::as_str))
        .filter(|value| !value.trim().is_empty())
        .map(ToString::to_string)
}

fn stringish_field(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    for key in keys {
        let Some(value) = obj.get(*key) else {
            continue;
        };
        if let Some(text) = value.as_str() {
            if !text.trim().is_empty() {
                return Some(text.to_string());
            }
        } else if value.is_number() || value.is_boolean() {
            return Some(value.to_string());
        }
    }
    None
}

fn number_field(obj: &Map<String, Value>, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| obj.get(*key).and_then(Value::as_f64))
}

fn insert_opt(map: &mut BTreeMap<String, String>, key: &str, value: Option<String>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        map.insert(key.to_string(), value);
    }
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
    PanlabelError::UnityPerceptionJsonInvalid {
        path: path.to_path_buf(),
        message: message.into(),
    }
}
