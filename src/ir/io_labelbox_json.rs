//! Labelbox JSON/NDJSON export-row reader and writer.
//!
//! The reader supports Labelbox's current export-row shape as NDJSON/JSONL,
//! a single JSON object row, or a JSON array of rows. Each row is expected to
//! contain `data_row`, `media_attributes`, and nested
//! `projects.*.labels[].annotations.objects[]` entries. Bounding boxes are
//! imported directly from `bounding_box` / `bbox` XYWH geometry. Polygons are
//! flattened to axis-aligned bounding-box envelopes. Other object annotation
//! kinds (points, masks, lines, etc.) are skipped with warnings while the image
//! row is still preserved in the IR.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::Path;

use serde_json::{Map, Value};

use super::io_adapter_common::{
    basename_from_uri_or_path, has_json_lines_extension, is_safe_relative_image_ref,
};
use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

pub const ATTR_DATA_ROW_ID: &str = "labelbox_data_row_id";
pub const ATTR_ROW_DATA: &str = "labelbox_row_data";
pub const ATTR_GLOBAL_KEY: &str = "labelbox_global_key";
pub const ATTR_PROJECT_ID: &str = "labelbox_project_id";
pub const ATTR_LABEL_INDEX: &str = "labelbox_label_index";
pub const ATTR_FEATURE_ID: &str = "labelbox_feature_id";
pub const ATTR_ANNOTATION_KIND: &str = "labelbox_annotation_kind";
pub const ATTR_GEOMETRY_TYPE: &str = "labelbox_geometry_type";
pub const ATTR_POLYGON_ENVELOPED: &str = "labelbox_polygon_enveloped";
pub const ATTR_SKIPPED_OBJECTS: &str = "labelbox_skipped_objects";
pub const ATTR_POLYGON_ENVELOPES: &str = "labelbox_polygon_envelopes";

const DEFAULT_PROJECT_ID: &str = "panlabel-project";
const STRING_HELPER_PATH: &str = "<labelbox string>";

#[derive(Debug)]
struct ParsedRow {
    source_index: usize,
    file_name: String,
    width: u32,
    height: u32,
    data_row_id: Option<String>,
    row_data: Option<String>,
    global_key: Option<String>,
    objects: Vec<ParsedObject>,
}

#[derive(Debug)]
struct ParsedObject {
    source_key: ObjectSourceKey,
    category_name: String,
    bbox: BBoxXYXY<Pixel>,
    geometry_type: GeometryType,
    feature_id: Option<String>,
    annotation_kind: Option<String>,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct ObjectSourceKey {
    project_id: String,
    label_index: usize,
    object_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeometryType {
    BoundingBox,
    PolygonEnvelope,
}

#[derive(Debug, Default)]
struct ReaderStats {
    skipped_objects: usize,
    polygon_envelopes: usize,
}

/// Read a Labelbox current export-row file from NDJSON/JSONL, a single JSON row,
/// or a JSON array of rows.
pub fn read_labelbox_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let contents = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    from_labelbox_str_with_path(&contents, path, is_jsonl_path(path))
}

/// Write a dataset as Labelbox export rows. `.ndjson` / `.jsonl` paths receive
/// newline-delimited rows; other paths receive a JSON array.
pub fn write_labelbox_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
    }

    let output = if is_jsonl_path(path) {
        to_labelbox_ndjson_string_with_path(dataset, path)?
    } else {
        to_labelbox_json_array_string_with_path(dataset, path)?
    };
    fs::write(path, output).map_err(PanlabelError::Io)
}

pub fn from_labelbox_json_str(json: &str) -> Result<Dataset, PanlabelError> {
    from_labelbox_str_with_path(json, Path::new(STRING_HELPER_PATH), false)
}

pub fn from_labelbox_ndjson_str(ndjson: &str) -> Result<Dataset, PanlabelError> {
    from_labelbox_str_with_path(ndjson, Path::new(STRING_HELPER_PATH), true)
}

pub fn to_labelbox_json_array_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    to_labelbox_json_array_string_with_path(dataset, Path::new(STRING_HELPER_PATH))
}

pub fn to_labelbox_ndjson_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    to_labelbox_ndjson_string_with_path(dataset, Path::new(STRING_HELPER_PATH))
}

fn from_labelbox_str_with_path(
    contents: &str,
    path: &Path,
    force_jsonl: bool,
) -> Result<Dataset, PanlabelError> {
    let rows = if force_jsonl {
        parse_jsonl_rows(contents, path)?
    } else {
        parse_json_or_rows(contents, path)?
    };
    rows_to_dataset(rows, path)
}

fn parse_jsonl_rows(contents: &str, path: &Path) -> Result<Vec<Value>, PanlabelError> {
    let mut rows = Vec::new();
    for (idx, line) in contents.lines().enumerate() {
        let line_num = idx + 1;
        if line.trim().is_empty() {
            continue;
        }
        let value =
            serde_json::from_str(line).map_err(|source| PanlabelError::LabelboxJsonlParse {
                path: path.to_path_buf(),
                line: line_num,
                message: source.to_string(),
            })?;
        rows.push(value);
    }
    if rows.is_empty() {
        return Err(PanlabelError::LabelboxJsonInvalid {
            path: path.to_path_buf(),
            message: "Labelbox JSONL file contains no rows".to_string(),
        });
    }
    Ok(rows)
}

fn parse_json_or_rows(contents: &str, path: &Path) -> Result<Vec<Value>, PanlabelError> {
    let value: Value =
        serde_json::from_str(contents).map_err(|source| PanlabelError::LabelboxJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    match value {
        Value::Array(rows) => {
            if rows.is_empty() {
                Err(PanlabelError::LabelboxJsonInvalid {
                    path: path.to_path_buf(),
                    message: "Labelbox JSON array contains no rows".to_string(),
                })
            } else {
                Ok(rows)
            }
        }
        Value::Object(_) => Ok(vec![value]),
        other => Err(PanlabelError::LabelboxJsonInvalid {
            path: path.to_path_buf(),
            message: format!(
                "expected a Labelbox export row object or array of rows, got {}",
                value_type_name(&other)
            ),
        }),
    }
}

fn rows_to_dataset(rows: Vec<Value>, path: &Path) -> Result<Dataset, PanlabelError> {
    let mut stats = ReaderStats::default();
    let mut parsed_rows = Vec::with_capacity(rows.len());
    let mut seen_file_names = BTreeSet::new();

    for (idx, row) in rows.iter().enumerate() {
        let parsed = parse_row(path, idx, row, &mut stats)?;
        if !seen_file_names.insert(parsed.file_name.clone()) {
            return Err(PanlabelError::LabelboxJsonInvalid {
                path: path.to_path_buf(),
                message: format!("duplicate derived image file_name '{}'", parsed.file_name),
            });
        }
        parsed_rows.push(parsed);
    }

    parsed_rows.sort_by(|a, b| {
        a.file_name
            .cmp(&b.file_name)
            .then_with(|| a.source_index.cmp(&b.source_index))
    });

    let mut category_names = BTreeSet::new();
    for row in &parsed_rows {
        for object in &row.objects {
            category_names.insert(object.category_name.clone());
        }
    }
    let categories: Vec<Category> = category_names
        .iter()
        .enumerate()
        .map(|(idx, name)| Category::new((idx + 1) as u64, name.clone()))
        .collect();
    let category_id_by_name: BTreeMap<String, CategoryId> = categories
        .iter()
        .map(|category| (category.name.clone(), category.id))
        .collect();

    let mut images = Vec::with_capacity(parsed_rows.len());
    let mut annotations = Vec::new();
    let mut next_ann_id = 1u64;

    for (image_idx, row) in parsed_rows.iter().enumerate() {
        let image_id = ImageId::new((image_idx + 1) as u64);
        let mut image = Image::new(image_id, row.file_name.clone(), row.width, row.height);
        if let Some(value) = row.data_row_id.as_ref().filter(|value| !value.is_empty()) {
            image
                .attributes
                .insert(ATTR_DATA_ROW_ID.to_string(), value.clone());
        }
        if let Some(value) = row.row_data.as_ref().filter(|value| !value.is_empty()) {
            image
                .attributes
                .insert(ATTR_ROW_DATA.to_string(), value.clone());
        }
        if let Some(value) = row.global_key.as_ref().filter(|value| !value.is_empty()) {
            image
                .attributes
                .insert(ATTR_GLOBAL_KEY.to_string(), value.clone());
        }
        images.push(image);

        let mut objects = row.objects.iter().collect::<Vec<_>>();
        objects.sort_by(|a, b| a.source_key.cmp(&b.source_key));
        for object in objects {
            let Some(category_id) = category_id_by_name.get(&object.category_name).copied() else {
                continue;
            };
            let mut annotation = Annotation::new(
                AnnotationId::new(next_ann_id),
                image_id,
                category_id,
                object.bbox,
            );
            annotation.attributes.insert(
                ATTR_PROJECT_ID.to_string(),
                object.source_key.project_id.clone(),
            );
            annotation.attributes.insert(
                ATTR_LABEL_INDEX.to_string(),
                object.source_key.label_index.to_string(),
            );
            if let Some(value) = object.feature_id.as_ref().filter(|value| !value.is_empty()) {
                annotation
                    .attributes
                    .insert(ATTR_FEATURE_ID.to_string(), value.clone());
            }
            if let Some(value) = object
                .annotation_kind
                .as_ref()
                .filter(|value| !value.is_empty())
            {
                annotation
                    .attributes
                    .insert(ATTR_ANNOTATION_KIND.to_string(), value.clone());
            }
            match object.geometry_type {
                GeometryType::BoundingBox => {
                    annotation
                        .attributes
                        .insert(ATTR_GEOMETRY_TYPE.to_string(), "bbox".to_string());
                }
                GeometryType::PolygonEnvelope => {
                    annotation
                        .attributes
                        .insert(ATTR_GEOMETRY_TYPE.to_string(), "polygon".to_string());
                    annotation
                        .attributes
                        .insert(ATTR_POLYGON_ENVELOPED.to_string(), "true".to_string());
                }
            }
            annotations.push(annotation);
            next_ann_id += 1;
        }
    }

    let mut info_attributes = BTreeMap::new();
    if stats.skipped_objects > 0 {
        info_attributes.insert(
            ATTR_SKIPPED_OBJECTS.to_string(),
            stats.skipped_objects.to_string(),
        );
    }
    if stats.polygon_envelopes > 0 {
        info_attributes.insert(
            ATTR_POLYGON_ENVELOPES.to_string(),
            stats.polygon_envelopes.to_string(),
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

fn parse_row(
    path: &Path,
    row_index: usize,
    value: &Value,
    stats: &mut ReaderStats,
) -> Result<ParsedRow, PanlabelError> {
    let row = value
        .as_object()
        .ok_or_else(|| invalid(path, "Labelbox row must be a JSON object"))?;
    let data_row = required_object(row.get("data_row"), path, "data_row")?;
    let media_attributes = required_object(row.get("media_attributes"), path, "media_attributes")?;

    let external_id = optional_string(data_row, "external_id");
    let row_data = optional_string(data_row, "row_data");
    let global_key = optional_string(data_row, "global_key");
    let data_row_id = optional_string(data_row, "id");

    let file_name = external_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .or_else(|| row_data.as_deref().and_then(derive_file_name_from_ref))
        .or_else(|| global_key.clone())
        .ok_or_else(|| {
            invalid(
                path,
                "data_row must contain external_id, row_data, or global_key",
            )
        })?;
    validate_relative_image_ref(&file_name, path)?;

    let width = parse_required_u32(
        media_attributes.get("width"),
        path,
        "media_attributes.width",
    )?;
    let height = parse_required_u32(
        media_attributes.get("height"),
        path,
        "media_attributes.height",
    )?;

    let mut objects = Vec::new();
    let projects = required_object(row.get("projects"), path, "projects")?;
    let mut project_entries: Vec<(&String, &Value)> = projects.iter().collect();
    project_entries.sort_by(|a, b| a.0.cmp(b.0));

    for (project_id, project_value) in project_entries {
        let Some(project) = project_value.as_object() else {
            warn_skip(
                path,
                row_index,
                project_id,
                None,
                "project value is not an object",
            );
            stats.skipped_objects += 1;
            continue;
        };
        let Some(labels) = project.get("labels").and_then(Value::as_array) else {
            continue;
        };
        for (label_index, label_value) in labels.iter().enumerate() {
            let Some(label) = label_value.as_object() else {
                continue;
            };
            let Some(label_objects) = label
                .get("annotations")
                .and_then(Value::as_object)
                .and_then(|annotations| annotations.get("objects"))
                .and_then(Value::as_array)
            else {
                continue;
            };
            for (object_index, object_value) in label_objects.iter().enumerate() {
                match parse_object(path, project_id, label_index, object_index, object_value)? {
                    Some(parsed) => {
                        if parsed.geometry_type == GeometryType::PolygonEnvelope {
                            stats.polygon_envelopes += 1;
                        }
                        objects.push(parsed);
                    }
                    None => {
                        let reason = unsupported_reason(object_value);
                        warn_skip(path, row_index, project_id, Some(object_index), &reason);
                        stats.skipped_objects += 1;
                    }
                }
            }
        }
    }

    Ok(ParsedRow {
        source_index: row_index,
        file_name,
        width,
        height,
        data_row_id,
        row_data,
        global_key,
        objects,
    })
}

fn parse_object(
    path: &Path,
    project_id: &str,
    label_index: usize,
    object_index: usize,
    value: &Value,
) -> Result<Option<ParsedObject>, PanlabelError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid(path, "annotations.objects[] entry must be an object"))?;
    let annotation_kind = optional_string(object, "annotation_kind")
        .or_else(|| optional_string(object, "kind"))
        .or_else(|| optional_string(object, "type"));
    let feature_id = optional_string(object, "feature_id");

    let geometry = if let Some(bbox_value) =
        object.get("bounding_box").or_else(|| object.get("bbox"))
    {
        let bbox = parse_bbox(path, bbox_value, "object.bounding_box")?;
        Some((bbox, GeometryType::BoundingBox))
    } else if let Some(polygon_value) = object.get("polygon") {
        let bbox = parse_polygon_envelope(path, polygon_value, "object.polygon")?;
        Some((bbox, GeometryType::PolygonEnvelope))
    } else if annotation_kind
        .as_deref()
        .map(kind_is_bbox)
        .unwrap_or(false)
    {
        return Err(invalid(
            path,
            format!("object {} in project '{}' is a bounding-box annotation but has no bounding_box geometry", object_index, project_id),
        ));
    } else if annotation_kind
        .as_deref()
        .map(kind_is_polygon)
        .unwrap_or(false)
    {
        return Err(invalid(
            path,
            format!(
                "object {} in project '{}' is a polygon annotation but has no polygon geometry",
                object_index, project_id
            ),
        ));
    } else {
        None
    };

    let Some((bbox, geometry_type)) = geometry else {
        return Ok(None);
    };

    let category_name = optional_string(object, "name")
        .or_else(|| optional_string(object, "value"))
        .or_else(|| optional_string(object, "title"))
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid(path, "detection object is missing non-empty 'name' label"))?;

    Ok(Some(ParsedObject {
        source_key: ObjectSourceKey {
            project_id: project_id.to_string(),
            label_index,
            object_index,
        },
        category_name,
        bbox,
        geometry_type,
        feature_id,
        annotation_kind,
    }))
}

fn parse_bbox(
    path: &Path,
    value: &Value,
    field_name: &str,
) -> Result<BBoxXYXY<Pixel>, PanlabelError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid(path, format!("{field_name} must be an object")))?;
    let left = parse_required_f64(
        object.get("left").or_else(|| object.get("x")),
        path,
        &format!("{field_name}.left"),
    )?;
    let top = parse_required_f64(
        object.get("top").or_else(|| object.get("y")),
        path,
        &format!("{field_name}.top"),
    )?;
    let width = parse_required_f64(object.get("width"), path, &format!("{field_name}.width"))?;
    let height = parse_required_f64(object.get("height"), path, &format!("{field_name}.height"))?;
    Ok(BBoxXYXY::<Pixel>::from_xywh(left, top, width, height))
}

fn parse_polygon_envelope(
    path: &Path,
    value: &Value,
    field_name: &str,
) -> Result<BBoxXYXY<Pixel>, PanlabelError> {
    let points = if let Some(points) = value.as_array() {
        points
    } else if let Some(points) = value
        .as_object()
        .and_then(|object| object.get("points"))
        .and_then(Value::as_array)
    {
        points
    } else {
        return Err(invalid(path, format!("{field_name} must be a point array")));
    };

    if points.is_empty() {
        return Err(invalid(
            path,
            format!("{field_name} must contain at least one point"),
        ));
    }

    let mut xmin = f64::INFINITY;
    let mut ymin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for (idx, point) in points.iter().enumerate() {
        let point_obj = point.as_object().ok_or_else(|| {
            invalid(
                path,
                format!("{field_name}[{idx}] must be an object with x/y"),
            )
        })?;
        let x = parse_required_f64(point_obj.get("x"), path, &format!("{field_name}[{idx}].x"))?;
        let y = parse_required_f64(point_obj.get("y"), path, &format!("{field_name}[{idx}].y"))?;
        xmin = xmin.min(x);
        ymin = ymin.min(y);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
    }
    Ok(BBoxXYXY::<Pixel>::from_xyxy(xmin, ymin, xmax, ymax))
}

fn to_labelbox_json_array_string_with_path(
    dataset: &Dataset,
    path: &Path,
) -> Result<String, PanlabelError> {
    let rows = dataset_to_labelbox_rows(dataset, path)?;
    serde_json::to_string_pretty(&rows).map_err(|source| PanlabelError::LabelboxJsonWrite {
        path: path.to_path_buf(),
        source,
    })
}

fn to_labelbox_ndjson_string_with_path(
    dataset: &Dataset,
    path: &Path,
) -> Result<String, PanlabelError> {
    let rows = dataset_to_labelbox_rows(dataset, path)?;
    let mut out = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut out, &row).map_err(|source| {
            PanlabelError::LabelboxJsonWrite {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writeln!(&mut out).map_err(PanlabelError::Io)?;
    }
    String::from_utf8(out).map_err(|source| PanlabelError::LabelboxJsonInvalid {
        path: path.to_path_buf(),
        message: source.to_string(),
    })
}

fn dataset_to_labelbox_rows(dataset: &Dataset, path: &Path) -> Result<Vec<Value>, PanlabelError> {
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
            return Err(invalid(
                path,
                format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            ));
        }
        if !category_lookup.contains_key(&ann.category_id) {
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

    let mut images_sorted: Vec<&Image> = dataset.images.iter().collect();
    images_sorted.sort_by(|a, b| a.file_name.cmp(&b.file_name).then_with(|| a.id.cmp(&b.id)));

    let mut rows = Vec::with_capacity(images_sorted.len());
    for image in images_sorted {
        if image.width == 0 || image.height == 0 {
            return Err(invalid(
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

        let mut anns = anns_by_image.remove(&image.id).unwrap_or_default();
        anns.sort_by_key(|ann| ann.id);
        let mut objects = Vec::with_capacity(anns.len());
        for ann in anns {
            let category = category_lookup[&ann.category_id];
            let (left, top, width, height) = ann.bbox.to_xywh();
            if width < 0.0 || height < 0.0 {
                return Err(invalid(
                    path,
                    format!(
                        "annotation {} has negative bbox width/height after XYWH conversion ({width}, {height})",
                        ann.id.as_u64()
                    ),
                ));
            }

            let mut bbox = Map::new();
            bbox.insert("top".to_string(), Value::from(top));
            bbox.insert("left".to_string(), Value::from(left));
            bbox.insert("height".to_string(), Value::from(height));
            bbox.insert("width".to_string(), Value::from(width));

            let mut object = Map::new();
            object.insert(
                "feature_id".to_string(),
                Value::String(format!("panlabel-ann-{}", ann.id.as_u64())),
            );
            object.insert("name".to_string(), Value::String(category.name.clone()));
            object.insert(
                "annotation_kind".to_string(),
                Value::String("ImageBoundingBox".to_string()),
            );
            object.insert("classifications".to_string(), Value::Array(Vec::new()));
            object.insert("bounding_box".to_string(), Value::Object(bbox));
            objects.push(Value::Object(object));
        }

        let row = labelbox_row_for_image(image, objects);
        rows.push(row);
    }

    Ok(rows)
}

fn labelbox_row_for_image(image: &Image, objects: Vec<Value>) -> Value {
    let mut data_row = Map::new();
    data_row.insert(
        "id".to_string(),
        Value::String(
            image
                .attributes
                .get(ATTR_DATA_ROW_ID)
                .cloned()
                .unwrap_or_else(|| format!("panlabel-image-{}", image.id.as_u64())),
        ),
    );
    data_row.insert(
        "external_id".to_string(),
        Value::String(image.file_name.clone()),
    );
    data_row.insert(
        "row_data".to_string(),
        Value::String(
            image
                .attributes
                .get(ATTR_ROW_DATA)
                .cloned()
                .unwrap_or_else(|| image.file_name.clone()),
        ),
    );
    if let Some(global_key) = image.attributes.get(ATTR_GLOBAL_KEY) {
        data_row.insert("global_key".to_string(), Value::String(global_key.clone()));
    }

    let mut media_attributes = Map::new();
    media_attributes.insert("height".to_string(), Value::from(image.height));
    media_attributes.insert("width".to_string(), Value::from(image.width));

    let mut annotations = Map::new();
    annotations.insert("objects".to_string(), Value::Array(objects));

    let mut label = Map::new();
    label.insert(
        "label_kind".to_string(),
        Value::String("Default".to_string()),
    );
    label.insert("annotations".to_string(), Value::Object(annotations));

    let mut project = Map::new();
    project.insert(
        "labels".to_string(),
        Value::Array(vec![Value::Object(label)]),
    );

    let mut projects = Map::new();
    projects.insert(DEFAULT_PROJECT_ID.to_string(), Value::Object(project));

    let mut row = Map::new();
    row.insert("data_row".to_string(), Value::Object(data_row));
    row.insert(
        "media_attributes".to_string(),
        Value::Object(media_attributes),
    );
    row.insert("projects".to_string(), Value::Object(projects));
    Value::Object(row)
}

fn parse_required_f64(
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

fn parse_required_u32(
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
                format!("field '{field_name}' must be a positive integer"),
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

fn required_object<'a>(
    value: Option<&'a Value>,
    path: &Path,
    field_name: &str,
) -> Result<&'a Map<String, Value>, PanlabelError> {
    value.and_then(Value::as_object).ok_or_else(|| {
        invalid(
            path,
            format!("missing or invalid object field '{field_name}'"),
        )
    })
}

fn optional_string(object: &Map<String, Value>, key: &str) -> Option<String> {
    object.get(key).and_then(Value::as_str).map(str::to_string)
}

fn kind_is_bbox(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized.contains("bbox")
        || normalized.contains("boundingbox")
        || normalized.contains("bounding_box")
}

fn kind_is_polygon(value: &str) -> bool {
    value.to_ascii_lowercase().contains("polygon")
}

fn unsupported_reason(value: &Value) -> String {
    let Some(object) = value.as_object() else {
        return "object entry is not a JSON object".to_string();
    };
    let kind = optional_string(object, "annotation_kind")
        .or_else(|| optional_string(object, "kind"))
        .or_else(|| optional_string(object, "type"))
        .unwrap_or_else(|| "unknown".to_string());
    format!("unsupported non-detection object kind '{kind}'")
}

fn warn_skip(
    path: &Path,
    row_index: usize,
    project_id: &str,
    object_index: Option<usize>,
    reason: &str,
) {
    let object_desc = object_index
        .map(|idx| format!(" object {idx}"))
        .unwrap_or_default();
    eprintln!(
        "Warning: skipped Labelbox row {row_index} project '{project_id}'{object_desc} in {}: {reason}",
        path.display()
    );
}

fn derive_file_name_from_ref(raw: &str) -> Option<String> {
    basename_from_uri_or_path(raw)
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

fn invalid(path: &Path, message: impl Into<String>) -> PanlabelError {
    PanlabelError::LabelboxJsonInvalid {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn is_jsonl_path(path: &Path) -> bool {
    has_json_lines_extension(path)
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

/// Lightweight structural probe used by CLI auto-detection.
pub(crate) fn is_likely_labelbox_row(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object
        .get("data_row")
        .map(Value::is_object)
        .unwrap_or(false)
        && object
            .get("media_attributes")
            .map(Value::is_object)
            .unwrap_or(false)
        && object
            .get("projects")
            .map(Value::is_object)
            .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row_json() -> &'static str {
        r#"{
            "data_row": {"id": "dr-1", "external_id": "img1.jpg", "row_data": "s3://bucket/img1.jpg"},
            "media_attributes": {"width": 100, "height": 80},
            "projects": {
                "proj-1": {
                    "labels": [{
                        "label_kind": "Default",
                        "annotations": {
                            "objects": [
                                {"feature_id": "bbox-1", "name": "cat", "annotation_kind": "ImageBoundingBox", "bounding_box": {"top": 20, "left": 10, "height": 30, "width": 40}},
                                {"feature_id": "poly-1", "name": "dog", "annotation_kind": "ImagePolygon", "polygon": [{"x": 5, "y": 6}, {"x": 30, "y": 4}, {"x": 20, "y": 40}]},
                                {"feature_id": "point-1", "name": "nose", "annotation_kind": "ImagePoint", "point": {"x": 1, "y": 2}}
                            ]
                        }
                    }]
                }
            }
        }"#
    }

    fn sample_row_ndjson_line() -> String {
        let value: Value = serde_json::from_str(sample_row_json()).expect("sample row JSON");
        serde_json::to_string(&value).expect("compact sample row")
    }

    #[test]
    fn parses_single_row_with_bbox_polygon_and_skip_count() {
        let dataset = from_labelbox_json_str(sample_row_json()).expect("parse labelbox row");
        assert_eq!(dataset.images.len(), 1);
        assert_eq!(dataset.categories.len(), 2);
        assert_eq!(dataset.annotations.len(), 2);
        assert_eq!(dataset.images[0].file_name, "img1.jpg");
        assert_eq!(dataset.images[0].width, 100);
        assert_eq!(dataset.images[0].attributes[ATTR_DATA_ROW_ID], "dr-1");
        assert_eq!(dataset.info.attributes[ATTR_SKIPPED_OBJECTS], "1");
        assert_eq!(dataset.info.attributes[ATTR_POLYGON_ENVELOPES], "1");

        let bbox_ann = dataset
            .annotations
            .iter()
            .find(|ann| ann.attributes.get(ATTR_FEATURE_ID).map(String::as_str) == Some("bbox-1"))
            .expect("bbox annotation");
        assert_eq!(bbox_ann.bbox.xmin(), 10.0);
        assert_eq!(bbox_ann.bbox.ymin(), 20.0);
        assert_eq!(bbox_ann.bbox.xmax(), 50.0);
        assert_eq!(bbox_ann.bbox.ymax(), 50.0);

        let poly_ann = dataset
            .annotations
            .iter()
            .find(|ann| {
                ann.attributes
                    .get(ATTR_POLYGON_ENVELOPED)
                    .map(String::as_str)
                    == Some("true")
            })
            .expect("polygon annotation");
        assert_eq!(poly_ann.bbox.xmin(), 5.0);
        assert_eq!(poly_ann.bbox.ymin(), 4.0);
        assert_eq!(poly_ann.bbox.xmax(), 30.0);
        assert_eq!(poly_ann.bbox.ymax(), 40.0);
    }

    #[test]
    fn parses_ndjson_rows_and_preserves_empty_images() {
        let ndjson = format!(
            "{}\n{}\n",
            sample_row_ndjson_line(),
            r#"{"data_row":{"external_id":"empty.jpg"},"media_attributes":{"width":10,"height":12},"projects":{"proj":{"labels":[{"annotations":{"objects":[]}}]}}}"#
        );
        let dataset = from_labelbox_ndjson_str(&ndjson).expect("parse ndjson");
        assert_eq!(dataset.images.len(), 2);
        assert!(dataset
            .images
            .iter()
            .any(|image| image.file_name == "empty.jpg"));
    }

    #[test]
    fn rejects_zero_integer_dimensions() {
        let json = r#"{
            "data_row": {"external_id": "img.jpg"},
            "media_attributes": {"width": 0, "height": 10},
            "projects": {"proj": {"labels": [{"annotations": {"objects": []}}]}}
        }"#;
        let err = from_labelbox_json_str(json).expect_err("zero width should fail");
        match err {
            PanlabelError::LabelboxJsonInvalid { message, .. } => {
                assert!(message.contains("positive integer"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn allows_safe_file_names_containing_double_dots() {
        let json = r#"{
            "data_row": {"external_id": "cat..final.jpg"},
            "media_attributes": {"width": 10, "height": 10},
            "projects": {"proj": {"labels": [{"annotations": {"objects": []}}]}}
        }"#;
        let dataset = from_labelbox_json_str(json).expect("safe file name should parse");
        assert_eq!(dataset.images[0].file_name, "cat..final.jpg");
    }

    #[test]
    fn writer_emits_current_export_row_shape() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "img.jpg", 100, 80)],
            categories: vec![Category::new(1u64, "cat")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 50.0, 60.0),
            )],
            ..Default::default()
        };
        let json = to_labelbox_json_array_string(&dataset).expect("serialize");
        let rows: Value = serde_json::from_str(&json).expect("parse output");
        assert_eq!(rows[0]["data_row"]["external_id"], "img.jpg");
        assert_eq!(rows[0]["media_attributes"]["width"], 100);
        let object =
            &rows[0]["projects"][DEFAULT_PROJECT_ID]["labels"][0]["annotations"]["objects"][0];
        assert_eq!(object["name"], "cat");
        assert_eq!(object["annotation_kind"], "ImageBoundingBox");
        assert_eq!(object["bounding_box"]["left"], 10.0);
    }
}
