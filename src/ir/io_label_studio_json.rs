//! Label Studio JSON reader and writer.
//!
//! This adapter supports Label Studio task-export JSON (array of tasks) for
//! rectanglelabels object-detection bounding boxes.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

// ============================================================================
// Label Studio schema types (internal)
// ============================================================================

#[derive(Debug, Deserialize)]
struct LsTask {
    #[serde(default)]
    data: LsTaskData,
    #[serde(default)]
    annotations: Option<Vec<LsResultSet>>,
    #[serde(default)]
    completions: Option<Vec<LsResultSet>>,
    #[serde(default)]
    predictions: Option<Vec<LsResultSet>>,
}

#[derive(Debug, Default, Deserialize)]
struct LsTaskData {
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct LsResultSet {
    #[serde(default)]
    result: Vec<LsResult>,
}

#[derive(Debug, Deserialize)]
struct LsResult {
    #[serde(rename = "type")]
    result_type: String,
    #[serde(default)]
    value: Option<serde_json::Value>,
    #[serde(default)]
    original_width: Option<u32>,
    #[serde(default)]
    original_height: Option<u32>,
    #[serde(default)]
    rotation: Option<f64>,
    #[serde(default)]
    from_name: Option<String>,
    #[serde(default)]
    to_name: Option<String>,
    #[serde(default)]
    score: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct LsRectangleValue {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    #[serde(default)]
    rectanglelabels: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LsTaskOut {
    id: u64,
    data: LsTaskDataOut,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    annotations: Vec<LsResultSetOut>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    predictions: Vec<LsResultSetOut>,
}

#[derive(Debug, Serialize)]
struct LsTaskDataOut {
    image: String,
    width: u32,
    height: u32,
}

#[derive(Debug, Serialize)]
struct LsResultSetOut {
    result: Vec<LsResultOut>,
}

#[derive(Debug, Serialize)]
struct LsResultOut {
    #[serde(rename = "type")]
    result_type: &'static str,
    value: LsRectangleValueOut,
    original_width: u32,
    original_height: u32,
    from_name: String,
    to_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    rotation: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
}

#[derive(Debug, Serialize)]
struct LsRectangleValueOut {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    rectanglelabels: Vec<String>,
}

#[derive(Debug)]
struct ParsedTask {
    file_name: String,
    image_ref: String,
    width: u32,
    height: u32,
    from_name: Option<String>,
    to_name: Option<String>,
    rows: Vec<ParsedAnnotation>,
}

#[derive(Debug)]
struct ParsedAnnotation {
    label: String,
    bbox: BBoxXYXY<Pixel>,
    confidence: Option<f64>,
    attributes: BTreeMap<String, String>,
}

// ============================================================================
// Public API
// ============================================================================

/// Read Label Studio task-export JSON into panlabel IR.
pub fn read_label_studio_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);

    let tasks: Vec<LsTask> =
        serde_json::from_reader(reader).map_err(|source| PanlabelError::LabelStudioJsonParse {
            path: path.to_path_buf(),
            source,
        })?;

    ls_to_ir(tasks, path)
}

/// Write panlabel IR as Label Studio task-export JSON.
pub fn write_label_studio_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let tasks = ir_to_ls(dataset, path)?;

    let file = File::create(path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &tasks).map_err(|source| {
        PanlabelError::LabelStudioJsonWrite {
            path: path.to_path_buf(),
            source,
        }
    })
}

/// Parse Label Studio task-export JSON from string.
pub fn from_label_studio_str(json: &str) -> Result<Dataset, PanlabelError> {
    let path = Path::new("<string>");
    let tasks: Vec<LsTask> =
        serde_json::from_str(json).map_err(|source| PanlabelError::LabelStudioJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    ls_to_ir(tasks, path)
}

/// Parse Label Studio task-export JSON from bytes.
pub fn from_label_studio_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    let path = Path::new("<bytes>");
    let tasks: Vec<LsTask> =
        serde_json::from_slice(bytes).map_err(|source| PanlabelError::LabelStudioJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    ls_to_ir(tasks, path)
}

/// Serialize panlabel IR to Label Studio task-export JSON string.
pub fn to_label_studio_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let path = Path::new("<string>");
    let tasks = ir_to_ls(dataset, path)?;
    serde_json::to_string_pretty(&tasks).map_err(|source| PanlabelError::LabelStudioJsonWrite {
        path: path.to_path_buf(),
        source,
    })
}

// ============================================================================
// Conversion: Label Studio -> IR
// ============================================================================

fn ls_to_ir(tasks: Vec<LsTask>, path: &Path) -> Result<Dataset, PanlabelError> {
    if tasks.is_empty() {
        return Ok(Dataset::default());
    }

    let mut parsed_tasks = Vec::with_capacity(tasks.len());
    let mut seen_file_names = BTreeSet::new();

    for (task_idx, task) in tasks.into_iter().enumerate() {
        let image_ref = task
            .data
            .image
            .ok_or_else(|| invalid(path, format!("task[{task_idx}] missing data.image")))?;

        let file_name = derive_image_file_name(&image_ref).ok_or_else(|| {
            invalid(
                path,
                format!(
                    "task[{task_idx}] data.image '{}' does not contain a valid filename",
                    image_ref
                ),
            )
        })?;

        if !seen_file_names.insert(file_name.clone()) {
            return Err(invalid(
                path,
                format!(
                    "duplicate image basename '{}' derived from data.image; panlabel requires unique basenames",
                    file_name
                ),
            ));
        }

        let annotation_results =
            select_annotation_results(task.annotations, task.completions, path, task_idx)?;

        let prediction_results = select_prediction_results(task.predictions, path, task_idx)?;

        let mut from_names = BTreeSet::new();
        let mut to_names = BTreeSet::new();

        let mut dims: Option<(u32, u32)> = None;
        let mut rows = Vec::new();

        for (result_idx, result) in annotation_results.iter().enumerate() {
            let parsed = parse_result(
                result,
                path,
                task_idx,
                result_idx,
                "annotations",
                &mut dims,
                &mut from_names,
                &mut to_names,
            )?;
            rows.push(parsed);
        }

        for (result_idx, result) in prediction_results.iter().enumerate() {
            let parsed = parse_result(
                result,
                path,
                task_idx,
                result_idx,
                "predictions",
                &mut dims,
                &mut from_names,
                &mut to_names,
            )?;
            rows.push(parsed);
        }

        let (width, height) = if let Some((w, h)) = dims {
            (w, h)
        } else if let (Some(w), Some(h)) = (task.data.width, task.data.height) {
            (w, h)
        } else {
            return Err(invalid(
                path,
                format!(
                    "task[{task_idx}] has no results with original_width/original_height and no data.width/data.height fallback"
                ),
            ));
        };

        if from_names.len() > 1 {
            return Err(invalid(
                path,
                format!("task[{task_idx}] has inconsistent from_name values across results"),
            ));
        }
        if to_names.len() > 1 {
            return Err(invalid(
                path,
                format!("task[{task_idx}] has inconsistent to_name values across results"),
            ));
        }

        let from_name = from_names.into_iter().next();
        let to_name = to_names.into_iter().next();

        parsed_tasks.push(ParsedTask {
            file_name,
            image_ref,
            width,
            height,
            from_name,
            to_name,
            rows,
        });
    }

    parsed_tasks.sort_by(|left, right| left.file_name.cmp(&right.file_name));

    let mut category_names = BTreeSet::new();
    for task in &parsed_tasks {
        for row in &task.rows {
            category_names.insert(row.label.clone());
        }
    }

    let categories: Vec<Category> = category_names
        .into_iter()
        .enumerate()
        .map(|(idx, name)| Category::new((idx + 1) as u64, name))
        .collect();

    let category_id_by_name: BTreeMap<String, CategoryId> = categories
        .iter()
        .map(|category| (category.name.clone(), category.id))
        .collect();

    let mut images = Vec::with_capacity(parsed_tasks.len());
    let mut image_id_by_name = BTreeMap::new();

    for (idx, task) in parsed_tasks.iter().enumerate() {
        let mut image = Image::new(
            (idx + 1) as u64,
            task.file_name.clone(),
            task.width,
            task.height,
        );
        image
            .attributes
            .insert("ls_image_ref".to_string(), task.image_ref.clone());
        if let Some(from_name) = &task.from_name {
            image
                .attributes
                .insert("ls_from_name".to_string(), from_name.clone());
        }
        if let Some(to_name) = &task.to_name {
            image
                .attributes
                .insert("ls_to_name".to_string(), to_name.clone());
        }

        image_id_by_name.insert(task.file_name.clone(), image.id);
        images.push(image);
    }

    let mut annotations = Vec::new();
    let mut next_annotation_id: u64 = 1;

    for task in parsed_tasks {
        let image_id = image_id_by_name
            .get(&task.file_name)
            .copied()
            .ok_or_else(|| {
                invalid(
                    path,
                    format!(
                        "internal error: missing image mapping for '{}'",
                        task.file_name
                    ),
                )
            })?;

        for parsed in task.rows {
            let category_id = category_id_by_name
                .get(&parsed.label)
                .copied()
                .ok_or_else(|| {
                    invalid(
                        path,
                        format!(
                            "internal error: missing category mapping for '{}'",
                            parsed.label
                        ),
                    )
                })?;

            let mut annotation = Annotation::new(
                AnnotationId::new(next_annotation_id),
                image_id,
                category_id,
                parsed.bbox,
            );
            annotation.confidence = parsed.confidence;
            annotation.attributes = parsed.attributes;
            annotations.push(annotation);
            next_annotation_id += 1;
        }
    }

    Ok(Dataset {
        info: DatasetInfo::default(),
        licenses: vec![],
        images,
        categories,
        annotations,
    })
}

fn select_annotation_results(
    annotations: Option<Vec<LsResultSet>>,
    completions: Option<Vec<LsResultSet>>,
    path: &Path,
    task_idx: usize,
) -> Result<Vec<LsResult>, PanlabelError> {
    if annotations.is_some() && completions.is_some() {
        return Err(invalid(
            path,
            format!("task[{task_idx}] has both annotations and completions; expected only one key"),
        ));
    }

    let selected = annotations.or(completions);
    let Some(mut sets) = selected else {
        return Ok(vec![]);
    };

    if sets.len() > 1 {
        return Err(invalid(
            path,
            format!(
                "task[{task_idx}] has {} annotation sets; panlabel currently requires <= 1",
                sets.len()
            ),
        ));
    }

    Ok(sets.pop().map(|set| set.result).unwrap_or_default())
}

fn select_prediction_results(
    predictions: Option<Vec<LsResultSet>>,
    path: &Path,
    task_idx: usize,
) -> Result<Vec<LsResult>, PanlabelError> {
    let Some(mut sets) = predictions else {
        return Ok(vec![]);
    };

    if sets.len() > 1 {
        return Err(invalid(
            path,
            format!(
                "task[{task_idx}] has {} prediction sets; panlabel currently requires <= 1",
                sets.len()
            ),
        ));
    }

    Ok(sets.pop().map(|set| set.result).unwrap_or_default())
}

fn parse_result(
    result: &LsResult,
    path: &Path,
    task_idx: usize,
    result_idx: usize,
    result_source: &str,
    dims: &mut Option<(u32, u32)>,
    from_names: &mut BTreeSet<String>,
    to_names: &mut BTreeSet<String>,
) -> Result<ParsedAnnotation, PanlabelError> {
    if result.result_type != "rectanglelabels" {
        return Err(invalid(
            path,
            format!(
                "task[{task_idx}] {result_source}[{result_idx}] unsupported result type '{}'; only rectanglelabels is supported",
                result.result_type
            ),
        ));
    }

    let raw_value = result.value.as_ref().ok_or_else(|| {
        invalid(
            path,
            format!("task[{task_idx}] {result_source}[{result_idx}] missing value object"),
        )
    })?;

    let value: LsRectangleValue = serde_json::from_value(raw_value.clone()).map_err(|source| {
        invalid(
            path,
            format!(
                "task[{task_idx}] {result_source}[{result_idx}] has invalid rectanglelabels value: {source}"
            ),
        )
    })?;

    if value.rectanglelabels.len() != 1 {
        return Err(invalid(
            path,
            format!(
                "task[{task_idx}] {result_source}[{result_idx}] rectanglelabels length is {}; expected exactly 1",
                value.rectanglelabels.len()
            ),
        ));
    }

    let original_width = result.original_width.ok_or_else(|| {
        invalid(
            path,
            format!("task[{task_idx}] {result_source}[{result_idx}] missing original_width"),
        )
    })?;

    let original_height = result.original_height.ok_or_else(|| {
        invalid(
            path,
            format!("task[{task_idx}] {result_source}[{result_idx}] missing original_height"),
        )
    })?;

    if let Some((existing_w, existing_h)) = dims {
        if *existing_w != original_width || *existing_h != original_height {
            return Err(invalid(
                path,
                format!(
                    "task[{task_idx}] has inconsistent original dimensions: ({existing_w}, {existing_h}) vs ({original_width}, {original_height})"
                ),
            ));
        }
    } else {
        *dims = Some((original_width, original_height));
    }

    if let Some(from_name) = result.from_name.as_ref().filter(|name| !name.is_empty()) {
        from_names.insert(from_name.clone());
    }
    if let Some(to_name) = result.to_name.as_ref().filter(|name| !name.is_empty()) {
        to_names.insert(to_name.clone());
    }

    let rotation = result.rotation.unwrap_or(0.0);
    let bbox = percent_bbox_to_pixel(
        value.x,
        value.y,
        value.width,
        value.height,
        original_width,
        original_height,
        rotation,
    );

    let mut attributes = BTreeMap::new();
    if rotation != 0.0 {
        attributes.insert("ls_rotation_deg".to_string(), rotation.to_string());
    }

    Ok(ParsedAnnotation {
        label: value.rectanglelabels[0].clone(),
        bbox,
        confidence: result.score,
        attributes,
    })
}

// ============================================================================
// Conversion: IR -> Label Studio
// ============================================================================

fn ir_to_ls(dataset: &Dataset, path: &Path) -> Result<Vec<LsTaskOut>, PanlabelError> {
    let image_by_id: BTreeMap<ImageId, &Image> = dataset
        .images
        .iter()
        .map(|image| (image.id, image))
        .collect();
    let category_name_by_id: BTreeMap<CategoryId, String> = dataset
        .categories
        .iter()
        .map(|category| (category.id, category.name.clone()))
        .collect();

    let mut annotations_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for annotation in &dataset.annotations {
        if !image_by_id.contains_key(&annotation.image_id) {
            return Err(invalid(
                path,
                format!(
                    "annotation {} references missing image {}",
                    annotation.id.as_u64(),
                    annotation.image_id.as_u64()
                ),
            ));
        }

        if !category_name_by_id.contains_key(&annotation.category_id) {
            return Err(invalid(
                path,
                format!(
                    "annotation {} references missing category {}",
                    annotation.id.as_u64(),
                    annotation.category_id.as_u64()
                ),
            ));
        }

        annotations_by_image
            .entry(annotation.image_id)
            .or_default()
            .push(annotation);
    }

    for annotations in annotations_by_image.values_mut() {
        annotations.sort_by_key(|annotation| annotation.id);
    }

    let mut image_ref_by_id: BTreeMap<ImageId, String> = BTreeMap::new();
    let mut seen_basenames = BTreeSet::new();
    for image in &dataset.images {
        let image_ref = image
            .attributes
            .get("ls_image_ref")
            .cloned()
            .unwrap_or_else(|| image.file_name.clone());

        let basename = derive_image_file_name(&image_ref).ok_or_else(|| {
            invalid(
                path,
                format!(
                    "image '{}' maps to invalid Label Studio image reference '{}'",
                    image.file_name, image_ref
                ),
            )
        })?;

        if !seen_basenames.insert(basename.clone()) {
            return Err(invalid(
                path,
                format!(
                    "multiple images map to basename '{}' for Label Studio output; unique basenames are required",
                    basename
                ),
            ));
        }

        image_ref_by_id.insert(image.id, image_ref);
    }

    let mut images_sorted: Vec<&Image> = dataset.images.iter().collect();
    images_sorted.sort_by(|left, right| left.file_name.cmp(&right.file_name));

    let mut tasks = Vec::with_capacity(images_sorted.len());

    for (idx, image) in images_sorted.into_iter().enumerate() {
        let image_ref = image_ref_by_id.get(&image.id).cloned().ok_or_else(|| {
            invalid(
                path,
                format!(
                    "internal error: missing image reference mapping for image {}",
                    image.id.as_u64()
                ),
            )
        })?;

        let from_name = image
            .attributes
            .get("ls_from_name")
            .cloned()
            .unwrap_or_else(|| "label".to_string());

        let to_name = image
            .attributes
            .get("ls_to_name")
            .cloned()
            .unwrap_or_else(|| "image".to_string());

        let image_annotations = annotations_by_image.remove(&image.id).unwrap_or_default();
        let mut annotation_results = Vec::new();
        let mut prediction_results = Vec::new();

        for annotation in image_annotations {
            let category_name = category_name_by_id
                .get(&annotation.category_id)
                .ok_or_else(|| {
                    invalid(
                        path,
                        format!(
                            "internal error: missing category {} while writing",
                            annotation.category_id.as_u64()
                        ),
                    )
                })?
                .clone();

            let rotation = annotation
                .attributes
                .get("ls_rotation_deg")
                .and_then(|value| value.parse::<f64>().ok());

            let (x, y, width, height) =
                pixel_bbox_to_percent(&annotation.bbox, image.width, image.height).ok_or_else(
                    || {
                        invalid(
                            path,
                            format!(
                        "image '{}' has zero width/height; cannot convert bbox {} to percentages",
                        image.file_name,
                        annotation.id.as_u64()
                    ),
                        )
                    },
                )?;

            let result = LsResultOut {
                result_type: "rectanglelabels",
                value: LsRectangleValueOut {
                    x,
                    y,
                    width,
                    height,
                    rectanglelabels: vec![category_name],
                },
                original_width: image.width,
                original_height: image.height,
                from_name: from_name.clone(),
                to_name: to_name.clone(),
                rotation,
                score: annotation.confidence,
            };

            if annotation.confidence.is_some() {
                prediction_results.push(result);
            } else {
                annotation_results.push(result);
            }
        }

        let task = LsTaskOut {
            id: (idx + 1) as u64,
            data: LsTaskDataOut {
                image: image_ref,
                width: image.width,
                height: image.height,
            },
            annotations: if annotation_results.is_empty() {
                vec![]
            } else {
                vec![LsResultSetOut {
                    result: annotation_results,
                }]
            },
            predictions: if prediction_results.is_empty() {
                vec![]
            } else {
                vec![LsResultSetOut {
                    result: prediction_results,
                }]
            },
        };

        tasks.push(task);
    }

    Ok(tasks)
}

// ============================================================================
// Helpers
// ============================================================================

fn invalid(path: &Path, message: impl Into<String>) -> PanlabelError {
    PanlabelError::LabelStudioJsonInvalid {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn derive_image_file_name(image_ref: &str) -> Option<String> {
    let no_query = image_ref.split('?').next().unwrap_or(image_ref);
    let no_fragment = no_query.split('#').next().unwrap_or(no_query);
    let normalized = no_fragment.replace('\\', "/");
    let candidate = normalized.rsplit('/').next()?;
    if candidate.is_empty() {
        return None;
    }
    Some(candidate.to_string())
}

fn percent_bbox_to_pixel(
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    image_width: u32,
    image_height: u32,
    rotation_deg: f64,
) -> BBoxXYXY<Pixel> {
    let w = image_width as f64;
    let h = image_height as f64;

    let xmin = (x / 100.0) * w;
    let ymin = (y / 100.0) * h;
    let xmax = ((x + width) / 100.0) * w;
    let ymax = ((y + height) / 100.0) * h;

    if rotation_deg == 0.0 {
        return BBoxXYXY::from_xyxy(xmin, ymin, xmax, ymax);
    }

    rotated_envelope_bbox(xmin, ymin, xmax, ymax, rotation_deg)
}

fn rotated_envelope_bbox(
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    rotation_deg: f64,
) -> BBoxXYXY<Pixel> {
    let theta = rotation_deg * (PI / 180.0);
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    let cx = (xmin + xmax) / 2.0;
    let cy = (ymin + ymax) / 2.0;

    let corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)];

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for (x, y) in corners {
        let dx = x - cx;
        let dy = y - cy;
        let rx = cx + (dx * cos_t) - (dy * sin_t);
        let ry = cy + (dx * sin_t) + (dy * cos_t);

        min_x = min_x.min(rx);
        min_y = min_y.min(ry);
        max_x = max_x.max(rx);
        max_y = max_y.max(ry);
    }

    BBoxXYXY::from_xyxy(min_x, min_y, max_x, max_y)
}

fn pixel_bbox_to_percent(
    bbox: &BBoxXYXY<Pixel>,
    image_width: u32,
    image_height: u32,
) -> Option<(f64, f64, f64, f64)> {
    if image_width == 0 || image_height == 0 {
        return None;
    }

    let w = image_width as f64;
    let h = image_height as f64;

    let x = (bbox.xmin() / w) * 100.0;
    let y = (bbox.ymin() / h) * 100.0;
    let width = ((bbox.xmax() - bbox.xmin()) / w) * 100.0;
    let height = ((bbox.ymax() - bbox.ymin()) / h) * 100.0;

    Some((x, y, width, height))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"[
  {
    "data": {"image": "https://example.com/assets/img_b.jpg"},
    "annotations": [
      {
        "result": [
          {
            "type": "rectanglelabels",
            "from_name": "bbox",
            "to_name": "image",
            "value": {
              "x": 10.0,
              "y": 10.0,
              "width": 40.0,
              "height": 50.0,
              "rectanglelabels": ["dog"]
            },
            "original_width": 200,
            "original_height": 100
          }
        ]
      }
    ],
    "predictions": [
      {
        "result": [
          {
            "type": "rectanglelabels",
            "from_name": "bbox",
            "to_name": "image",
            "value": {
              "x": 50.0,
              "y": 20.0,
              "width": 10.0,
              "height": 20.0,
              "rectanglelabels": ["cat"]
            },
            "original_width": 200,
            "original_height": 100,
            "score": 0.9
          }
        ]
      }
    ]
  },
  {
    "data": {"image": "img_a.jpg"},
    "annotations": [
      {
        "result": [
          {
            "type": "rectanglelabels",
            "from_name": "bbox",
            "to_name": "image",
            "value": {
              "x": 0.0,
              "y": 0.0,
              "width": 20.0,
              "height": 50.0,
              "rectanglelabels": ["cat"]
            },
            "original_width": 100,
            "original_height": 100
          }
        ]
      }
    ]
  }
]"#
    }

    #[test]
    fn parse_assigns_deterministic_ids() {
        let dataset = from_label_studio_str(sample_json()).expect("parse dataset");

        assert_eq!(dataset.images.len(), 2);
        assert_eq!(dataset.categories.len(), 2);
        assert_eq!(dataset.annotations.len(), 3);

        // Images sorted by derived basename.
        assert_eq!(dataset.images[0].file_name, "img_a.jpg");
        assert_eq!(dataset.images[0].id.as_u64(), 1);
        assert_eq!(dataset.images[1].file_name, "img_b.jpg");
        assert_eq!(dataset.images[1].id.as_u64(), 2);

        // Categories sorted lexicographically.
        assert_eq!(dataset.categories[0].name, "cat");
        assert_eq!(dataset.categories[0].id.as_u64(), 1);
        assert_eq!(dataset.categories[1].name, "dog");
        assert_eq!(dataset.categories[1].id.as_u64(), 2);

        // prediction score -> confidence
        assert!(dataset
            .annotations
            .iter()
            .any(|ann| ann.confidence == Some(0.9)));

        assert_eq!(
            dataset.images[1].attributes.get("ls_image_ref"),
            Some(&"https://example.com/assets/img_b.jpg".to_string())
        );
    }

    #[test]
    fn parse_rotation_sets_attribute() {
        let json = r#"[
  {
    "data": {"image": "img_rot.jpg"},
    "annotations": [
      {
        "result": [
          {
            "type": "rectanglelabels",
            "value": {
              "x": 10.0,
              "y": 20.0,
              "width": 30.0,
              "height": 40.0,
              "rectanglelabels": ["box"]
            },
            "rotation": 35.0,
            "original_width": 100,
            "original_height": 200
          }
        ]
      }
    ]
  }
]"#;

        let dataset = from_label_studio_str(json).expect("parse rotated");
        let ann = &dataset.annotations[0];
        assert_eq!(
            ann.attributes.get("ls_rotation_deg"),
            Some(&"35".to_string())
        );
    }

    #[test]
    fn parse_rejects_multiple_annotation_sets() {
        let json = r#"[
  {
    "data": {"image": "img.jpg"},
    "annotations": [{"result": []}, {"result": []}]
  }
]"#;

        let err = from_label_studio_str(json).expect_err("expected invalid error");
        match err {
            PanlabelError::LabelStudioJsonInvalid { message, .. } => {
                assert!(message.contains("annotation sets"));
            }
            other => panic!("expected LabelStudioJsonInvalid, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_unsupported_result_type() {
        let json = r#"[
  {
    "data": {"image": "img.jpg"},
    "annotations": [
      {
        "result": [
          {
            "type": "choices",
            "value": {"choices": ["yes"]},
            "original_width": 10,
            "original_height": 10
          }
        ]
      }
    ]
  }
]"#;

        let err = from_label_studio_str(json).expect_err("expected invalid error");
        match err {
            PanlabelError::LabelStudioJsonInvalid { message, .. } => {
                assert!(message.contains("unsupported result type"));
            }
            other => panic!("expected LabelStudioJsonInvalid, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_duplicate_basenames() {
        let json = r#"[
  {
    "data": {"image": "https://a.example/x/img.jpg", "width": 10, "height": 10},
    "annotations": [{"result": []}],
    "predictions": []
  },
  {
    "data": {"image": "https://b.example/y/img.jpg", "width": 10, "height": 10},
    "annotations": [{"result": []}],
    "predictions": []
  }
]"#;

        let err = from_label_studio_str(json).expect_err("expected invalid error");
        match err {
            PanlabelError::LabelStudioJsonInvalid { message, .. } => {
                assert!(message.contains("duplicate image basename"));
            }
            other => panic!("expected LabelStudioJsonInvalid, got {other:?}"),
        }
    }

    #[test]
    fn write_then_read_roundtrip_semantic() {
        let dataset = from_label_studio_str(sample_json()).expect("parse original");
        let json = to_label_studio_string(&dataset).expect("write string");
        let restored = from_label_studio_str(&json).expect("parse restored");

        assert_eq!(dataset.images.len(), restored.images.len());
        assert_eq!(dataset.categories.len(), restored.categories.len());
        assert_eq!(dataset.annotations.len(), restored.annotations.len());

        for (left, right) in dataset.annotations.iter().zip(restored.annotations.iter()) {
            assert!((left.bbox.xmin() - right.bbox.xmin()).abs() < 1e-6);
            assert!((left.bbox.ymin() - right.bbox.ymin()).abs() < 1e-6);
            assert!((left.bbox.xmax() - right.bbox.xmax()).abs() < 1e-6);
            assert!((left.bbox.ymax() - right.bbox.ymax()).abs() < 1e-6);
        }
    }

    #[test]
    fn parse_rejects_inconsistent_from_name() {
        let json = r#"[
  {
    "data": {"image": "img.jpg"},
    "annotations": [
      {
        "result": [
          {
            "type": "rectanglelabels",
            "from_name": "bbox_a",
            "to_name": "image",
            "value": {
              "x": 10.0,
              "y": 10.0,
              "width": 10.0,
              "height": 10.0,
              "rectanglelabels": ["cat"]
            },
            "original_width": 100,
            "original_height": 100
          },
          {
            "type": "rectanglelabels",
            "from_name": "bbox_b",
            "to_name": "image",
            "value": {
              "x": 20.0,
              "y": 20.0,
              "width": 10.0,
              "height": 10.0,
              "rectanglelabels": ["cat"]
            },
            "original_width": 100,
            "original_height": 100
          }
        ]
      }
    ]
  }
]"#;

        let err = from_label_studio_str(json).expect_err("expected inconsistent from_name error");
        match err {
            PanlabelError::LabelStudioJsonInvalid { message, .. } => {
                assert!(message.contains("inconsistent from_name"));
            }
            other => panic!("expected LabelStudioJsonInvalid, got {other:?}"),
        }
    }

    #[test]
    fn writer_rejects_duplicate_output_basenames() {
        let dataset = Dataset {
            images: vec![
                Image::new(1u64, "train/shared.jpg", 100, 100),
                Image::new(2u64, "val/shared.jpg", 100, 100),
            ],
            categories: vec![Category::new(1u64, "cat")],
            annotations: vec![],
            ..Default::default()
        };

        let err = to_label_studio_string(&dataset).expect_err("expected duplicate basename error");
        match err {
            PanlabelError::LabelStudioJsonInvalid { message, .. } => {
                assert!(message.contains("unique basenames are required"));
            }
            other => panic!("expected LabelStudioJsonInvalid, got {other:?}"),
        }
    }
}
