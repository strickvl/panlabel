//! Edge Impulse bounding_boxes.labels bbox-only adapter.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::io_bbox_adapters_common::{
    dataset_from_raw, f64_field, image_dimensions_if_found, string_field, RawAnn, RawImage,
};
use super::model::{Dataset, DatasetInfo};
use super::BBoxXYXY;
use crate::error::PanlabelError;

pub fn read_edge_impulse_labels(path: &Path) -> Result<Dataset, PanlabelError> {
    let label_path = labels_path(path);
    let file = File::open(&label_path).map_err(PanlabelError::Io)?;
    let value: Value =
        serde_json::from_reader(file).map_err(|source| PanlabelError::EdgeImpulseJsonParse {
            path: label_path.clone(),
            source,
        })?;
    edge_value_to_ir(&value, &label_path, true)
}

#[cfg(feature = "fuzzing")]
pub fn from_edge_impulse_labels_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    let path = Path::new("<fuzz>");
    let value: Value =
        serde_json::from_slice(bytes).map_err(|source| PanlabelError::EdgeImpulseJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    edge_value_to_ir(&value, path, false)
}

pub fn write_edge_impulse_labels(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let label_path = if path.extension().is_some() {
        path.to_path_buf()
    } else {
        fs::create_dir_all(path).map_err(PanlabelError::Io)?;
        path.join("bounding_boxes.labels")
    };
    let file = File::create(&label_path).map_err(PanlabelError::Io)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &to_edge_value(dataset)).map_err(|source| {
        PanlabelError::EdgeImpulseJsonWrite {
            path: label_path,
            source,
        }
    })
}

pub(crate) fn is_likely_edge_impulse_labels(value: &Value) -> bool {
    value.get("files").is_some_and(Value::is_array)
        || value.get("boundingBoxes").is_some_and(Value::is_object)
        || value.get("type").and_then(Value::as_str) == Some("bounding-box-labels")
}

fn labels_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("bounding_boxes.labels")
    } else {
        path.to_path_buf()
    }
}

fn edge_value_to_ir(
    value: &Value,
    path: &Path,
    probe_image_dimensions: bool,
) -> Result<Dataset, PanlabelError> {
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut images = Vec::new();
    let mut anns = Vec::new();
    if let Some(files) = value.get("files").and_then(Value::as_array) {
        for file in files {
            let image = string_field(file, "path").unwrap_or_else(|| "image.jpg".into());
            let boxes = file
                .get("boundingBoxes")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let dims = if probe_image_dimensions {
                image_dimensions_if_found(base, &image)
            } else {
                None
            }
            .unwrap_or_else(|| infer_dims(&boxes));
            images.push(RawImage {
                file_name: image.clone(),
                width: dims.0,
                height: dims.1,
                attributes: BTreeMap::new(),
            });
            for bb in boxes {
                add_box(&mut anns, &image, &bb);
            }
        }
    } else if let Some(map) = value.get("boundingBoxes").and_then(Value::as_object) {
        for (image, boxes_v) in map {
            let boxes = boxes_v.as_array().cloned().unwrap_or_default();
            let dims = if probe_image_dimensions {
                image_dimensions_if_found(base, image)
            } else {
                None
            }
            .unwrap_or_else(|| infer_dims(&boxes));
            images.push(RawImage {
                file_name: image.clone(),
                width: dims.0,
                height: dims.1,
                attributes: BTreeMap::new(),
            });
            for bb in boxes {
                add_box(&mut anns, image, &bb);
            }
        }
    } else {
        return Err(PanlabelError::EdgeImpulseJsonInvalid {
            path: path.to_path_buf(),
            message: "expected files array or boundingBoxes object".into(),
        });
    }
    Ok(dataset_from_raw(
        images,
        anns,
        vec![],
        DatasetInfo::default(),
    ))
}

fn add_box(anns: &mut Vec<RawAnn>, image: &str, bb: &Value) {
    if let (Some(x), Some(y), Some(w), Some(h)) = (
        f64_field(bb, "x"),
        f64_field(bb, "y"),
        f64_field(bb, "width"),
        f64_field(bb, "height"),
    ) {
        anns.push(RawAnn {
            image: image.to_string(),
            category: string_field(bb, "label").unwrap_or_else(|| "object".into()),
            bbox: BBoxXYXY::from_xywh(x, y, w, h),
            confidence: None,
            attributes: BTreeMap::new(),
        });
    }
}

fn infer_dims(boxes: &[Value]) -> (u32, u32) {
    let mut w: f64 = 1.0;
    let mut h: f64 = 1.0;
    for bb in boxes {
        if let (Some(x), Some(y), Some(bw), Some(bh)) = (
            f64_field(bb, "x"),
            f64_field(bb, "y"),
            f64_field(bb, "width"),
            f64_field(bb, "height"),
        ) {
            w = w.max(x + bw);
            h = h.max(y + bh);
        }
    }
    (w.ceil() as u32, h.ceil() as u32)
}

fn to_edge_value(dataset: &Dataset) -> Value {
    let cat_lookup: BTreeMap<_, _> = dataset.categories.iter().map(|c| (c.id, c)).collect();
    let anns_by_image = super::io_bbox_adapters_common::annotations_by_image(dataset);
    let mut images: Vec<_> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    let files: Vec<Value> = images.into_iter().map(|img| {
        let bbs: Vec<Value> = anns_by_image.get(&img.id).into_iter().flat_map(|v| v.iter()).map(|ann| { let (x,y,w,h)=ann.bbox.to_xywh(); json!({"label": cat_lookup.get(&ann.category_id).map(|c| c.name.as_str()).unwrap_or("object"), "x": x, "y": y, "width": w, "height": h}) }).collect();
        json!({"path": img.file_name, "category": "training", "boundingBoxes": bbs})
    }).collect();
    json!({"version": 1, "type": "bounding-box-labels", "files": files})
}
