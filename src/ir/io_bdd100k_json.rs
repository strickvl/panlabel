//! BDD100K / Scalabel JSON bbox-only adapter.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use serde_json::{json, Value};

use super::io_bbox_adapters_common::{
    dataset_from_raw, f64_field, image_dimensions_if_found, scalar_to_string, string_field,
    u32_field, RawAnn, RawImage,
};
use super::model::{Dataset, DatasetInfo};
use super::BBoxXYXY;
use crate::error::PanlabelError;

pub fn read_bdd100k_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let value: Value =
        serde_json::from_reader(file).map_err(|source| PanlabelError::Bdd100kJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    bdd100k_value_to_ir(&value, path)
}

pub fn write_bdd100k_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let file = File::create(path).map_err(PanlabelError::Io)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &to_bdd100k_value(dataset)).map_err(
        |source| PanlabelError::Bdd100kJsonWrite {
            path: path.to_path_buf(),
            source,
        },
    )
}

pub(crate) fn is_likely_bdd100k_file(value: &Value) -> bool {
    fn frame_like(v: &Value) -> bool {
        v.get("labels").is_some_and(Value::is_array)
            && (v.get("name").is_some() || v.get("url").is_some() || v.get("videoName").is_some())
    }
    value
        .as_array()
        .is_some_and(|a| a.first().is_some_and(frame_like))
        || value
            .get("frames")
            .and_then(Value::as_array)
            .is_some_and(|a| a.first().is_some_and(frame_like))
}

fn bdd100k_value_to_ir(value: &Value, path: &Path) -> Result<Dataset, PanlabelError> {
    let empty = Vec::new();
    let frames: Vec<&Value> = if let Some(arr) = value.as_array() {
        arr.iter().collect()
    } else if let Some(arr) = value.get("frames").and_then(Value::as_array) {
        arr.iter().collect()
    } else {
        empty.iter().collect()
    };
    if frames.is_empty() {
        return Err(PanlabelError::Bdd100kJsonInvalid {
            path: path.to_path_buf(),
            message: "expected array root or object.frames array".into(),
        });
    }

    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut images = Vec::new();
    let mut anns = Vec::new();
    let mut skipped = 0usize;
    for (idx, frame) in frames.iter().enumerate() {
        let image_name = string_field(frame, "name")
            .or_else(|| string_field(frame, "url"))
            .unwrap_or_else(|| format!("bdd100k-frame-{idx}.jpg"));
        let mut max_x: f64 = 1.0;
        let mut max_y: f64 = 1.0;
        let mut frame_attrs = BTreeMap::new();
        if let Some(attrs) = frame.get("attributes").and_then(Value::as_object) {
            for (k, v) in attrs {
                if let Some(s) = scalar_to_string(v) {
                    frame_attrs.insert(format!("bdd100k_attr_{k}"), s);
                }
            }
        }
        for label in frame
            .get("labels")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(box2d) = label.get("box2d") else {
                skipped += 1;
                continue;
            };
            let Some((x1, y1, x2, y2)) = read_box2d(box2d) else {
                skipped += 1;
                continue;
            };
            max_x = max_x.max(x2);
            max_y = max_y.max(y2);
            let mut attrs = BTreeMap::new();
            if let Some(label_attrs) = label.get("attributes").and_then(Value::as_object) {
                for (k, v) in label_attrs {
                    if let Some(s) = scalar_to_string(v) {
                        attrs.insert(format!("bdd100k_label_attr_{k}"), s);
                    }
                }
            }
            anns.push(RawAnn {
                image: image_name.clone(),
                category: string_field(label, "category").unwrap_or_else(|| "object".into()),
                bbox: BBoxXYXY::from_xyxy(x1, y1, x2, y2),
                confidence: f64_field(label, "score"),
                attributes: attrs,
            });
        }
        let dims = explicit_dims(frame)
            .or_else(|| image_dimensions_if_found(base, &image_name))
            .unwrap_or((max_x.ceil() as u32, max_y.ceil() as u32));
        images.push(RawImage {
            file_name: image_name,
            width: dims.0.max(1),
            height: dims.1.max(1),
            attributes: frame_attrs,
        });
    }
    let mut info = DatasetInfo::default();
    if skipped > 0 {
        info.attributes.insert(
            "bdd100k_unsupported_labels_skipped".into(),
            skipped.to_string(),
        );
    }
    Ok(dataset_from_raw(images, anns, vec![], info))
}

fn read_box2d(v: &Value) -> Option<(f64, f64, f64, f64)> {
    Some((
        f64_field(v, "x1")?,
        f64_field(v, "y1")?,
        f64_field(v, "x2")?,
        f64_field(v, "y2")?,
    ))
}
fn explicit_dims(v: &Value) -> Option<(u32, u32)> {
    Some((u32_field(v, "width")?, u32_field(v, "height")?))
}

fn to_bdd100k_value(dataset: &Dataset) -> Value {
    let cat_lookup: BTreeMap<_, _> = dataset.categories.iter().map(|c| (c.id, c)).collect();
    let anns_by_image = super::io_bbox_adapters_common::annotations_by_image(dataset);
    let mut images: Vec<_> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    Value::Array(images.into_iter().map(|img| {
        let labels: Vec<Value> = anns_by_image.get(&img.id).into_iter().flat_map(|v| v.iter()).map(|ann| {
            let mut obj = serde_json::Map::new();
            obj.insert("category".into(), json!(cat_lookup.get(&ann.category_id).map(|c| c.name.as_str()).unwrap_or("object")));
            obj.insert("box2d".into(), json!({"x1": ann.bbox.xmin(), "y1": ann.bbox.ymin(), "x2": ann.bbox.xmax(), "y2": ann.bbox.ymax()}));
            if let Some(score) = ann.confidence { obj.insert("score".into(), json!(score)); }
            Value::Object(obj)
        }).collect();
        json!({"name": img.file_name, "width": img.width, "height": img.height, "labels": labels})
    }).collect())
}
