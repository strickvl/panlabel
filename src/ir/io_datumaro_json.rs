//! Datumaro JSON bbox-only adapter.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use serde_json::{json, Value};

use super::io_bbox_adapters_common::{dataset_from_raw, f64_field, RawAnn, RawImage};
use super::model::{Dataset, DatasetInfo};
use super::{BBoxXYXY, CategoryId};
use crate::error::PanlabelError;

pub fn read_datumaro_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let value: Value =
        serde_json::from_reader(file).map_err(|source| PanlabelError::DatumaroJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    datumaro_value_to_ir(&value, path)
}

pub fn write_datumaro_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let value = to_datumaro_value(dataset);
    let file = File::create(path).map_err(PanlabelError::Io)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &value).map_err(|source| {
        PanlabelError::DatumaroJsonWrite {
            path: path.to_path_buf(),
            source,
        }
    })
}

pub(crate) fn is_likely_datumaro_file(value: &Value) -> bool {
    value.get("items").is_some_and(Value::is_array)
        && value.get("categories").is_some_and(Value::is_object)
}

fn datumaro_value_to_ir(value: &Value, path: &Path) -> Result<Dataset, PanlabelError> {
    let items = value
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| PanlabelError::DatumaroJsonInvalid {
            path: path.to_path_buf(),
            message: "expected top-level items array".to_string(),
        })?;

    let mut categories_hint = Vec::new();
    if let Some(labels) = value
        .pointer("/categories/label/labels")
        .and_then(Value::as_array)
    {
        for label in labels {
            if let Some(name) = label.get("name").and_then(Value::as_str) {
                let parent = label
                    .get("parent")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string);
                categories_hint.push((name.to_string(), parent));
            }
        }
    }

    let mut images = Vec::new();
    let mut anns = Vec::new();
    let mut skipped = 0usize;
    for item in items {
        let image_name = item
            .pointer("/image/path")
            .and_then(Value::as_str)
            .or_else(|| item.get("id").and_then(Value::as_str))
            .unwrap_or("datumaro-item.jpg")
            .to_string();
        let (width, height) = datumaro_size(item).unwrap_or((1, 1));
        images.push(RawImage {
            file_name: image_name.clone(),
            width,
            height,
            attributes: BTreeMap::new(),
        });

        for ann in item
            .get("annotations")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if ann.get("type").and_then(Value::as_str) != Some("bbox") {
                skipped += 1;
                continue;
            }
            let Some(b) = ann.get("bbox").and_then(Value::as_array) else {
                skipped += 1;
                continue;
            };
            if b.len() < 4 || !b.iter().take(4).all(Value::is_number) {
                skipped += 1;
                continue;
            }
            let label_id = ann.get("label_id").and_then(Value::as_u64).unwrap_or(0) as usize;
            let category = categories_hint
                .get(label_id)
                .map(|(n, _)| n.clone())
                .unwrap_or_else(|| format!("label_{label_id}"));
            anns.push(RawAnn {
                image: image_name.clone(),
                category,
                bbox: BBoxXYXY::from_xywh(
                    b[0].as_f64().unwrap(),
                    b[1].as_f64().unwrap(),
                    b[2].as_f64().unwrap(),
                    b[3].as_f64().unwrap(),
                ),
                confidence: f64_field(ann, "score"),
                attributes: BTreeMap::new(),
            });
        }
    }

    let mut info = DatasetInfo::default();
    if skipped > 0 {
        info.attributes.insert(
            "datumaro_unsupported_annotations_skipped".into(),
            skipped.to_string(),
        );
    }
    Ok(dataset_from_raw(images, anns, categories_hint, info))
}

fn datumaro_size(item: &Value) -> Option<(u32, u32)> {
    let size = item.pointer("/image/size")?;
    if let Some(obj) = size.as_object() {
        return Some((
            obj.get("width")?.as_u64()? as u32,
            obj.get("height")?.as_u64()? as u32,
        ));
    }
    let arr = size.as_array()?;
    if arr.len() >= 2 {
        Some((arr[1].as_u64()? as u32, arr[0].as_u64()? as u32))
    } else {
        None
    }
}

fn to_datumaro_value(dataset: &Dataset) -> Value {
    let mut cats: Vec<_> = dataset.categories.iter().collect();
    cats.sort_by_key(|c| c.id);
    let cat_index: BTreeMap<CategoryId, usize> =
        cats.iter().enumerate().map(|(i, c)| (c.id, i)).collect();
    let label_values: Vec<Value> = cats
        .iter()
        .map(|cat| {
            let mut obj = serde_json::Map::new();
            obj.insert("name".into(), json!(cat.name));
            if let Some(parent) = &cat.supercategory {
                obj.insert("parent".into(), json!(parent));
            }
            Value::Object(obj)
        })
        .collect();

    let anns_by_image = super::io_bbox_adapters_common::annotations_by_image(dataset);
    let mut images: Vec<_> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    let items: Vec<Value> = images.into_iter().map(|img| {
        let anns: Vec<Value> = anns_by_image.get(&img.id).into_iter().flat_map(|v| v.iter()).map(|ann| {
            let (x, y, w, h) = ann.bbox.to_xywh();
            let mut obj = serde_json::Map::new();
            obj.insert("id".into(), json!(ann.id.as_u64()));
            obj.insert("type".into(), json!("bbox"));
            obj.insert("bbox".into(), json!([x, y, w, h]));
            obj.insert("label_id".into(), json!(*cat_index.get(&ann.category_id).unwrap_or(&0)));
            if let Some(score) = ann.confidence { obj.insert("score".into(), json!(score)); }
            Value::Object(obj)
        }).collect();
        json!({"id": img.file_name, "image": {"path": img.file_name, "size": [img.height, img.width]}, "annotations": anns})
    }).collect();

    json!({"categories": {"label": {"labels": label_values}}, "items": items})
}
