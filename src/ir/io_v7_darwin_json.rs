//! V7 Darwin JSON bbox-only adapter.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use serde_json::{json, Value};

use super::io_bbox_adapters_common::{
    dataset_from_raw, f64_field, image_dimensions_if_found, string_field, u32_field, RawAnn,
    RawImage,
};
use super::model::{Dataset, DatasetInfo};
use super::BBoxXYXY;
use crate::error::PanlabelError;

pub fn read_v7_darwin_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let value: Value =
        serde_json::from_reader(file).map_err(|source| PanlabelError::V7DarwinJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    v7_value_to_ir(&value, path)
}

pub fn write_v7_darwin_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let file = File::create(path).map_err(PanlabelError::Io)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &to_v7_value(dataset)).map_err(|source| {
        PanlabelError::V7DarwinJsonWrite {
            path: path.to_path_buf(),
            source,
        }
    })
}

pub(crate) fn is_likely_v7_darwin_file(value: &Value) -> bool {
    fn item_like(v: &Value) -> bool {
        v.get("annotations").is_some_and(Value::is_array)
            && (v.get("item").is_some() || v.get("slots").is_some() || v.get("name").is_some())
    }
    value
        .as_array()
        .is_some_and(|a| a.first().is_some_and(item_like))
        || item_like(value)
}

fn v7_value_to_ir(value: &Value, path: &Path) -> Result<Dataset, PanlabelError> {
    let owned_items: Vec<Value> = if let Some(arr) = value.as_array() {
        arr.clone()
    } else {
        vec![value.clone()]
    };
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut images = Vec::new();
    let mut anns = Vec::new();
    let mut skipped = 0usize;
    for (idx, root) in owned_items.iter().enumerate() {
        let item = root.get("item").unwrap_or(root);
        let image_name = string_field(item, "name")
            .or_else(|| {
                root.pointer("/slots/0/source_files/0/file_name")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            })
            .or_else(|| {
                root.pointer("/slots/0/file_name")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            })
            .unwrap_or_else(|| format!("darwin-item-{idx}.jpg"));
        let mut max_x: f64 = 1.0;
        let mut max_y: f64 = 1.0;
        for ann in root
            .get("annotations")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(bb) = ann.get("bounding_box") else {
                skipped += 1;
                continue;
            };
            let x = f64_field(bb, "x").unwrap_or(0.0);
            let y = f64_field(bb, "y").unwrap_or(0.0);
            let w = f64_field(bb, "w").or_else(|| f64_field(bb, "width"));
            let h = f64_field(bb, "h").or_else(|| f64_field(bb, "height"));
            let (Some(w), Some(h)) = (w, h) else {
                skipped += 1;
                continue;
            };
            max_x = max_x.max(x + w);
            max_y = max_y.max(y + h);
            let mut attrs = BTreeMap::new();
            if let Some(id) = string_field(ann, "id") {
                attrs.insert("darwin_annotation_id".into(), id);
            }
            anns.push(RawAnn {
                image: image_name.clone(),
                category: string_field(ann, "name").unwrap_or_else(|| "object".into()),
                bbox: BBoxXYXY::from_xywh(x, y, w, h),
                confidence: None,
                attributes: attrs,
            });
        }
        let dims = root
            .get("slots")
            .and_then(Value::as_array)
            .and_then(|slots| slots.first())
            .and_then(|slot| Some((u32_field(slot, "width")?, u32_field(slot, "height")?)))
            .or_else(|| image_dimensions_if_found(base, &image_name))
            .unwrap_or((max_x.ceil() as u32, max_y.ceil() as u32));
        images.push(RawImage {
            file_name: image_name,
            width: dims.0.max(1),
            height: dims.1.max(1),
            attributes: BTreeMap::new(),
        });
    }
    let mut info = DatasetInfo::default();
    if skipped > 0 {
        info.attributes.insert(
            "darwin_unsupported_annotations_skipped".into(),
            skipped.to_string(),
        );
    }
    Ok(dataset_from_raw(images, anns, vec![], info))
}

fn to_v7_value(dataset: &Dataset) -> Value {
    let cat_lookup: BTreeMap<_, _> = dataset.categories.iter().map(|c| (c.id, c)).collect();
    let anns_by_image = super::io_bbox_adapters_common::annotations_by_image(dataset);
    let mut images: Vec<_> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    Value::Array(images.into_iter().map(|img| {
        let annotations: Vec<Value> = anns_by_image.get(&img.id).into_iter().flat_map(|v| v.iter()).map(|ann| {
            let (x, y, w, h) = ann.bbox.to_xywh();
            json!({"id": ann.id.as_u64().to_string(), "name": cat_lookup.get(&ann.category_id).map(|c| c.name.as_str()).unwrap_or("object"), "bounding_box": {"x": x, "y": y, "w": w, "h": h}})
        }).collect();
        json!({"version": "2.0", "item": {"name": img.file_name}, "slots": [{"name": "0", "width": img.width, "height": img.height, "source_files": [{"file_name": img.file_name}]}], "annotations": annotations})
    }).collect())
}
