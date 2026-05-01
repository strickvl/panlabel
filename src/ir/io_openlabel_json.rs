//! ASAM OpenLABEL JSON 2D bbox subset adapter.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use serde_json::{json, Value};

use super::io_bbox_adapters_common::{
    dataset_from_raw, image_dimensions_if_found, string_field, u32_field, RawAnn, RawImage,
};
use super::model::{Dataset, DatasetInfo};
use super::BBoxXYXY;
use crate::error::PanlabelError;

pub fn read_openlabel_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let value: Value =
        serde_json::from_reader(file).map_err(|source| PanlabelError::OpenLabelJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    openlabel_value_to_ir(&value, path)
}

pub fn write_openlabel_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let file = File::create(path).map_err(PanlabelError::Io)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &to_openlabel_value(dataset)).map_err(
        |source| PanlabelError::OpenLabelJsonWrite {
            path: path.to_path_buf(),
            source,
        },
    )
}

pub(crate) fn is_likely_openlabel_file(value: &Value) -> bool {
    value
        .get("openlabel")
        .and_then(|v| v.get("frames"))
        .is_some_and(Value::is_object)
}

fn openlabel_value_to_ir(value: &Value, path: &Path) -> Result<Dataset, PanlabelError> {
    let openlabel = value
        .get("openlabel")
        .ok_or_else(|| PanlabelError::OpenLabelJsonInvalid {
            path: path.to_path_buf(),
            message: "missing openlabel object".into(),
        })?;
    let frames = openlabel
        .get("frames")
        .and_then(Value::as_object)
        .ok_or_else(|| PanlabelError::OpenLabelJsonInvalid {
            path: path.to_path_buf(),
            message: "missing openlabel.frames object".into(),
        })?;
    let object_types = openlabel
        .get("objects")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut images = Vec::new();
    let mut anns = Vec::new();
    let mut skipped = 0usize;
    for (frame_key, frame) in frames {
        let props = frame
            .get("frame_properties")
            .or_else(|| frame.get("properties"))
            .unwrap_or(&Value::Null);
        let image_name = string_field(props, "file_name")
            .or_else(|| string_field(props, "name"))
            .unwrap_or_else(|| format!("frame_{frame_key}.jpg"));
        let mut max_x: f64 = 1.0;
        let mut max_y: f64 = 1.0;
        for (object_id, object) in frame
            .get("objects")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            let bbox_arr = object
                .pointer("/object_data/bbox")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if bbox_arr.is_empty() {
                skipped += 1;
                continue;
            }
            for bbox in bbox_arr {
                let Some(val) = bbox.get("val").and_then(Value::as_array) else {
                    skipped += 1;
                    continue;
                };
                if val.len() < 4 {
                    skipped += 1;
                    continue;
                }
                if val.len() >= 5
                    && val[4]
                        .as_f64()
                        .map(|alpha| alpha.abs() > f64::EPSILON)
                        .unwrap_or(false)
                {
                    skipped += 1;
                    continue;
                }
                let (cx, cy, w, h) = (
                    val[0].as_f64().unwrap_or(0.0),
                    val[1].as_f64().unwrap_or(0.0),
                    val[2].as_f64().unwrap_or(0.0),
                    val[3].as_f64().unwrap_or(0.0),
                );
                let box_xy = BBoxXYXY::from_cxcywh(cx, cy, w, h);
                max_x = max_x.max(box_xy.xmax());
                max_y = max_y.max(box_xy.ymax());
                let category = object_types
                    .get(object_id)
                    .and_then(|o| string_field(o, "type"))
                    .or_else(|| string_field(&bbox, "name"))
                    .unwrap_or_else(|| "object".into());
                let mut attrs = BTreeMap::new();
                attrs.insert("openlabel_object_id".into(), object_id.clone());
                anns.push(RawAnn {
                    image: image_name.clone(),
                    category,
                    bbox: box_xy,
                    confidence: None,
                    attributes: attrs,
                });
            }
        }
        let dims = match (u32_field(props, "width"), u32_field(props, "height")) {
            (Some(width), Some(height)) => (width, height),
            _ => image_dimensions_if_found(base, &image_name)
                .unwrap_or((max_x.ceil() as u32, max_y.ceil() as u32)),
        };
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
            "openlabel_unsupported_data_skipped".into(),
            skipped.to_string(),
        );
    }
    Ok(dataset_from_raw(images, anns, vec![], info))
}

fn to_openlabel_value(dataset: &Dataset) -> Value {
    let cat_lookup: BTreeMap<_, _> = dataset.categories.iter().map(|c| (c.id, c)).collect();
    let anns_by_image = super::io_bbox_adapters_common::annotations_by_image(dataset);
    let mut images: Vec<_> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    let mut frames = serde_json::Map::new();
    let mut top_objects = serde_json::Map::new();
    for (idx, img) in images.into_iter().enumerate() {
        let mut objects = serde_json::Map::new();
        for ann in anns_by_image
            .get(&img.id)
            .into_iter()
            .flat_map(|v| v.iter())
        {
            let obj_id = ann.id.as_u64().to_string();
            let cat = cat_lookup
                .get(&ann.category_id)
                .map(|c| c.name.as_str())
                .unwrap_or("object");
            let (cx, cy, w, h) = ann.bbox.to_cxcywh();
            objects.insert(
                obj_id.clone(),
                json!({"object_data": {"bbox": [{"name": cat, "val": [cx, cy, w, h]}]}}),
            );
            top_objects.insert(obj_id, json!({"type": cat}));
        }
        frames.insert(idx.to_string(), json!({"frame_properties": {"file_name": img.file_name, "width": img.width, "height": img.height}, "objects": objects}));
    }
    json!({"openlabel": {"metadata": {"schema_version": "1.0.0"}, "objects": top_objects, "frames": frames}})
}
