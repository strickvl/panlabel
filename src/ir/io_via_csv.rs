//! VGG Image Annotator CSV bbox-only adapter.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde_json::{json, Value};

use super::io_bbox_adapters_common::{
    dataset_from_raw, image_dimensions_if_found, scalar_to_string, RawAnn, RawImage,
};
use super::model::{Dataset, DatasetInfo};
use super::BBoxXYXY;
use crate::error::PanlabelError;

const HEADER: [&str; 7] = [
    "filename",
    "file_size",
    "file_attributes",
    "region_count",
    "region_id",
    "region_shape_attributes",
    "region_attributes",
];

pub fn read_via_csv(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    read_via_csv_from_reader(BufReader::new(file), path, true)
}

#[cfg(feature = "fuzzing")]
pub fn from_via_csv_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    read_via_csv_from_reader(bytes, Path::new("<fuzz>"), false)
}

fn read_via_csv_from_reader<R: Read>(
    reader: R,
    path: &Path,
    probe_image_dimensions: bool,
) -> Result<Dataset, PanlabelError> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(reader);
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut images_by_name: BTreeMap<String, RawImage> = BTreeMap::new();
    let mut anns = Vec::new();
    let mut skipped = 0usize;
    for (idx, result) in rdr.records().enumerate() {
        let row_num = idx + 1;
        let rec = result.map_err(|source| PanlabelError::ViaCsvParse {
            path: path.to_path_buf(),
            source,
        })?;
        if row_num == 1 && is_via_csv_header(&rec) {
            continue;
        }
        if rec.len() != 7 {
            return Err(PanlabelError::ViaCsvInvalid {
                path: path.to_path_buf(),
                message: format!("row {row_num}: expected 7 columns, got {}", rec.len()),
            });
        }
        let filename = rec.get(0).unwrap_or("").to_string();
        if filename.is_empty() {
            continue;
        }
        let mut attrs = BTreeMap::new();
        if let Some(size) = rec.get(1).filter(|s| !s.is_empty()) {
            attrs.insert("via_csv_size_bytes".into(), size.to_string());
        }
        if let Ok(file_attrs) = serde_json::from_str::<Value>(rec.get(2).unwrap_or("{}")) {
            if let Some(obj) = file_attrs.as_object() {
                for (k, v) in obj {
                    if let Some(s) = scalar_to_string(v) {
                        attrs.insert(format!("via_csv_file_attr_{k}"), s);
                    }
                }
            }
        }
        images_by_name.entry(filename.clone()).or_insert_with(|| {
            let dims = if probe_image_dimensions {
                image_dimensions_if_found(base, &filename)
            } else {
                None
            }
            .unwrap_or((1, 1));
            RawImage {
                file_name: filename.clone(),
                width: dims.0,
                height: dims.1,
                attributes: attrs,
            }
        });
        let shape_str = rec.get(5).unwrap_or("");
        if shape_str.trim().is_empty() {
            continue;
        }
        let shape: Value =
            serde_json::from_str(shape_str).map_err(|e| PanlabelError::ViaCsvInvalid {
                path: path.to_path_buf(),
                message: format!("row {row_num}: invalid region_shape_attributes JSON: {e}"),
            })?;
        if shape.get("name").and_then(Value::as_str) != Some("rect") {
            skipped += 1;
            continue;
        }
        let x = shape.get("x").and_then(Value::as_f64).unwrap_or(0.0);
        let y = shape.get("y").and_then(Value::as_f64).unwrap_or(0.0);
        let w = shape.get("width").and_then(Value::as_f64).unwrap_or(0.0);
        let h = shape.get("height").and_then(Value::as_f64).unwrap_or(0.0);
        let region_attrs: Value =
            serde_json::from_str(rec.get(6).unwrap_or("{}")).map_err(|e| {
                PanlabelError::ViaCsvInvalid {
                    path: path.to_path_buf(),
                    message: format!("row {row_num}: invalid region_attributes JSON: {e}"),
                }
            })?;
        let label = resolve_label(&region_attrs);
        let mut ann_attrs = BTreeMap::new();
        if let Some(obj) = region_attrs.as_object() {
            for (k, v) in obj {
                if let Some(s) = scalar_to_string(v) {
                    ann_attrs.insert(format!("via_csv_region_attr_{k}"), s);
                }
            }
        }
        anns.push(RawAnn {
            image: filename,
            category: label,
            bbox: BBoxXYXY::from_xywh(x, y, w, h),
            confidence: None,
            attributes: ann_attrs,
        });
    }
    let mut info = DatasetInfo::default();
    if skipped > 0 {
        info.attributes.insert(
            "via_csv_non_rect_regions_skipped".into(),
            skipped.to_string(),
        );
    }
    Ok(dataset_from_raw(
        images_by_name.into_values().collect(),
        anns,
        vec![],
        info,
    ))
}

pub fn write_via_csv(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let csv_string = to_via_csv_string(dataset)?;
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(csv_string.as_bytes())
        .map_err(PanlabelError::Io)?;
    writer.flush().map_err(PanlabelError::Io)
}

pub fn to_via_csv_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let cat_lookup: BTreeMap<_, _> = dataset.categories.iter().map(|c| (c.id, c)).collect();
    let anns_by_image = super::io_bbox_adapters_common::annotations_by_image(dataset);
    let mut images: Vec<_> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());
    wtr.write_record(HEADER)
        .map_err(|source| PanlabelError::ViaCsvWrite {
            path: "<string>".into(),
            source,
        })?;
    for img in images {
        let anns = anns_by_image.get(&img.id).cloned().unwrap_or_default();
        if anns.is_empty() {
            wtr.write_record([&img.file_name, "", "{}", "0", "", "", ""])
                .map_err(|source| PanlabelError::ViaCsvWrite {
                    path: "<string>".into(),
                    source,
                })?;
        } else {
            let count = anns.len().to_string();
            for (idx, ann) in anns.iter().enumerate() {
                let (x, y, bw, bh) = ann.bbox.to_xywh();
                let shape = json!({"name":"rect","x":x,"y":y,"width":bw,"height":bh}).to_string();
                let attrs = json!({"label": cat_lookup.get(&ann.category_id).map(|c| c.name.as_str()).unwrap_or("object")}).to_string();
                wtr.write_record([
                    &img.file_name,
                    "",
                    "{}",
                    &count,
                    &idx.to_string(),
                    &shape,
                    &attrs,
                ])
                .map_err(|source| PanlabelError::ViaCsvWrite {
                    path: "<string>".into(),
                    source,
                })?;
            }
        }
    }
    let bytes = wtr
        .into_inner()
        .map_err(|e| PanlabelError::Io(e.into_error()))?;
    String::from_utf8(bytes).map_err(|e| PanlabelError::ViaCsvInvalid {
        path: "<string>".into(),
        message: e.to_string(),
    })
}

pub(crate) fn is_via_csv_header(record: &csv::StringRecord) -> bool {
    record.len() == 7
        && HEADER
            .iter()
            .enumerate()
            .all(|(i, h)| record.get(i).is_some_and(|c| c.eq_ignore_ascii_case(h)))
}

fn resolve_label(attrs: &Value) -> String {
    let Some(obj) = attrs.as_object() else {
        return "object".into();
    };
    for key in ["label", "class"] {
        if let Some(s) = obj
            .get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            return s.to_string();
        }
    }
    let scalars: Vec<String> = obj.values().filter_map(scalar_to_string).collect();
    if scalars.len() == 1 && !scalars[0].is_empty() {
        scalars[0].clone()
    } else {
        "object".into()
    }
}
