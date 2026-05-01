//! WIDER Face aggregate TXT adapter.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use super::io_bbox_adapters_common::{
    dataset_from_raw, image_dimensions_or_error, RawAnn, RawImage,
};
use super::model::{Dataset, DatasetInfo};
use super::BBoxXYXY;
use crate::error::PanlabelError;

const ATTRS: [&str; 6] = [
    "wider_face_blur",
    "wider_face_expression",
    "wider_face_illumination",
    "wider_face_invalid",
    "wider_face_occlusion",
    "wider_face_pose",
];

pub fn read_wider_face_txt(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .collect::<Result<_, _>>()
        .map_err(PanlabelError::Io)?;
    parse_wider_lines(path, &lines)
}

pub fn write_wider_face_txt(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let text = to_wider_face_txt_string(dataset)?;
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(text.as_bytes())
        .map_err(PanlabelError::Io)?;
    writer.flush().map_err(PanlabelError::Io)
}

pub fn to_wider_face_txt_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let anns_by_image = super::io_bbox_adapters_common::annotations_by_image(dataset);
    let mut images: Vec<_> = dataset.images.iter().collect();
    images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    let mut out = String::new();
    out.push_str(&format!("{}\n", images.len()));
    for img in images {
        out.push_str(&format!("{}\n", img.file_name));
        let anns = anns_by_image.get(&img.id).cloned().unwrap_or_default();
        out.push_str(&format!("{}\n", anns.len()));
        for ann in anns {
            let (x, y, w, h) = ann.bbox.to_xywh();
            let vals: Vec<String> = ATTRS
                .iter()
                .map(|k| {
                    ann.attributes
                        .get(*k)
                        .cloned()
                        .unwrap_or_else(|| "0".into())
                })
                .collect();
            out.push_str(&format!("{x} {y} {w} {h} {}\n", vals.join(" ")));
        }
    }
    Ok(out)
}

pub(crate) fn looks_like_wider_face_txt_file(path: &Path) -> Result<bool, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .take(4)
        .collect::<Result<_, _>>()
        .map_err(PanlabelError::Io)?;
    Ok(lines.len() >= 3
        && lines[0].trim().parse::<usize>().is_ok()
        && lines[2].trim().parse::<usize>().is_ok())
}

fn parse_wider_lines(path: &Path, lines: &[String]) -> Result<Dataset, PanlabelError> {
    if lines.is_empty() {
        return Err(PanlabelError::WiderFaceTxtInvalid {
            path: path.to_path_buf(),
            message: "empty file".into(),
        });
    }
    let count: usize = lines[0]
        .trim()
        .parse()
        .map_err(|_| PanlabelError::WiderFaceTxtParse {
            path: path.to_path_buf(),
            line: 1,
            message: "expected number of images".into(),
        })?;
    let mut cursor = 1usize;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut images = Vec::new();
    let mut anns = Vec::new();
    for _ in 0..count {
        if cursor + 1 >= lines.len() {
            return Err(PanlabelError::WiderFaceTxtInvalid {
                path: path.to_path_buf(),
                message: "truncated image block".into(),
            });
        }
        let image_name = lines[cursor].trim().to_string();
        cursor += 1;
        let n_boxes: usize =
            lines[cursor]
                .trim()
                .parse()
                .map_err(|_| PanlabelError::WiderFaceTxtParse {
                    path: path.to_path_buf(),
                    line: cursor + 1,
                    message: "expected number of boxes".into(),
                })?;
        cursor += 1;
        let (width, height) = image_dimensions_or_error(
            base,
            &image_name,
            || PanlabelError::WiderFaceImageNotFound {
                path: path.to_path_buf(),
                image_ref: image_name.clone(),
            },
            |p, source| PanlabelError::WiderFaceImageDimensionRead { path: p, source },
        )?;
        images.push(RawImage {
            file_name: image_name.clone(),
            width,
            height,
            attributes: BTreeMap::new(),
        });
        for _ in 0..n_boxes {
            if cursor >= lines.len() {
                return Err(PanlabelError::WiderFaceTxtInvalid {
                    path: path.to_path_buf(),
                    message: "truncated bbox rows".into(),
                });
            }
            let line_no = cursor + 1;
            let parts: Vec<&str> = lines[cursor].split_whitespace().collect();
            cursor += 1;
            if parts.len() < 4 {
                return Err(PanlabelError::WiderFaceTxtParse {
                    path: path.to_path_buf(),
                    line: line_no,
                    message: "expected at least x y w h".into(),
                });
            }
            let parse = |i: usize| {
                parts[i]
                    .parse::<f64>()
                    .map_err(|_| PanlabelError::WiderFaceTxtParse {
                        path: path.to_path_buf(),
                        line: line_no,
                        message: format!("invalid numeric field {}", i + 1),
                    })
            };
            let mut attrs = BTreeMap::new();
            for (idx, key) in ATTRS.iter().enumerate() {
                if let Some(v) = parts.get(4 + idx) {
                    attrs.insert((*key).into(), (*v).into());
                }
            }
            anns.push(RawAnn {
                image: image_name.clone(),
                category: "face".into(),
                bbox: BBoxXYXY::from_xywh(parse(0)?, parse(1)?, parse(2)?, parse(3)?),
                confidence: None,
                attributes: attrs,
            });
        }
    }
    Ok(dataset_from_raw(
        images,
        anns,
        vec![("face".into(), None)],
        DatasetInfo::default(),
    ))
}
