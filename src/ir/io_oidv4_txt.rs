//! OIDv4 Toolkit-style TXT adapter.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::io_adapter_common::write_images_readme;
use super::io_bbox_adapters_common::{
    dataset_from_raw, image_dimensions_or_error, RawAnn, RawImage,
};
use super::model::{Dataset, DatasetInfo};
use super::BBoxXYXY;
use crate::error::PanlabelError;

pub fn read_oidv4_txt(path: &Path) -> Result<Dataset, PanlabelError> {
    if path.is_dir() {
        read_oidv4_dir(path)
    } else {
        read_oidv4_file(path)
    }
}

pub fn write_oidv4_txt(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    if path.extension().is_some() {
        write_oidv4_single_file(path, dataset)
    } else {
        write_oidv4_dir(path, dataset)
    }
}

pub(crate) fn looks_like_oidv4_txt_file(path: &Path) -> Result<bool, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    for line in BufReader::new(file).lines().take(8) {
        let line = line.map_err(PanlabelError::Io)?;
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        return Ok(parts.len() == 5 && parts[1..].iter().all(|p| p.parse::<f64>().is_ok()));
    }
    Ok(true)
}

pub(crate) fn dir_has_oidv4_label_files(path: &Path) -> Result<bool, PanlabelError> {
    for entry in walkdir::WalkDir::new(path).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;
        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("txt"))
            && entry
                .path()
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                == Some("Label")
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn read_oidv4_dir(path: &Path) -> Result<Dataset, PanlabelError> {
    let mut label_files = Vec::new();
    for entry in walkdir::WalkDir::new(path).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::FormatDetectionFailed {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;
        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("txt"))
            && entry
                .path()
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                == Some("Label")
        {
            label_files.push(entry.path().to_path_buf());
        }
    }
    label_files.sort();
    oidv4_files_to_ir(path, label_files)
}

fn read_oidv4_file(path: &Path) -> Result<Dataset, PanlabelError> {
    oidv4_files_to_ir(
        path.parent().unwrap_or_else(|| Path::new(".")),
        vec![path.to_path_buf()],
    )
}

fn oidv4_files_to_ir(root: &Path, files: Vec<PathBuf>) -> Result<Dataset, PanlabelError> {
    let mut images = Vec::new();
    let mut anns = Vec::new();
    let mut seen = BTreeSet::new();
    for label_path in files {
        let stem = label_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
            .to_string();
        let image_name = resolve_image_name(root, &label_path, &stem);
        if seen.insert(image_name.clone()) {
            let base = label_path.parent().and_then(|p| p.parent()).unwrap_or(root);
            let (width, height) = image_dimensions_or_error(
                base,
                &image_name,
                || PanlabelError::Oidv4ImageNotFound {
                    path: label_path.clone(),
                    image_ref: image_name.clone(),
                },
                |p, source| PanlabelError::Oidv4ImageDimensionRead { path: p, source },
            )?;
            images.push(RawImage {
                file_name: image_name.clone(),
                width,
                height,
                attributes: BTreeMap::new(),
            });
        }
        let file = File::open(&label_path).map_err(PanlabelError::Io)?;
        for (idx, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(PanlabelError::Io)?;
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 5 {
                return Err(PanlabelError::Oidv4TxtParse {
                    path: label_path.clone(),
                    line: idx + 1,
                    message: "expected class xmin ymin xmax ymax".into(),
                });
            }
            let parse = |i: usize| {
                parts[i]
                    .parse::<f64>()
                    .map_err(|_| PanlabelError::Oidv4TxtParse {
                        path: label_path.clone(),
                        line: idx + 1,
                        message: format!("invalid numeric field {}", i + 1),
                    })
            };
            anns.push(RawAnn {
                image: image_name.clone(),
                category: parts[0].to_string(),
                bbox: BBoxXYXY::from_xyxy(parse(1)?, parse(2)?, parse(3)?, parse(4)?),
                confidence: None,
                attributes: BTreeMap::new(),
            });
        }
    }
    Ok(dataset_from_raw(
        images,
        anns,
        vec![],
        DatasetInfo::default(),
    ))
}

fn resolve_image_name(root: &Path, label_path: &Path, stem: &str) -> String {
    let base = label_path.parent().and_then(|p| p.parent()).unwrap_or(root);
    for ext in super::io_bbox_adapters_common::IMAGE_EXTENSIONS {
        let name = format!("{stem}{ext}");
        if base.join(&name).is_file() || root.join("images").join(&name).is_file() {
            return name;
        }
    }
    format!("{stem}.jpg")
}

fn write_oidv4_dir(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let labels = path.join("Label");
    fs::create_dir_all(&labels).map_err(PanlabelError::Io)?;
    write_images_readme(path, "PanLabel does not copy image binaries.\n")?;
    let cat_lookup: BTreeMap<_, _> = dataset.categories.iter().map(|c| (c.id, c)).collect();
    let anns_by_image = super::io_bbox_adapters_common::annotations_by_image(dataset);
    for img in &dataset.images {
        let stem = Path::new(&img.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&img.file_name);
        let mut out = String::new();
        for ann in anns_by_image
            .get(&img.id)
            .into_iter()
            .flat_map(|v| v.iter())
        {
            out.push_str(&format!(
                "{} {} {} {} {}\n",
                cat_lookup
                    .get(&ann.category_id)
                    .map(|c| c.name.as_str())
                    .unwrap_or("object"),
                ann.bbox.xmin(),
                ann.bbox.ymin(),
                ann.bbox.xmax(),
                ann.bbox.ymax()
            ));
        }
        fs::write(labels.join(format!("{stem}.txt")), out).map_err(PanlabelError::Io)?;
    }
    Ok(())
}

fn write_oidv4_single_file(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let cat_lookup: BTreeMap<_, _> = dataset.categories.iter().map(|c| (c.id, c)).collect();
    let mut anns: Vec<_> = dataset.annotations.iter().collect();
    anns.sort_by_key(|a| a.id);
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let mut w = BufWriter::new(file);
    for ann in anns {
        writeln!(
            w,
            "{} {} {} {} {}",
            cat_lookup
                .get(&ann.category_id)
                .map(|c| c.name.as_str())
                .unwrap_or("object"),
            ann.bbox.xmin(),
            ann.bbox.ymin(),
            ann.bbox.xmax(),
            ann.bbox.ymax()
        )
        .map_err(PanlabelError::Io)?;
    }
    Ok(())
}
