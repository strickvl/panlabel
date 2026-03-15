//! KITTI format reader and writer.
//!
//! This module supports the KITTI object detection layout with a `label_2/`
//! directory containing one `.txt` file per image and an optional `image_2/`
//! directory for corresponding images. The canonical IR remains pixel-space XYXY.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

const LABEL_EXTENSION: &str = "txt";
const IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "bmp", "webp"];
const IMAGE_DIR_README: &str = "This directory is a placeholder. Panlabel does not copy image files during conversion.\nPlace your original images here to complete the KITTI dataset layout.\n";

struct KittiRow {
    class_name: String,
    truncated: f64,
    occluded: u8,
    alpha: f64,
    bbox_left: f64,
    bbox_top: f64,
    bbox_right: f64,
    bbox_bottom: f64,
    dim_height: f64,
    dim_width: f64,
    dim_length: f64,
    loc_x: f64,
    loc_y: f64,
    loc_z: f64,
    rotation_y: f64,
    score: Option<f64>,
}

struct KittiLayout {
    #[cfg_attr(not(test), allow(dead_code))]
    root: PathBuf,
    labels_dir: PathBuf,
    images_dir: Option<PathBuf>,
}

pub fn read_kitti_dir(path: &Path) -> Result<Dataset, PanlabelError> {
    let layout = discover_layout(path)?;
    let label_files = collect_label_files(&layout.labels_dir)?;

    let mut parsed_files: Vec<(PathBuf, String, Vec<KittiRow>)> =
        Vec::with_capacity(label_files.len());
    for label_path in &label_files {
        let stem = label_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let contents = fs::read_to_string(label_path).map_err(PanlabelError::Io)?;
        let rows = parse_kitti_label(&contents, label_path)?;
        parsed_files.push((label_path.clone(), stem, rows));
    }

    // Resolve image dimensions for each label file
    let mut image_defs: BTreeMap<String, (String, u32, u32)> = BTreeMap::new();
    let mut category_names = BTreeSet::new();

    for (label_path, stem, rows) in &parsed_files {
        if image_defs.contains_key(stem) {
            return Err(PanlabelError::KittiLabelParse {
                path: label_path.clone(),
                line: 0,
                message: format!("duplicate label stem '{stem}' found in multiple files"),
            });
        }

        for row in rows {
            category_names.insert(row.class_name.clone());
        }

        let (file_name, width, height) = resolve_image(label_path, stem, &layout)?;
        image_defs.insert(stem.clone(), (file_name, width, height));
    }

    let categories: Vec<Category> = category_names
        .into_iter()
        .enumerate()
        .map(|(idx, name)| Category::new((idx + 1) as u64, name))
        .collect();

    let category_id_by_name: BTreeMap<String, CategoryId> =
        categories.iter().map(|c| (c.name.clone(), c.id)).collect();

    let images: Vec<Image> = image_defs
        .values()
        .enumerate()
        .map(|(idx, (file_name, width, height))| {
            Image::new((idx + 1) as u64, file_name.clone(), *width, *height)
        })
        .collect();

    let image_id_by_stem: BTreeMap<String, ImageId> = image_defs
        .keys()
        .enumerate()
        .map(|(idx, stem)| (stem.clone(), ImageId::new((idx + 1) as u64)))
        .collect();

    let mut annotations = Vec::new();
    let mut next_ann_id: u64 = 1;

    for (label_path, stem, rows) in parsed_files {
        let image_id = image_id_by_stem[&stem];

        for row in rows {
            let category_id = category_id_by_name
                .get(&row.class_name)
                .copied()
                .ok_or_else(|| PanlabelError::KittiLabelParse {
                    path: label_path.clone(),
                    line: 0,
                    message: format!(
                        "internal error: category '{}' missing from lookup",
                        row.class_name
                    ),
                })?;

            let mut ann = Annotation::new(
                AnnotationId::new(next_ann_id),
                image_id,
                category_id,
                BBoxXYXY::<Pixel>::from_xyxy(
                    row.bbox_left,
                    row.bbox_top,
                    row.bbox_right,
                    row.bbox_bottom,
                ),
            );

            if let Some(score) = row.score {
                ann.confidence = Some(score);
            }

            ann.attributes
                .insert("kitti_truncated".to_string(), row.truncated.to_string());
            ann.attributes
                .insert("kitti_occluded".to_string(), row.occluded.to_string());
            ann.attributes
                .insert("kitti_alpha".to_string(), row.alpha.to_string());
            ann.attributes
                .insert("kitti_dim_height".to_string(), row.dim_height.to_string());
            ann.attributes
                .insert("kitti_dim_width".to_string(), row.dim_width.to_string());
            ann.attributes
                .insert("kitti_dim_length".to_string(), row.dim_length.to_string());
            ann.attributes
                .insert("kitti_loc_x".to_string(), row.loc_x.to_string());
            ann.attributes
                .insert("kitti_loc_y".to_string(), row.loc_y.to_string());
            ann.attributes
                .insert("kitti_loc_z".to_string(), row.loc_z.to_string());
            ann.attributes
                .insert("kitti_rotation_y".to_string(), row.rotation_y.to_string());

            annotations.push(ann);
            next_ann_id += 1;
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

pub fn write_kitti_dir(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    fs::create_dir_all(path).map_err(PanlabelError::Io)?;

    let labels_dir = path.join("label_2");
    let images_dir = path.join("image_2");
    fs::create_dir_all(&labels_dir).map_err(PanlabelError::Io)?;
    fs::create_dir_all(&images_dir).map_err(PanlabelError::Io)?;
    fs::write(images_dir.join("README.txt"), IMAGE_DIR_README).map_err(PanlabelError::Io)?;

    let image_by_id: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();
    let category_name_by_id: BTreeMap<CategoryId, String> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.clone()))
        .collect();

    let mut annotations_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        if !image_by_id.contains_key(&ann.image_id) {
            return Err(PanlabelError::KittiWriteError {
                path: path.to_path_buf(),
                message: format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            });
        }
        if !category_name_by_id.contains_key(&ann.category_id) {
            return Err(PanlabelError::KittiWriteError {
                path: path.to_path_buf(),
                message: format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            });
        }
        annotations_by_image
            .entry(ann.image_id)
            .or_default()
            .push(ann);
    }

    let mut images_sorted: Vec<&Image> = dataset.images.iter().collect();
    images_sorted.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    for image in images_sorted {
        if image.file_name.contains('/') || image.file_name.contains('\\') {
            return Err(PanlabelError::KittiWriteError {
                path: path.to_path_buf(),
                message: format!(
                    "KITTI is a flat format; image file_name '{}' contains a path separator",
                    image.file_name
                ),
            });
        }

        let label_name = Path::new(&image.file_name).with_extension(LABEL_EXTENSION);
        let label_path = labels_dir.join(&label_name);

        let mut anns = annotations_by_image.remove(&image.id).unwrap_or_default();
        anns.sort_by_key(|ann| ann.id);

        let mut file = fs::File::create(&label_path).map_err(PanlabelError::Io)?;

        for ann in anns {
            let class_name = &category_name_by_id[&ann.category_id];
            let row = annotation_to_row(ann, class_name);
            let line = format_kitti_row(&row);
            writeln!(file, "{}", line).map_err(PanlabelError::Io)?;
        }
    }

    Ok(())
}

pub fn from_kitti_str(txt: &str) -> Result<(), PanlabelError> {
    parse_kitti_label(txt, Path::new("<memory>"))?;
    Ok(())
}

pub fn from_kitti_slice(bytes: &[u8]) -> Result<(), PanlabelError> {
    let txt = std::str::from_utf8(bytes).map_err(|source| PanlabelError::KittiLabelParse {
        path: PathBuf::from("<memory>"),
        line: 0,
        message: format!("input is not valid UTF-8: {source}"),
    })?;
    from_kitti_str(txt)
}

pub fn to_kitti_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let category_name_by_id: BTreeMap<CategoryId, String> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.clone()))
        .collect();

    let mut lines = Vec::new();
    let mut anns_sorted: Vec<&Annotation> = dataset.annotations.iter().collect();
    anns_sorted.sort_by_key(|ann| ann.id);

    for ann in anns_sorted {
        let class_name = category_name_by_id.get(&ann.category_id).ok_or_else(|| {
            PanlabelError::KittiWriteError {
                path: PathBuf::from("<string>"),
                message: format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            }
        })?;
        let row = annotation_to_row(ann, class_name);
        lines.push(format_kitti_row(&row));
    }

    Ok(lines.join("\n"))
}

// ---------------------------------------------------------------------------
// Layout discovery
// ---------------------------------------------------------------------------

fn discover_layout(input: &Path) -> Result<KittiLayout, PanlabelError> {
    if !input.is_dir() {
        return Err(PanlabelError::KittiLayoutInvalid {
            path: input.to_path_buf(),
            message: "input must be a directory".to_string(),
        });
    }

    let (root, labels_dir) = if input.join("label_2").is_dir() {
        (input.to_path_buf(), input.join("label_2"))
    } else if is_dir_named(input, "label_2") {
        let root = input
            .parent()
            .ok_or_else(|| PanlabelError::KittiLayoutInvalid {
                path: input.to_path_buf(),
                message: "label_2 directory has no parent directory".to_string(),
            })?
            .to_path_buf();
        (root, input.to_path_buf())
    } else {
        return Err(PanlabelError::KittiLayoutInvalid {
            path: input.to_path_buf(),
            message:
                "expected a KITTI dataset root containing label_2/ or a label_2/ directory itself"
                    .to_string(),
        });
    };

    let images_dir = root.join("image_2");
    let images_dir = images_dir.is_dir().then_some(images_dir);

    Ok(KittiLayout {
        root,
        labels_dir,
        images_dir,
    })
}

// ---------------------------------------------------------------------------
// File collection
// ---------------------------------------------------------------------------

fn collect_label_files(dir: &Path) -> Result<Vec<PathBuf>, PanlabelError> {
    let mut files = Vec::new();

    for entry in fs::read_dir(dir).map_err(PanlabelError::Io)? {
        let entry = entry.map_err(PanlabelError::Io)?;
        let path = entry.path();
        if path.is_file() && has_extension(&path, LABEL_EXTENSION) {
            files.push(path);
        }
    }

    files.sort_by_cached_key(|p| file_name_string(p));
    Ok(files)
}

// ---------------------------------------------------------------------------
// Image resolution
// ---------------------------------------------------------------------------

fn resolve_image(
    _label_path: &Path,
    stem: &str,
    layout: &KittiLayout,
) -> Result<(String, u32, u32), PanlabelError> {
    if let Some(images_dir) = &layout.images_dir {
        if let Some(image_path) = find_image_for_stem(images_dir, stem) {
            let file_name = image_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(stem)
                .to_string();
            let (width, height) = read_image_dimensions(&image_path)?;
            return Ok((file_name, width, height));
        }
    }

    // No image found: fall back to stem + .png and 0x0 dimensions
    Ok((format!("{stem}.png"), 0, 0))
}

fn find_image_for_stem(images_dir: &Path, stem: &str) -> Option<PathBuf> {
    for ext in IMAGE_EXTENSIONS {
        let candidate = images_dir.join(format!("{stem}.{ext}"));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn read_image_dimensions(path: &Path) -> Result<(u32, u32), PanlabelError> {
    let size = imagesize::size(path).map_err(|source| PanlabelError::KittiImageDimensionRead {
        path: path.to_path_buf(),
        source,
    })?;

    let width: u32 = size
        .width
        .try_into()
        .map_err(|_| PanlabelError::KittiLayoutInvalid {
            path: path.to_path_buf(),
            message: format!("image width {} does not fit in u32", size.width),
        })?;
    let height: u32 = size
        .height
        .try_into()
        .map_err(|_| PanlabelError::KittiLayoutInvalid {
            path: path.to_path_buf(),
            message: format!("image height {} does not fit in u32", size.height),
        })?;

    Ok((width, height))
}

// ---------------------------------------------------------------------------
// Label parsing
// ---------------------------------------------------------------------------

fn parse_kitti_label(txt: &str, path: &Path) -> Result<Vec<KittiRow>, PanlabelError> {
    let mut rows = Vec::new();
    for (line_idx, line) in txt.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row = parse_kitti_line(line, path, line_idx + 1)?;
        rows.push(row);
    }
    Ok(rows)
}

fn parse_kitti_line(line: &str, path: &Path, line_num: usize) -> Result<KittiRow, PanlabelError> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    let n = fields.len();

    if !(15..=16).contains(&n) {
        return Err(PanlabelError::KittiLabelParse {
            path: path.to_path_buf(),
            line: line_num,
            message: format!("expected 15 or 16 fields, got {n}"),
        });
    }

    let class_name = fields[0].to_string();

    let truncated = parse_f64(fields[1], "truncated", path, line_num)?;
    let occluded = parse_u8(fields[2], "occluded", path, line_num)?;
    let alpha = parse_f64(fields[3], "alpha", path, line_num)?;
    let bbox_left = parse_f64(fields[4], "bbox_left", path, line_num)?;
    let bbox_top = parse_f64(fields[5], "bbox_top", path, line_num)?;
    let bbox_right = parse_f64(fields[6], "bbox_right", path, line_num)?;
    let bbox_bottom = parse_f64(fields[7], "bbox_bottom", path, line_num)?;
    let dim_height = parse_f64(fields[8], "dim_height", path, line_num)?;
    let dim_width = parse_f64(fields[9], "dim_width", path, line_num)?;
    let dim_length = parse_f64(fields[10], "dim_length", path, line_num)?;
    let loc_x = parse_f64(fields[11], "loc_x", path, line_num)?;
    let loc_y = parse_f64(fields[12], "loc_y", path, line_num)?;
    let loc_z = parse_f64(fields[13], "loc_z", path, line_num)?;
    let rotation_y = parse_f64(fields[14], "rotation_y", path, line_num)?;

    let score = if n == 16 {
        Some(parse_f64(fields[15], "score", path, line_num)?)
    } else {
        None
    };

    Ok(KittiRow {
        class_name,
        truncated,
        occluded,
        alpha,
        bbox_left,
        bbox_top,
        bbox_right,
        bbox_bottom,
        dim_height,
        dim_width,
        dim_length,
        loc_x,
        loc_y,
        loc_z,
        rotation_y,
        score,
    })
}

fn parse_f64(value: &str, field: &str, path: &Path, line_num: usize) -> Result<f64, PanlabelError> {
    value
        .parse::<f64>()
        .map_err(|_| PanlabelError::KittiLabelParse {
            path: path.to_path_buf(),
            line: line_num,
            message: format!("invalid {field} value '{value}'; expected floating-point number"),
        })
}

fn parse_u8(value: &str, field: &str, path: &Path, line_num: usize) -> Result<u8, PanlabelError> {
    value
        .parse::<u8>()
        .map_err(|_| PanlabelError::KittiLabelParse {
            path: path.to_path_buf(),
            line: line_num,
            message: format!("invalid {field} value '{value}'; expected integer 0-255"),
        })
}

// ---------------------------------------------------------------------------
// Writer helpers
// ---------------------------------------------------------------------------

fn annotation_to_row(ann: &Annotation, class_name: &str) -> KittiRow {
    let attr = |key: &str, default: f64| -> f64 {
        ann.attributes
            .get(key)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(default)
    };

    let occluded: u8 = ann
        .attributes
        .get("kitti_occluded")
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(0);

    KittiRow {
        class_name: class_name.to_string(),
        truncated: attr("kitti_truncated", 0.0),
        occluded,
        alpha: attr("kitti_alpha", -10.0),
        bbox_left: ann.bbox.xmin(),
        bbox_top: ann.bbox.ymin(),
        bbox_right: ann.bbox.xmax(),
        bbox_bottom: ann.bbox.ymax(),
        dim_height: attr("kitti_dim_height", -1.0),
        dim_width: attr("kitti_dim_width", -1.0),
        dim_length: attr("kitti_dim_length", -1.0),
        loc_x: attr("kitti_loc_x", -1000.0),
        loc_y: attr("kitti_loc_y", -1000.0),
        loc_z: attr("kitti_loc_z", -1000.0),
        rotation_y: attr("kitti_rotation_y", -10.0),
        score: ann.confidence,
    }
}

fn format_kitti_row(row: &KittiRow) -> String {
    let base = format!(
        "{} {:.2} {} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
        row.class_name,
        row.truncated,
        row.occluded,
        row.alpha,
        row.bbox_left,
        row.bbox_top,
        row.bbox_right,
        row.bbox_bottom,
        row.dim_height,
        row.dim_width,
        row.dim_length,
        row.loc_x,
        row.loc_y,
        row.loc_z,
        row.rotation_y,
    );

    match row.score {
        Some(score) => format!("{} {:.2}", base, score),
        None => base,
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

fn has_extension(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

fn is_dir_named(path: &Path, dir_name: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case(dir_name))
        .unwrap_or(false)
}

fn file_name_string(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sample_line() {
        let line =
            "Car 0.00 0 -1.57 614.24 181.78 727.31 284.77 1.52 1.60 3.23 1.51 1.65 13.73 -1.59";
        let row = parse_kitti_line(line, Path::new("test.txt"), 1).expect("parse line");
        assert_eq!(row.class_name, "Car");
        assert!((row.truncated - 0.0).abs() < 1e-6);
        assert_eq!(row.occluded, 0);
        assert!((row.alpha - (-1.57)).abs() < 1e-6);
        assert!((row.bbox_left - 614.24).abs() < 1e-6);
        assert!((row.bbox_top - 181.78).abs() < 1e-6);
        assert!((row.bbox_right - 727.31).abs() < 1e-6);
        assert!((row.bbox_bottom - 284.77).abs() < 1e-6);
        assert!((row.dim_height - 1.52).abs() < 1e-6);
        assert!((row.dim_width - 1.60).abs() < 1e-6);
        assert!((row.dim_length - 3.23).abs() < 1e-6);
        assert!((row.loc_x - 1.51).abs() < 1e-6);
        assert!((row.loc_y - 1.65).abs() < 1e-6);
        assert!((row.loc_z - 13.73).abs() < 1e-6);
        assert!((row.rotation_y - (-1.59)).abs() < 1e-6);
        assert!(row.score.is_none());
    }

    #[test]
    fn parse_line_with_score() {
        let line = "Pedestrian 0.50 1 0.20 100.00 150.00 200.00 300.00 1.70 0.60 0.80 -5.00 1.80 20.00 -0.50 0.95";
        let row = parse_kitti_line(line, Path::new("test.txt"), 1).expect("parse line with score");
        assert_eq!(row.class_name, "Pedestrian");
        assert_eq!(row.occluded, 1);
        assert!((row.score.unwrap() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn parse_rejects_wrong_field_count() {
        let line = "Car 0.00 0 -1.57 614.24 181.78 727.31 284.77";
        let result = parse_kitti_line(line, Path::new("test.txt"), 1);
        assert!(result.is_err());
    }

    #[test]
    fn roundtrip_via_string() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "000001.png", 1224, 370)],
            categories: vec![Category::new(1u64, "Car")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(614.24, 181.78, 727.31, 284.77),
            )
            .with_attribute("kitti_truncated", "0.00")
            .with_attribute("kitti_occluded", "0")
            .with_attribute("kitti_alpha", "-1.57")
            .with_attribute("kitti_dim_height", "1.52")
            .with_attribute("kitti_dim_width", "1.60")
            .with_attribute("kitti_dim_length", "3.23")
            .with_attribute("kitti_loc_x", "1.51")
            .with_attribute("kitti_loc_y", "1.65")
            .with_attribute("kitti_loc_z", "13.73")
            .with_attribute("kitti_rotation_y", "-1.59")],
            ..Default::default()
        };

        let output = to_kitti_string(&dataset).expect("serialize");
        let parsed = parse_kitti_label(&output, Path::new("<test>")).expect("parse roundtrip");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].class_name, "Car");
        assert!((parsed[0].bbox_left - 614.24).abs() < 0.01);
        assert!((parsed[0].bbox_top - 181.78).abs() < 0.01);
        assert!((parsed[0].bbox_right - 727.31).abs() < 0.01);
        assert!((parsed[0].bbox_bottom - 284.77).abs() < 0.01);
    }

    #[test]
    fn roundtrip_dir() {
        let temp = tempfile::tempdir().expect("create temp dir");
        let dataset = Dataset {
            images: vec![
                Image::new(1u64, "img1.png", 640, 480),
                Image::new(2u64, "img2.png", 800, 600),
            ],
            categories: vec![
                Category::new(1u64, "Car"),
                Category::new(2u64, "Pedestrian"),
            ],
            annotations: vec![
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 200.0),
                ),
                Annotation::new(
                    2u64,
                    2u64,
                    2u64,
                    BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 150.0, 250.0),
                ),
            ],
            ..Default::default()
        };

        write_kitti_dir(temp.path(), &dataset).expect("write kitti dir");

        assert!(temp.path().join("label_2/img1.txt").is_file());
        assert!(temp.path().join("label_2/img2.txt").is_file());
        assert!(temp.path().join("image_2/README.txt").is_file());

        // Read back (without images, so dimensions will be 0x0)
        let restored = read_kitti_dir(temp.path()).expect("read kitti dir");
        assert_eq!(restored.images.len(), 2);
        assert_eq!(restored.categories.len(), 2);
        assert_eq!(restored.annotations.len(), 2);

        // Check bbox preserved
        let ann = &restored.annotations[0];
        assert!((ann.bbox.xmin() - 10.0).abs() < 0.01);
        assert!((ann.bbox.ymin() - 20.0).abs() < 0.01);
        assert!((ann.bbox.xmax() - 100.0).abs() < 0.01);
        assert!((ann.bbox.ymax() - 200.0).abs() < 0.01);
    }

    #[test]
    fn writer_rejects_path_separators_in_filename() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "subdir/img.png", 100, 100)],
            categories: vec![Category::new(1u64, "Car")],
            annotations: vec![],
            ..Default::default()
        };

        let temp = tempfile::tempdir().expect("create temp dir");
        let result = write_kitti_dir(temp.path(), &dataset);
        assert!(result.is_err());
    }

    #[test]
    fn empty_label_file_for_unannotated_image() {
        let temp = tempfile::tempdir().expect("create temp dir");
        let dataset = Dataset {
            images: vec![Image::new(1u64, "empty.png", 100, 100)],
            categories: vec![Category::new(1u64, "Car")],
            annotations: vec![],
            ..Default::default()
        };

        write_kitti_dir(temp.path(), &dataset).expect("write");
        let contents =
            fs::read_to_string(temp.path().join("label_2/empty.txt")).expect("read label");
        assert!(contents.is_empty());
    }

    #[test]
    fn discover_layout_accepts_root_or_label_dir() {
        let temp = tempfile::tempdir().expect("create temp dir");
        fs::create_dir_all(temp.path().join("label_2")).expect("create label dir");
        fs::create_dir_all(temp.path().join("image_2")).expect("create image dir");

        let root_layout = discover_layout(temp.path()).expect("discover from root");
        assert_eq!(root_layout.root, temp.path());
        assert_eq!(root_layout.labels_dir, temp.path().join("label_2"));
        assert_eq!(root_layout.images_dir, Some(temp.path().join("image_2")));

        let label_layout =
            discover_layout(&temp.path().join("label_2")).expect("discover from label_2");
        assert_eq!(label_layout.root, temp.path());
        assert_eq!(label_layout.labels_dir, temp.path().join("label_2"));
        assert_eq!(label_layout.images_dir, Some(temp.path().join("image_2")));
    }

    #[test]
    fn from_kitti_str_validates_parse() {
        let valid =
            "Car 0.00 0 -1.57 100.00 150.00 200.00 300.00 1.50 1.60 3.20 1.00 1.50 10.00 -1.50";
        assert!(from_kitti_str(valid).is_ok());

        let invalid = "Car 0.00 notanumber";
        assert!(from_kitti_str(invalid).is_err());
    }

    #[test]
    fn writer_defaults_for_missing_attributes() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "test.png", 100, 100)],
            categories: vec![Category::new(1u64, "Car")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 30.0, 40.0),
            )],
            ..Default::default()
        };

        let output = to_kitti_string(&dataset).expect("serialize");
        let parsed = parse_kitti_label(&output, Path::new("<test>")).expect("parse");
        assert_eq!(parsed.len(), 1);
        assert!((parsed[0].truncated - 0.0).abs() < 1e-6);
        assert_eq!(parsed[0].occluded, 0);
        assert!((parsed[0].alpha - (-10.0)).abs() < 1e-6);
        assert!((parsed[0].dim_height - (-1.0)).abs() < 1e-6);
        assert!((parsed[0].loc_x - (-1000.0)).abs() < 1e-1);
        assert!((parsed[0].rotation_y - (-10.0)).abs() < 1e-6);
    }
}
