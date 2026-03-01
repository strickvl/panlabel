//! Ultralytics-style YOLO reader and writer.
//!
//! This module handles directory-based YOLO datasets with `images/` and `labels/`
//! trees. The canonical IR representation remains pixel-space XYXY boxes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use walkdir::WalkDir;

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Normalized};
use crate::error::PanlabelError;

const IMAGE_EXTENSIONS: [&str; 5] = ["jpg", "png", "jpeg", "bmp", "webp"];
const LABEL_EXTENSION: &str = "txt";

/// Read a YOLO dataset directory into IR.
///
/// `path` may be the dataset root containing `images/` + `labels/`, or the
/// `labels/` directory itself.
pub fn read_yolo_dir(path: &Path) -> Result<Dataset, PanlabelError> {
    let layout = discover_layout(path)?;
    let class_map = read_class_map(&layout)?;

    let mut image_files = collect_files_with_extensions(&layout.images_dir, &IMAGE_EXTENSIONS)?;
    image_files.sort_by_cached_key(|image_path| rel_string(&layout.images_dir, image_path));

    let mut images = Vec::with_capacity(image_files.len());
    let mut image_lookup: BTreeMap<String, ImageMeta> = BTreeMap::new();

    for (index, image_path) in image_files.iter().enumerate() {
        let rel = rel_string(&layout.images_dir, image_path);
        let (width, height) = read_image_dimensions(image_path)?;
        let image_id = ImageId::new((index + 1) as u64);

        images.push(Image::new(image_id, rel.clone(), width, height));
        image_lookup.insert(
            rel,
            ImageMeta {
                id: image_id,
                width,
                height,
            },
        );
    }

    let categories: Vec<Category> = class_map
        .names
        .iter()
        .enumerate()
        .map(|(i, name)| Category::new((i + 1) as u64, name.clone()))
        .collect();

    let mut label_files = collect_files_with_extensions(&layout.labels_dir, &[LABEL_EXTENSION])?;
    label_files.sort_by_cached_key(|label_path| rel_string(&layout.labels_dir, label_path));

    let mut annotations = Vec::new();
    let mut next_annotation_id: u64 = 1;

    for label_path in label_files {
        let label_rel = label_path.strip_prefix(&layout.labels_dir).map_err(|_| {
            PanlabelError::YoloLayoutInvalid {
                path: label_path.clone(),
                message: format!(
                    "label path '{}' is outside labels dir '{}'",
                    label_path.display(),
                    layout.labels_dir.display()
                ),
            }
        })?;

        let image_path = find_image_for_label(&layout.images_dir, label_rel).ok_or_else(|| {
            PanlabelError::YoloImageNotFound {
                label_path: label_path.clone(),
                expected_stem: rel_string(&layout.labels_dir, &label_path.with_extension("")),
            }
        })?;

        let image_rel = rel_string(&layout.images_dir, &image_path);
        let image_meta =
            image_lookup
                .get(&image_rel)
                .ok_or_else(|| PanlabelError::YoloImageNotFound {
                    label_path: label_path.clone(),
                    expected_stem: rel_string(&layout.labels_dir, &label_path.with_extension("")),
                })?;

        let content = fs::read_to_string(&label_path).map_err(PanlabelError::Io)?;
        for (line_idx, line) in content.lines().enumerate() {
            let line_num = line_idx + 1;
            let Some(parsed) = parse_label_line(line, &label_path, line_num)? else {
                continue;
            };

            if parsed.class_id >= class_map.names.len() {
                return Err(PanlabelError::YoloLabelParse {
                    path: label_path.clone(),
                    line: line_num,
                    message: format!(
                        "class_id {} is out of range for class map with {} class(es)",
                        parsed.class_id,
                        class_map.names.len()
                    ),
                });
            }

            let bbox_norm =
                BBoxXYXY::<Normalized>::from_cxcywh(parsed.cx, parsed.cy, parsed.w, parsed.h);
            let bbox_px = bbox_norm.to_pixel(image_meta.width as f64, image_meta.height as f64);

            annotations.push(Annotation::new(
                AnnotationId::new(next_annotation_id),
                image_meta.id,
                CategoryId::new(parsed.class_id as u64 + 1),
                bbox_px,
            ));
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

/// Write an IR dataset as a YOLO directory.
///
/// Creates `images/` + `labels/` directories and writes `data.yaml` + label
/// files. Image binaries are not copied.
pub fn write_yolo_dir(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    fs::create_dir_all(path).map_err(PanlabelError::Io)?;

    let images_dir = path.join("images");
    let labels_dir = path.join("labels");
    fs::create_dir_all(&images_dir).map_err(PanlabelError::Io)?;
    fs::create_dir_all(&labels_dir).map_err(PanlabelError::Io)?;

    let image_lookup: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();

    let mut categories_sorted: Vec<&Category> = dataset.categories.iter().collect();
    categories_sorted.sort_by_key(|cat| cat.id);

    let category_to_class: BTreeMap<CategoryId, usize> = categories_sorted
        .iter()
        .enumerate()
        .map(|(idx, cat)| (cat.id, idx))
        .collect();

    let mut annotations_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        if !image_lookup.contains_key(&ann.image_id) {
            return Err(PanlabelError::YoloWriteError {
                path: path.to_path_buf(),
                message: format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            });
        }

        if !category_to_class.contains_key(&ann.category_id) {
            return Err(PanlabelError::YoloWriteError {
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
        let label_rel_path = Path::new(&image.file_name).with_extension(LABEL_EXTENSION);
        let label_path = labels_dir.join(&label_rel_path);

        if let Some(parent) = label_path.parent() {
            fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
        }

        let mut label_file = fs::File::create(&label_path).map_err(PanlabelError::Io)?;

        let mut anns = annotations_by_image.remove(&image.id).unwrap_or_default();
        anns.sort_by_key(|ann| ann.id);

        for ann in anns {
            let class_id = *category_to_class
                .get(&ann.category_id)
                .expect("checked category existence above");

            let bbox_norm = ann
                .bbox
                .to_normalized(image.width as f64, image.height as f64);
            let (cx, cy, w, h) = bbox_norm.to_cxcywh();

            writeln!(
                label_file,
                "{} {:.6} {:.6} {:.6} {:.6}",
                class_id, cx, cy, w, h
            )
            .map_err(PanlabelError::Io)?;
        }
    }

    write_data_yaml(path, &categories_sorted)?;

    Ok(())
}

#[derive(Clone, Copy)]
struct ImageMeta {
    id: ImageId,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug)]
struct YoloLayout {
    #[cfg_attr(not(test), allow(dead_code))]
    root: PathBuf,
    images_dir: PathBuf,
    labels_dir: PathBuf,
    class_map_source: YoloClassMapSource,
}

#[derive(Clone, Debug)]
enum YoloClassMapSource {
    DataYaml(PathBuf),
    ClassesTxt(PathBuf),
    Inferred,
}

#[derive(Debug)]
struct YoloClassMap {
    names: Vec<String>,
}

#[derive(Debug, PartialEq)]
struct YoloLabelRow {
    class_id: usize,
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
}

fn discover_layout(input: &Path) -> Result<YoloLayout, PanlabelError> {
    if !input.is_dir() {
        return Err(PanlabelError::YoloLayoutInvalid {
            path: input.to_path_buf(),
            message: "input must be a directory".to_string(),
        });
    }

    let (root, labels_dir) = if input.join("labels").is_dir() {
        (input.to_path_buf(), input.join("labels"))
    } else if is_dir_named(input, "labels") {
        let root = input
            .parent()
            .ok_or_else(|| PanlabelError::YoloLayoutInvalid {
                path: input.to_path_buf(),
                message: "labels directory has no parent directory".to_string(),
            })?
            .to_path_buf();
        (root, input.to_path_buf())
    } else {
        return Err(PanlabelError::YoloLayoutInvalid {
            path: input.to_path_buf(),
            message:
                "expected a YOLO dataset root containing labels/ or a labels/ directory itself"
                    .to_string(),
        });
    };

    let images_dir = root.join("images");
    if !images_dir.is_dir() {
        return Err(PanlabelError::YoloLayoutInvalid {
            path: images_dir,
            message: "missing images/ directory".to_string(),
        });
    }

    if !labels_dir.is_dir() {
        return Err(PanlabelError::YoloLayoutInvalid {
            path: labels_dir,
            message: "missing labels/ directory".to_string(),
        });
    }

    let data_yaml = root.join("data.yaml");
    let classes_txt = root.join("classes.txt");
    let class_map_source = if data_yaml.is_file() {
        YoloClassMapSource::DataYaml(data_yaml)
    } else if classes_txt.is_file() {
        YoloClassMapSource::ClassesTxt(classes_txt)
    } else {
        YoloClassMapSource::Inferred
    };

    Ok(YoloLayout {
        root,
        images_dir,
        labels_dir,
        class_map_source,
    })
}

fn read_class_map(layout: &YoloLayout) -> Result<YoloClassMap, PanlabelError> {
    match &layout.class_map_source {
        YoloClassMapSource::DataYaml(path) => read_data_yaml_names(path),
        YoloClassMapSource::ClassesTxt(path) => read_classes_txt(path),
        YoloClassMapSource::Inferred => infer_class_map(&layout.labels_dir),
    }
}

#[derive(Debug, Deserialize)]
struct DataYaml {
    names: DataYamlNames,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DataYamlNames {
    Sequence(Vec<String>),
    Mapping(BTreeMap<usize, String>),
}

fn read_data_yaml_names(path: &Path) -> Result<YoloClassMap, PanlabelError> {
    let data = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let parsed: DataYaml =
        serde_yaml::from_str(&data).map_err(|source| PanlabelError::YoloDataYamlParse {
            path: path.to_path_buf(),
            source,
        })?;

    let names = match parsed.names {
        DataYamlNames::Sequence(names) => names,
        DataYamlNames::Mapping(mapping) => {
            if mapping.is_empty() {
                Vec::new()
            } else {
                let max_index = *mapping.keys().max().expect("checked non-empty");
                let mut names = vec![String::new(); max_index + 1];
                for (index, name) in mapping {
                    names[index] = name;
                }
                for (index, name) in names.iter_mut().enumerate() {
                    if name.trim().is_empty() {
                        *name = format!("class_{}", index);
                    }
                }
                names
            }
        }
    };

    Ok(YoloClassMap { names })
}

fn write_data_yaml(output_root: &Path, categories: &[&Category]) -> Result<(), PanlabelError> {
    let mut yaml = String::from("names:\n");
    for (idx, category) in categories.iter().enumerate() {
        yaml.push_str(&format!(
            "  {}: {}\n",
            idx,
            yaml_single_quoted(&category.name)
        ));
    }

    let path = output_root.join("data.yaml");
    fs::write(&path, yaml).map_err(PanlabelError::Io)
}

fn yaml_single_quoted(raw: &str) -> String {
    format!("'{}'", raw.replace('\'', "''"))
}

fn read_classes_txt(path: &Path) -> Result<YoloClassMap, PanlabelError> {
    let data = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let mut names = Vec::new();

    for (line_idx, line) in data.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(PanlabelError::YoloClassesTxtInvalid {
                path: path.to_path_buf(),
                message: format!("line {} is empty", line_idx + 1),
            });
        }
        names.push(trimmed.to_string());
    }

    Ok(YoloClassMap { names })
}

fn infer_class_map(labels_dir: &Path) -> Result<YoloClassMap, PanlabelError> {
    let mut label_files = collect_files_with_extensions(labels_dir, &[LABEL_EXTENSION])?;
    label_files.sort_by_cached_key(|label_path| rel_string(labels_dir, label_path));

    let mut class_ids = BTreeSet::new();

    for label_path in label_files {
        let content = fs::read_to_string(&label_path).map_err(PanlabelError::Io)?;
        for (line_idx, line) in content.lines().enumerate() {
            let line_num = line_idx + 1;
            let Some(parsed) = parse_label_line(line, &label_path, line_num)? else {
                continue;
            };
            class_ids.insert(parsed.class_id);
        }
    }

    let max_class_id = class_ids.into_iter().max();
    let names = match max_class_id {
        Some(max_id) => (0..=max_id).map(|id| format!("class_{}", id)).collect(),
        None => Vec::new(),
    };

    Ok(YoloClassMap { names })
}

fn collect_files_with_extensions(
    root: &Path,
    extensions: &[&str],
) -> Result<Vec<PathBuf>, PanlabelError> {
    let mut files = Vec::new();

    for entry in WalkDir::new(root).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::YoloLayoutInvalid {
            path: root.to_path_buf(),
            message: format!("failed while traversing directory: {source}"),
        })?;

        if entry.file_type().is_file() && has_extension(entry.path(), extensions) {
            files.push(entry.path().to_path_buf());
        }
    }

    Ok(files)
}

fn has_extension(path: &Path, allowed: &[&str]) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };

    allowed
        .iter()
        .any(|allowed_ext| ext.eq_ignore_ascii_case(allowed_ext))
}

fn read_image_dimensions(path: &Path) -> Result<(u32, u32), PanlabelError> {
    let size = imagesize::size(path).map_err(|source| PanlabelError::YoloImageDimensionRead {
        path: path.to_path_buf(),
        source,
    })?;

    let width: u32 = size
        .width
        .try_into()
        .map_err(|_| PanlabelError::YoloLayoutInvalid {
            path: path.to_path_buf(),
            message: format!("image width {} does not fit in u32", size.width),
        })?;

    let height: u32 = size
        .height
        .try_into()
        .map_err(|_| PanlabelError::YoloLayoutInvalid {
            path: path.to_path_buf(),
            message: format!("image height {} does not fit in u32", size.height),
        })?;

    Ok((width, height))
}

fn find_image_for_label(images_dir: &Path, label_rel_path: &Path) -> Option<PathBuf> {
    let stem_rel_path = label_rel_path.with_extension("");
    for ext in IMAGE_EXTENSIONS {
        let candidate = images_dir.join(&stem_rel_path).with_extension(ext);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn parse_label_line(
    line: &str,
    file_path: &Path,
    line_num: usize,
) -> Result<Option<YoloLabelRow>, PanlabelError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    // Take at most 6 tokens so pathological inputs do not allocate unbounded memory.
    let tokens: Vec<&str> = trimmed.split_whitespace().take(6).collect();

    if tokens.len() < 5 {
        return Err(PanlabelError::YoloLabelParse {
            path: file_path.to_path_buf(),
            line: line_num,
            message: format!("expected 5 tokens, found {}", tokens.len()),
        });
    }

    if tokens.len() > 5 {
        return Err(PanlabelError::YoloLabelParse {
            path: file_path.to_path_buf(),
            line: line_num,
            message: "segmentation/pose annotations not supported; panlabel currently handles object detection bboxes only"
                .to_string(),
        });
    }

    let class_id = tokens[0]
        .parse::<usize>()
        .map_err(|_| PanlabelError::YoloLabelParse {
            path: file_path.to_path_buf(),
            line: line_num,
            message: format!(
                "invalid class_id '{}'; expected non-negative integer",
                tokens[0]
            ),
        })?;

    let cx = parse_f64_token(tokens[1], "x_center", file_path, line_num)?;
    let cy = parse_f64_token(tokens[2], "y_center", file_path, line_num)?;
    let w = parse_f64_token(tokens[3], "width", file_path, line_num)?;
    let h = parse_f64_token(tokens[4], "height", file_path, line_num)?;

    Ok(Some(YoloLabelRow {
        class_id,
        cx,
        cy,
        w,
        h,
    }))
}

/// Fuzz-only entrypoint for YOLO single-line parsing.
#[cfg(feature = "fuzzing")]
pub fn fuzz_parse_label_line(input: &str) -> Result<(), PanlabelError> {
    let _ = parse_label_line(input, Path::new("<fuzz>"), 1)?;
    Ok(())
}

fn parse_f64_token(
    raw: &str,
    field_name: &str,
    file_path: &Path,
    line_num: usize,
) -> Result<f64, PanlabelError> {
    raw.parse::<f64>()
        .map_err(|_| PanlabelError::YoloLabelParse {
            path: file_path.to_path_buf(),
            line: line_num,
            message: format!("invalid {field_name} '{raw}'; expected floating-point number"),
        })
}

fn is_dir_named(path: &Path, dir_name: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case(dir_name))
        .unwrap_or(false)
}

fn rel_string(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bmp_bytes(width: u32, height: u32) -> Vec<u8> {
        let row_stride = (width * 3).div_ceil(4) * 4;
        let pixel_array_size = row_stride * height;
        let file_size = 54 + pixel_array_size;

        let mut bytes = Vec::with_capacity(file_size as usize);
        bytes.extend_from_slice(b"BM");
        bytes.extend_from_slice(&file_size.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes.extend_from_slice(&54u32.to_le_bytes());

        bytes.extend_from_slice(&40u32.to_le_bytes());
        bytes.extend_from_slice(&(width as i32).to_le_bytes());
        bytes.extend_from_slice(&(height as i32).to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&24u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&pixel_array_size.to_le_bytes());
        bytes.extend_from_slice(&2835u32.to_le_bytes());
        bytes.extend_from_slice(&2835u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        bytes.resize(file_size as usize, 0);
        bytes
    }

    fn write_bmp(path: &Path, width: u32, height: u32) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dirs");
        }
        fs::write(path, bmp_bytes(width, height)).expect("write bmp file");
    }

    fn create_basic_layout(root: &Path) {
        fs::create_dir_all(root.join("images/train")).expect("create images dir");
        fs::create_dir_all(root.join("labels/train")).expect("create labels dir");
    }

    #[test]
    fn parse_label_line_accepts_valid_rows() {
        let parsed = parse_label_line("2 0.5 0.25 0.3 0.1", Path::new("a.txt"), 1)
            .expect("parse should succeed")
            .expect("line should produce a row");

        assert_eq!(
            parsed,
            YoloLabelRow {
                class_id: 2,
                cx: 0.5,
                cy: 0.25,
                w: 0.3,
                h: 0.1,
            }
        );
    }

    #[test]
    fn parse_label_line_skips_empty_rows() {
        let parsed = parse_label_line("   ", Path::new("a.txt"), 2).expect("parse should succeed");
        assert!(parsed.is_none());
    }

    #[test]
    fn parse_label_line_rejects_short_rows() {
        let err = parse_label_line("0 0.1 0.2", Path::new("a.txt"), 3).unwrap_err();
        assert!(matches!(err, PanlabelError::YoloLabelParse { .. }));
    }

    #[test]
    fn parse_label_line_rejects_segmentation_rows() {
        let err = parse_label_line("0 0.1 0.2 0.3 0.4 0.5", Path::new("a.txt"), 4).unwrap_err();
        assert!(matches!(err, PanlabelError::YoloLabelParse { .. }));
    }

    #[test]
    fn discover_layout_accepts_root_or_labels_dir() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_basic_layout(temp.path());

        let root_layout = discover_layout(temp.path()).expect("discover from root");
        assert_eq!(root_layout.root, temp.path());
        assert_eq!(root_layout.images_dir, temp.path().join("images"));
        assert_eq!(root_layout.labels_dir, temp.path().join("labels"));

        let labels_layout =
            discover_layout(&temp.path().join("labels")).expect("discover from labels dir");
        assert_eq!(labels_layout.root, temp.path());
        assert_eq!(labels_layout.images_dir, temp.path().join("images"));
        assert_eq!(labels_layout.labels_dir, temp.path().join("labels"));
    }

    #[test]
    fn class_map_prefers_data_yaml_over_classes_txt() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_basic_layout(temp.path());

        fs::write(
            temp.path().join("data.yaml"),
            "names:\n  0: person\n  1: bicycle\n",
        )
        .expect("write data yaml");
        fs::write(temp.path().join("classes.txt"), "wrong\nvalues\n").expect("write classes");

        let layout = discover_layout(temp.path()).expect("discover layout");
        let class_map = read_class_map(&layout).expect("read class map");
        assert_eq!(class_map.names, vec!["person", "bicycle"]);
    }

    #[test]
    fn class_map_falls_back_to_inferred_names() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_basic_layout(temp.path());

        fs::write(
            temp.path().join("labels/train/example.txt"),
            "2 0.5 0.5 0.5 0.5\n0 0.2 0.2 0.1 0.1\n",
        )
        .expect("write label file");

        let layout = discover_layout(temp.path()).expect("discover layout");
        let class_map = read_class_map(&layout).expect("read class map");
        assert_eq!(class_map.names, vec!["class_0", "class_1", "class_2"]);
    }

    #[test]
    fn find_image_for_label_prefers_extension_order() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_basic_layout(temp.path());
        fs::create_dir_all(temp.path().join("images/train")).expect("create train dir");

        fs::write(temp.path().join("images/train/sample.png"), b"dummy").expect("write png");
        fs::write(temp.path().join("images/train/sample.jpg"), b"dummy").expect("write jpg");

        let found =
            find_image_for_label(&temp.path().join("images"), Path::new("train/sample.txt"))
                .expect("should find image");

        assert!(found.ends_with("sample.jpg"));
    }

    #[test]
    fn read_yolo_dir_assigns_deterministic_ids() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_basic_layout(temp.path());

        write_bmp(&temp.path().join("images/train/b.bmp"), 10, 10);
        write_bmp(&temp.path().join("images/train/a.bmp"), 20, 10);

        fs::write(temp.path().join("data.yaml"), "names:\n  - cat\n  - dog\n")
            .expect("write data yaml");

        fs::write(
            temp.path().join("labels/train/a.txt"),
            "1 0.5 0.5 0.5 0.5\n0 0.25 0.25 0.2 0.2\n",
        )
        .expect("write a labels");
        fs::write(
            temp.path().join("labels/train/b.txt"),
            "0 0.5 0.5 1.0 1.0\n",
        )
        .expect("write b labels");

        let dataset = read_yolo_dir(temp.path()).expect("read yolo dataset");

        assert_eq!(dataset.images.len(), 2);
        assert_eq!(dataset.categories.len(), 2);
        assert_eq!(dataset.annotations.len(), 3);

        assert_eq!(dataset.images[0].id.as_u64(), 1);
        assert_eq!(dataset.images[0].file_name, "train/a.bmp");
        assert_eq!(dataset.images[1].id.as_u64(), 2);
        assert_eq!(dataset.images[1].file_name, "train/b.bmp");

        assert_eq!(dataset.annotations[0].id.as_u64(), 1);
        assert_eq!(dataset.annotations[0].image_id.as_u64(), 1);
        assert_eq!(dataset.annotations[1].id.as_u64(), 2);
        assert_eq!(dataset.annotations[1].image_id.as_u64(), 1);
        assert_eq!(dataset.annotations[2].id.as_u64(), 3);
        assert_eq!(dataset.annotations[2].image_id.as_u64(), 2);

        let first_bbox = &dataset.annotations[0].bbox;
        assert!((first_bbox.xmin() - 5.0).abs() < 1e-6);
        assert!((first_bbox.ymin() - 2.5).abs() < 1e-6);
        assert!((first_bbox.xmax() - 15.0).abs() < 1e-6);
        assert!((first_bbox.ymax() - 7.5).abs() < 1e-6);
    }

    #[test]
    fn read_yolo_dir_fails_when_label_image_is_missing() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_basic_layout(temp.path());

        fs::write(
            temp.path().join("labels/train/missing.txt"),
            "0 0.5 0.5 0.5 0.5\n",
        )
        .expect("write labels");

        let err = read_yolo_dir(temp.path()).unwrap_err();
        assert!(matches!(err, PanlabelError::YoloImageNotFound { .. }));
    }

    #[test]
    fn write_yolo_dir_creates_data_yaml_and_empty_label_files() {
        let temp = tempfile::tempdir().expect("create temp dir");

        let dataset = Dataset {
            images: vec![Image::new(1u64, "train/no_ann.bmp", 8, 8)],
            categories: vec![Category::new(1u64, "cat")],
            annotations: vec![],
            ..Default::default()
        };

        write_yolo_dir(temp.path(), &dataset).expect("write yolo dataset");

        let data_yaml = fs::read_to_string(temp.path().join("data.yaml")).expect("read data.yaml");
        assert!(data_yaml.contains("names:"));
        assert!(data_yaml.contains("0: 'cat'"));

        let label_path = temp.path().join("labels/train/no_ann.txt");
        assert!(label_path.is_file());
        assert!(fs::read_to_string(label_path)
            .expect("read label")
            .is_empty());
        assert!(temp.path().join("images").is_dir());
        assert!(!temp.path().join("images/train/no_ann.bmp").exists());
    }
}
