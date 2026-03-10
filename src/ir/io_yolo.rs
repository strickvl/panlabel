//! Ultralytics-style YOLO reader and writer.
//!
//! This module handles directory-based YOLO datasets with `images/` and `labels/`
//! trees, including split-aware layouts (train/val/test) specified in `data.yaml`.
//! The canonical IR representation remains pixel-space XYXY boxes.

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

/// Options for controlling YOLO dataset reading behavior.
#[derive(Clone, Debug, Default)]
pub struct YoloReadOptions {
    /// If set, read only the named split (e.g., "train", "val", "test").
    /// When None, all available splits are merged into a single dataset.
    pub split: Option<String>,
}

/// Read a YOLO dataset directory into IR.
///
/// `path` may be the dataset root containing `images/` + `labels/`, or the
/// `labels/` directory itself. Equivalent to `read_yolo_dir_with_options`
/// with default options (merge all splits).
pub fn read_yolo_dir(path: &Path) -> Result<Dataset, PanlabelError> {
    read_yolo_dir_with_options(path, &YoloReadOptions::default())
}

/// Read a YOLO dataset directory into IR with configurable options.
///
/// Supports both flat layouts (`images/` + `labels/`) and split-aware
/// layouts where `data.yaml` contains `train:`, `val:`, and/or `test:`
/// path keys. In split-aware mode, IR image file names are prefixed with
/// the split name (e.g. `train/img.jpg`) for collision avoidance.
pub fn read_yolo_dir_with_options(
    path: &Path,
    options: &YoloReadOptions,
) -> Result<Dataset, PanlabelError> {
    let source = discover_source(path)?;

    // Select which splits to read
    let selected_splits: Vec<&YoloSplitLayout> = match &options.split {
        Some(requested) => {
            if !source.is_split_aware {
                return Err(PanlabelError::YoloLayoutInvalid {
                    path: path.to_path_buf(),
                    message: format!(
                        "--split '{}' was specified but this is a flat YOLO layout \
                         (no train/val/test paths in data.yaml)",
                        requested
                    ),
                });
            }
            let split = source
                .splits
                .iter()
                .find(|s| s.split_name == *requested)
                .ok_or_else(|| PanlabelError::YoloLayoutInvalid {
                    path: path.to_path_buf(),
                    message: format!(
                        "requested split '{}' not found; available splits: {}",
                        requested,
                        source
                            .splits
                            .iter()
                            .map(|s| s.split_name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                })?;
            vec![split]
        }
        None => source.splits.iter().collect(),
    };

    // Resolve class map
    let labels_dirs: Vec<&Path> = selected_splits
        .iter()
        .map(|s| s.labels_dir.as_path())
        .collect();
    let class_map = resolve_class_map(&source.class_map_source, &labels_dirs)?;

    // Phase 1: Collect all images across selected splits with logical names
    let mut all_image_entries: Vec<(String, PathBuf)> = Vec::new();

    for split in &selected_splits {
        let mut image_files = collect_files_with_extensions(&split.images_dir, &IMAGE_EXTENSIONS)?;
        image_files.sort_by_cached_key(|p| rel_string(&split.images_dir, p));

        for image_path in image_files {
            let rel = rel_string(&split.images_dir, &image_path);
            let name = logical_name(source.is_split_aware, &split.split_name, rel);
            all_image_entries.push((name, image_path));
        }
    }

    // Sort globally by logical name for deterministic IDs
    all_image_entries.sort_by(|a, b| a.0.cmp(&b.0));

    // Build images and lookup
    let mut images = Vec::with_capacity(all_image_entries.len());
    let mut image_lookup: BTreeMap<String, ImageMeta> = BTreeMap::new();

    for (index, (logical_name, image_path)) in all_image_entries.iter().enumerate() {
        let (width, height) = read_image_dimensions(image_path)?;
        let image_id = ImageId::new((index + 1) as u64);

        images.push(Image::new(image_id, logical_name.clone(), width, height));
        image_lookup.insert(
            logical_name.clone(),
            ImageMeta {
                id: image_id,
                width,
                height,
            },
        );
    }

    // Build categories
    let categories: Vec<Category> = class_map
        .names
        .iter()
        .enumerate()
        .map(|(i, name)| Category::new((i + 1) as u64, name.clone()))
        .collect();

    // Phase 2: Collect all labels across selected splits
    let mut all_label_entries: Vec<(String, PathBuf, usize)> = Vec::new();

    for (split_idx, split) in selected_splits.iter().enumerate() {
        let mut label_files = collect_files_with_extensions(&split.labels_dir, &[LABEL_EXTENSION])?;
        label_files.sort_by_cached_key(|p| rel_string(&split.labels_dir, p));

        for label_path in label_files {
            let rel = rel_string(&split.labels_dir, &label_path);
            let name = logical_name(source.is_split_aware, &split.split_name, rel);
            all_label_entries.push((name, label_path, split_idx));
        }
    }

    // Sort globally by logical label path for deterministic annotation IDs
    all_label_entries.sort_by(|a, b| a.0.cmp(&b.0));

    // Parse labels and create annotations
    let mut annotations = Vec::new();
    let mut next_annotation_id: u64 = 1;

    for (_, label_path, split_idx) in &all_label_entries {
        let split = &selected_splits[*split_idx];

        let label_rel = label_path.strip_prefix(&split.labels_dir).map_err(|_| {
            PanlabelError::YoloLayoutInvalid {
                path: label_path.clone(),
                message: format!(
                    "label path '{}' is outside labels dir '{}'",
                    label_path.display(),
                    split.labels_dir.display()
                ),
            }
        })?;

        // Find image in same split
        let image_path = find_image_for_label(&split.images_dir, label_rel).ok_or_else(|| {
            PanlabelError::YoloImageNotFound {
                label_path: label_path.clone(),
                expected_stem: rel_string(&split.labels_dir, &label_path.with_extension("")),
            }
        })?;

        // Compute logical image name (same key used in image_lookup)
        let image_rel = rel_string(&split.images_dir, &image_path);
        let logical_image_name = logical_name(source.is_split_aware, &split.split_name, image_rel);

        let image_meta = image_lookup.get(&logical_image_name).ok_or_else(|| {
            PanlabelError::YoloImageNotFound {
                label_path: label_path.clone(),
                expected_stem: rel_string(&split.labels_dir, &label_path.with_extension("")),
            }
        })?;

        let content = fs::read_to_string(label_path).map_err(PanlabelError::Io)?;
        for (line_idx, line) in content.lines().enumerate() {
            let line_num = line_idx + 1;
            let Some(parsed) = parse_label_line(line, label_path, line_num)? else {
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

    // Build dataset info with provenance for split-aware mode
    let mut info = DatasetInfo::default();
    if source.is_split_aware {
        let all_split_names: Vec<&str> = source
            .splits
            .iter()
            .map(|s| s.split_name.as_str())
            .collect();
        let read_split_names: Vec<&str> = selected_splits
            .iter()
            .map(|s| s.split_name.as_str())
            .collect();
        info.attributes
            .insert("yolo_layout_mode".to_string(), "split_aware".to_string());
        info.attributes
            .insert("yolo_splits_found".to_string(), all_split_names.join(","));
        info.attributes
            .insert("yolo_splits_read".to_string(), read_split_names.join(","));
    }

    Ok(Dataset {
        info,
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

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ImageMeta {
    id: ImageId,
    width: u32,
    height: u32,
}

/// Discovery result describing one or more YOLO split layouts.
#[derive(Clone, Debug)]
struct YoloSource {
    is_split_aware: bool,
    splits: Vec<YoloSplitLayout>,
    class_map_source: YoloClassMapSource,
}

/// A single images+labels pair, optionally associated with a named split.
#[derive(Clone, Debug)]
struct YoloSplitLayout {
    /// Empty string for flat layouts, otherwise "train"/"val"/"test".
    split_name: String,
    images_dir: PathBuf,
    labels_dir: PathBuf,
}

#[derive(Clone, Debug)]
enum YoloClassMapSource {
    /// Names already parsed from data.yaml — avoids re-reading the file.
    DataYaml(Vec<String>),
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

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Expanded data.yaml for split-aware discovery.
///
/// All fields are optional so that any valid YAML mapping can be parsed
/// (real data.yaml files contain many extra fields like `nc`, `download`, etc.).
#[derive(Debug, Deserialize)]
struct DataYamlFull {
    names: Option<DataYamlNames>,
    train: Option<String>,
    val: Option<String>,
    test: Option<String>,
    path: Option<String>,
}

/// Discover the YOLO source layout: flat or split-aware.
fn discover_source(input: &Path) -> Result<YoloSource, PanlabelError> {
    if !input.is_dir() {
        return Err(PanlabelError::YoloLayoutInvalid {
            path: input.to_path_buf(),
            message: "input must be a directory".to_string(),
        });
    }

    // Determine root directory
    let root = if is_dir_named(input, "labels") {
        input
            .parent()
            .ok_or_else(|| PanlabelError::YoloLayoutInvalid {
                path: input.to_path_buf(),
                message: "labels directory has no parent directory".to_string(),
            })?
            .to_path_buf()
    } else {
        input.to_path_buf()
    };

    // Try to parse data.yaml for split information
    let data_yaml_path = root.join("data.yaml");
    let mut parsed_full: Option<DataYamlFull> = if data_yaml_path.is_file() {
        let data = fs::read_to_string(&data_yaml_path).map_err(PanlabelError::Io)?;
        Some(
            serde_yaml::from_str(&data).map_err(|source| PanlabelError::YoloDataYamlParse {
                path: data_yaml_path.clone(),
                source,
            })?,
        )
    } else {
        None
    };

    // Check for split-aware mode (take ownership of parsed_full to move names out)
    let has_splits = parsed_full
        .as_ref()
        .map(|p| p.train.is_some() || p.val.is_some() || p.test.is_some())
        .unwrap_or(false);

    if has_splits {
        let parsed = parsed_full.take().expect("checked is_some via has_splits");

        // Resolve base path for split-relative paths
        let base_path = match &parsed.path {
            Some(p) => {
                let p = Path::new(p);
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    root.join(p)
                }
            }
            None => root.clone(),
        };

        // Determine class map source (eagerly resolve names to avoid re-reading)
        let classes_txt = root.join("classes.txt");
        let class_map_source = if let Some(names) = parsed.names {
            YoloClassMapSource::DataYaml(resolve_data_yaml_names(names))
        } else if classes_txt.is_file() {
            YoloClassMapSource::ClassesTxt(classes_txt)
        } else {
            YoloClassMapSource::Inferred
        };

        // Resolve each split
        let mut splits = Vec::new();
        for (name, raw_path) in [
            ("train", &parsed.train),
            ("val", &parsed.val),
            ("test", &parsed.test),
        ] {
            if let Some(raw) = raw_path {
                let split_layout = resolve_split_layout(&base_path, name, raw)?;
                splits.push(split_layout);
            }
        }

        return Ok(YoloSource {
            is_split_aware: true,
            splits,
            class_map_source,
        });
    }

    // Flat layout fallback
    let images_dir = root.join("images");
    let labels_dir = if is_dir_named(input, "labels") {
        input.to_path_buf()
    } else {
        root.join("labels")
    };

    // Validate flat layout
    if !labels_dir.is_dir() {
        return Err(PanlabelError::YoloLayoutInvalid {
            path: input.to_path_buf(),
            message:
                "expected a YOLO dataset root containing labels/ or a labels/ directory itself"
                    .to_string(),
        });
    }

    if !images_dir.is_dir() {
        return Err(PanlabelError::YoloLayoutInvalid {
            path: images_dir,
            message: "missing images/ directory".to_string(),
        });
    }

    // Determine class map source for flat mode (eagerly resolve names)
    let classes_txt = root.join("classes.txt");
    let class_map_source = if let Some(names) = parsed_full.and_then(|p| p.names) {
        YoloClassMapSource::DataYaml(resolve_data_yaml_names(names))
    } else if classes_txt.is_file() {
        YoloClassMapSource::ClassesTxt(classes_txt)
    } else {
        YoloClassMapSource::Inferred
    };

    Ok(YoloSource {
        is_split_aware: false,
        splits: vec![YoloSplitLayout {
            split_name: String::new(),
            images_dir,
            labels_dir,
        }],
        class_map_source,
    })
}

/// Resolve a single split from data.yaml into concrete images + labels dirs.
///
/// Supports three common path styles:
/// - Pattern A: `train: images/train` — swap `images` component to `labels`
/// - Pattern B: `train: train/images` — swap `images` component to `labels`
/// - Pattern C: `train: train` — expect `images/` + `labels/` inside
fn resolve_split_layout(
    base_path: &Path,
    split_name: &str,
    raw_path: &str,
) -> Result<YoloSplitLayout, PanlabelError> {
    let resolved = if Path::new(raw_path).is_absolute() {
        PathBuf::from(raw_path)
    } else {
        base_path.join(raw_path)
    };

    // Pattern C: resolved is a split root containing images/ + labels/
    if resolved.join("images").is_dir() && resolved.join("labels").is_dir() {
        return Ok(YoloSplitLayout {
            split_name: split_name.to_string(),
            images_dir: resolved.join("images"),
            labels_dir: resolved.join("labels"),
        });
    }

    // Pattern A/B: resolved is an images directory; derive labels by swapping
    // the "images" component in raw_path with "labels"
    if resolved.is_dir() {
        if let Some(labels_dir) = derive_labels_path(base_path, raw_path) {
            if labels_dir.is_dir() {
                return Ok(YoloSplitLayout {
                    split_name: split_name.to_string(),
                    images_dir: resolved,
                    labels_dir,
                });
            }
        }
    }

    Err(PanlabelError::YoloLayoutInvalid {
        path: resolved,
        message: format!(
            "split '{}' path '{}' could not be resolved to a valid YOLO layout. \
             Expected: a directory with images/ + labels/ inside, or \
             a path containing an 'images' component with a corresponding 'labels' sibling",
            split_name, raw_path
        ),
    })
}

/// Derive the labels directory path by replacing the rightmost "images"
/// component in `raw_path` with "labels", then joining onto `base_path`.
fn derive_labels_path(base_path: &Path, raw_path: &str) -> Option<PathBuf> {
    let normalized = raw_path.replace('\\', "/");
    let parts: Vec<&str> = normalized.split('/').collect();

    for i in (0..parts.len()).rev() {
        if parts[i].eq_ignore_ascii_case("images") {
            let mut new_parts = parts.clone();
            new_parts[i] = "labels";
            let labels_raw = new_parts.join("/");
            return Some(base_path.join(labels_raw));
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Class map resolution
// ---------------------------------------------------------------------------

/// Resolve the class map from a source, scanning multiple labels dirs if inferred.
fn resolve_class_map(
    class_map_source: &YoloClassMapSource,
    labels_dirs: &[&Path],
) -> Result<YoloClassMap, PanlabelError> {
    match class_map_source {
        YoloClassMapSource::DataYaml(names) => Ok(YoloClassMap {
            names: names.clone(),
        }),
        YoloClassMapSource::ClassesTxt(path) => read_classes_txt(path),
        YoloClassMapSource::Inferred => infer_class_map_from_dirs(labels_dirs),
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum DataYamlNames {
    Sequence(Vec<String>),
    Mapping(BTreeMap<usize, String>),
}

/// Convert a `DataYamlNames` enum into a flat `Vec<String>`.
fn resolve_data_yaml_names(raw: DataYamlNames) -> Vec<String> {
    match raw {
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
    }
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

/// Infer the class map by scanning multiple labels directories.
fn infer_class_map_from_dirs(labels_dirs: &[&Path]) -> Result<YoloClassMap, PanlabelError> {
    let mut class_ids = BTreeSet::new();

    for labels_dir in labels_dirs {
        let label_files = collect_files_with_extensions(labels_dir, &[LABEL_EXTENSION])?;

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
    }

    let max_class_id = class_ids.into_iter().max();
    let names = match max_class_id {
        Some(max_id) => (0..=max_id).map(|id| format!("class_{}", id)).collect(),
        None => Vec::new(),
    };

    Ok(YoloClassMap { names })
}

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Label parsing
// ---------------------------------------------------------------------------

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

/// Prefix a relative path with the split name when in split-aware mode.
fn logical_name(is_split_aware: bool, split_name: &str, rel: String) -> String {
    if is_split_aware {
        format!("{}/{}", split_name, rel)
    } else {
        rel
    }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

        let root_source = discover_source(temp.path()).expect("discover from root");
        assert!(!root_source.is_split_aware);
        assert_eq!(root_source.splits.len(), 1);
        assert_eq!(root_source.splits[0].images_dir, temp.path().join("images"));
        assert_eq!(root_source.splits[0].labels_dir, temp.path().join("labels"));

        let labels_source =
            discover_source(&temp.path().join("labels")).expect("discover from labels dir");
        assert!(!labels_source.is_split_aware);
        assert_eq!(
            labels_source.splits[0].images_dir,
            temp.path().join("images")
        );
        assert_eq!(
            labels_source.splits[0].labels_dir,
            temp.path().join("labels")
        );
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

        let source = discover_source(temp.path()).expect("discover source");
        let labels_dirs: Vec<&Path> = source
            .splits
            .iter()
            .map(|s| s.labels_dir.as_path())
            .collect();
        let class_map =
            resolve_class_map(&source.class_map_source, &labels_dirs).expect("read class map");
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

        let source = discover_source(temp.path()).expect("discover source");
        let labels_dirs: Vec<&Path> = source
            .splits
            .iter()
            .map(|s| s.labels_dir.as_path())
            .collect();
        let class_map =
            resolve_class_map(&source.class_map_source, &labels_dirs).expect("read class map");
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

    // -------------------------------------------------------------------
    // Split-aware tests
    // -------------------------------------------------------------------

    /// Helper to create a split-aware YOLO layout (images/X + labels/X style).
    fn create_split_layout_images_style(root: &Path, splits: &[&str]) {
        for split in splits {
            fs::create_dir_all(root.join(format!("images/{}", split)))
                .expect("create images split dir");
            fs::create_dir_all(root.join(format!("labels/{}", split)))
                .expect("create labels split dir");
        }
    }

    #[test]
    fn discover_source_detects_split_aware_layout() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_split_layout_images_style(temp.path(), &["train", "val"]);

        fs::write(
            temp.path().join("data.yaml"),
            "names:\n  - cat\n  - dog\ntrain: images/train\nval: images/val\n",
        )
        .expect("write data yaml");

        let source = discover_source(temp.path()).expect("discover source");
        assert!(source.is_split_aware);
        assert_eq!(source.splits.len(), 2);
        assert_eq!(source.splits[0].split_name, "train");
        assert_eq!(source.splits[1].split_name, "val");
    }

    #[test]
    fn discover_source_flat_fallback_when_no_split_keys() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_basic_layout(temp.path());

        fs::write(temp.path().join("data.yaml"), "names:\n  - cat\n").expect("write data yaml");

        let source = discover_source(temp.path()).expect("discover source");
        assert!(!source.is_split_aware);
        assert_eq!(source.splits.len(), 1);
        assert_eq!(source.splits[0].split_name, "");
    }

    #[test]
    fn split_aware_read_merges_all_splits() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_split_layout_images_style(temp.path(), &["train", "val"]);

        write_bmp(&temp.path().join("images/train/a.bmp"), 10, 10);
        write_bmp(&temp.path().join("images/val/b.bmp"), 20, 10);

        fs::write(
            temp.path().join("data.yaml"),
            "names:\n  - cat\ntrain: images/train\nval: images/val\n",
        )
        .expect("write data yaml");

        fs::write(
            temp.path().join("labels/train/a.txt"),
            "0 0.5 0.5 0.5 0.5\n",
        )
        .expect("write train label");
        fs::write(temp.path().join("labels/val/b.txt"), "0 0.3 0.3 0.2 0.2\n")
            .expect("write val label");

        let dataset = read_yolo_dir(temp.path()).expect("read split-aware dataset");

        assert_eq!(dataset.images.len(), 2);
        assert_eq!(dataset.annotations.len(), 2);

        // Images should be split-prefixed and sorted lexicographically
        assert_eq!(dataset.images[0].file_name, "train/a.bmp");
        assert_eq!(dataset.images[1].file_name, "val/b.bmp");

        // Provenance attributes should record split info
        assert_eq!(
            dataset.info.attributes.get("yolo_layout_mode"),
            Some(&"split_aware".to_string())
        );
        assert_eq!(
            dataset.info.attributes.get("yolo_splits_found"),
            Some(&"train,val".to_string())
        );
        assert_eq!(
            dataset.info.attributes.get("yolo_splits_read"),
            Some(&"train,val".to_string())
        );
    }

    #[test]
    fn split_aware_read_selects_single_split() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_split_layout_images_style(temp.path(), &["train", "val"]);

        write_bmp(&temp.path().join("images/train/a.bmp"), 10, 10);
        write_bmp(&temp.path().join("images/val/b.bmp"), 20, 10);

        fs::write(
            temp.path().join("data.yaml"),
            "names:\n  - cat\ntrain: images/train\nval: images/val\n",
        )
        .expect("write data yaml");

        fs::write(
            temp.path().join("labels/train/a.txt"),
            "0 0.5 0.5 0.5 0.5\n",
        )
        .expect("write train label");
        fs::write(temp.path().join("labels/val/b.txt"), "0 0.3 0.3 0.2 0.2\n")
            .expect("write val label");

        let options = YoloReadOptions {
            split: Some("val".to_string()),
        };
        let dataset = read_yolo_dir_with_options(temp.path(), &options).expect("read single split");

        assert_eq!(dataset.images.len(), 1);
        assert_eq!(dataset.annotations.len(), 1);

        // Even with --split, file names are split-prefixed (option A)
        assert_eq!(dataset.images[0].file_name, "val/b.bmp");

        // Provenance records which splits were found vs read
        assert_eq!(
            dataset.info.attributes.get("yolo_splits_found"),
            Some(&"train,val".to_string())
        );
        assert_eq!(
            dataset.info.attributes.get("yolo_splits_read"),
            Some(&"val".to_string())
        );
    }

    #[test]
    fn split_aware_read_errors_on_missing_split() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_split_layout_images_style(temp.path(), &["train"]);

        fs::write(
            temp.path().join("data.yaml"),
            "names:\n  - cat\ntrain: images/train\n",
        )
        .expect("write data yaml");

        let options = YoloReadOptions {
            split: Some("val".to_string()),
        };
        let err = read_yolo_dir_with_options(temp.path(), &options).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("val"),
            "error should mention requested split: {msg}"
        );
        assert!(
            msg.contains("train"),
            "error should list available splits: {msg}"
        );
    }

    #[test]
    fn split_flag_errors_on_flat_layout() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_basic_layout(temp.path());

        fs::write(temp.path().join("data.yaml"), "names:\n  - cat\n").expect("write data yaml");

        let options = YoloReadOptions {
            split: Some("train".to_string()),
        };
        let err = read_yolo_dir_with_options(temp.path(), &options).unwrap_err();
        assert!(err.to_string().contains("flat YOLO layout"));
    }

    #[test]
    fn split_aware_supports_split_root_style() {
        // Pattern C: train: train -> expects train/images + train/labels
        let temp = tempfile::tempdir().expect("create temp dir");
        fs::create_dir_all(temp.path().join("train/images")).expect("create dir");
        fs::create_dir_all(temp.path().join("train/labels")).expect("create dir");

        write_bmp(&temp.path().join("train/images/a.bmp"), 10, 10);

        fs::write(
            temp.path().join("data.yaml"),
            "names:\n  - cat\ntrain: train\n",
        )
        .expect("write data yaml");

        fs::write(
            temp.path().join("train/labels/a.txt"),
            "0 0.5 0.5 0.5 0.5\n",
        )
        .expect("write label");

        let dataset = read_yolo_dir(temp.path()).expect("read split-root style");

        assert_eq!(dataset.images.len(), 1);
        assert_eq!(dataset.images[0].file_name, "train/a.bmp");
    }

    #[test]
    fn split_aware_supports_data_yaml_path_key() {
        let temp = tempfile::tempdir().expect("create temp dir");
        let data_root = temp.path().join("datasets/mydata");
        fs::create_dir_all(data_root.join("images/train")).expect("create dir");
        fs::create_dir_all(data_root.join("labels/train")).expect("create dir");

        write_bmp(&data_root.join("images/train/a.bmp"), 10, 10);

        fs::write(data_root.join("labels/train/a.txt"), "0 0.5 0.5 0.5 0.5\n")
            .expect("write label");

        // data.yaml with path: key pointing to the data root
        fs::write(
            temp.path().join("data.yaml"),
            format!(
                "path: {}\nnames:\n  - cat\ntrain: images/train\n",
                data_root.display()
            ),
        )
        .expect("write data yaml");

        // Create the minimal YOLO structure at temp root so discover_source
        // doesn't fail on flat layout validation (it won't reach flat path
        // because data.yaml has split keys, but we need data.yaml to be
        // at root)
        // Actually, data.yaml is at temp.path(), and discover_source reads
        // root/data.yaml. The root is temp.path(). data.yaml has train: key,
        // so we enter split-aware mode. base_path = data_root (absolute path).
        // resolve_split_layout(data_root, "train", "images/train") ->
        //   resolved = data_root/images/train -> check labels sibling

        let source = discover_source(temp.path()).expect("discover with path key");
        assert!(source.is_split_aware);
        assert_eq!(source.splits[0].images_dir, data_root.join("images/train"));
        assert_eq!(source.splits[0].labels_dir, data_root.join("labels/train"));
    }

    #[test]
    fn split_aware_class_map_inferred_across_splits() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_split_layout_images_style(temp.path(), &["train", "val"]);

        write_bmp(&temp.path().join("images/train/a.bmp"), 10, 10);
        write_bmp(&temp.path().join("images/val/b.bmp"), 10, 10);

        // data.yaml with splits but NO names -> inferred class map
        fs::write(
            temp.path().join("data.yaml"),
            "train: images/train\nval: images/val\n",
        )
        .expect("write data yaml");

        // Class 0 in train, class 2 in val -> inferred names should cover 0..2
        fs::write(
            temp.path().join("labels/train/a.txt"),
            "0 0.5 0.5 0.5 0.5\n",
        )
        .expect("write train label");
        fs::write(temp.path().join("labels/val/b.txt"), "2 0.3 0.3 0.2 0.2\n")
            .expect("write val label");

        let dataset = read_yolo_dir(temp.path()).expect("read with inferred class map");
        assert_eq!(dataset.categories.len(), 3);
        assert_eq!(dataset.categories[0].name, "class_0");
        assert_eq!(dataset.categories[1].name, "class_1");
        assert_eq!(dataset.categories[2].name, "class_2");
    }

    #[test]
    fn flat_read_has_no_split_provenance() {
        let temp = tempfile::tempdir().expect("create temp dir");
        create_basic_layout(temp.path());

        write_bmp(&temp.path().join("images/train/a.bmp"), 10, 10);
        fs::write(temp.path().join("data.yaml"), "names:\n  - cat\n").expect("write data yaml");
        fs::write(
            temp.path().join("labels/train/a.txt"),
            "0 0.5 0.5 0.5 0.5\n",
        )
        .expect("write label");

        let dataset = read_yolo_dir(temp.path()).expect("read flat");
        assert!(dataset.info.attributes.get("yolo_layout_mode").is_none());
        assert!(dataset.info.attributes.get("yolo_splits_found").is_none());
    }
}
