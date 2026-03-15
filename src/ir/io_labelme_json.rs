//! LabelMe JSON format reader and writer.
//!
//! LabelMe is one of the most widely used open-source annotation tools (~12k
//! GitHub stars). It produces one JSON file per image containing a `shapes`
//! array with rectangles and polygons.
//!
//! # Format Reference
//!
//! ```json
//! {
//!   "version": "5.0.1",
//!   "flags": {},
//!   "shapes": [
//!     {
//!       "label": "cat",
//!       "points": [[100.0, 150.0], [200.0, 250.0]],
//!       "shape_type": "rectangle",
//!       "flags": {}
//!     }
//!   ],
//!   "imagePath": "img001.jpg",
//!   "imageWidth": 640,
//!   "imageHeight": 480,
//!   "imageData": null
//! }
//! ```
//!
//! # Supported shapes
//!
//! - `rectangle` (2 points: top-left, bottom-right)
//! - `polygon` (3+ points: converted to axis-aligned bbox envelope)
//!
//! Other shape types are rejected.
//!
//! # Directory layouts
//!
//! The reader accepts three input modes:
//! - **Single file**: one `.json` file → one-image dataset
//! - **Separate directory**: `annotations/` directory with `.json` files
//! - **Co-located directory**: `.json` files alongside image files
//!
//! The writer always produces the canonical separate layout with `annotations/`
//! and `images/README.txt`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::model::{Annotation, Category, Dataset, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

const IMAGES_README: &str = "This directory is a placeholder. Panlabel does not copy image files during conversion.\nPlace your original images here to complete the LabelMe dataset layout.\n";
const ATTR_IMAGE_PATH: &str = "labelme_image_path";
const ATTR_SHAPE_TYPE: &str = "labelme_shape_type";
const LABELME_VERSION: &str = "5.0.1";

// ============================================================================
// LabelMe Schema Types (internal to this module)
// ============================================================================

/// One LabelMe annotation file.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LabelMeFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    version: Option<String>,

    #[serde(default)]
    flags: serde_json::Value,

    shapes: Vec<LabelMeShape>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    image_path: Option<String>,

    #[serde(default, skip_deserializing)]
    image_data: Option<serde_json::Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    image_height: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    image_width: Option<u32>,
}

/// A single shape annotation in LabelMe.
#[derive(Debug, Serialize, Deserialize)]
struct LabelMeShape {
    label: String,
    points: Vec<[f64; 2]>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    shape_type: Option<String>,

    #[serde(default)]
    flags: serde_json::Value,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    group_id: Option<serde_json::Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

// ============================================================================
// Public API
// ============================================================================

/// Reads a LabelMe dataset from a file or directory.
///
/// Accepts:
/// - A single `.json` file (one-image dataset)
/// - A directory containing `annotations/` with `.json` files (separate layout)
/// - A directory containing `.json` files co-located with images
pub fn read_labelme_json(path: &Path) -> Result<Dataset, PanlabelError> {
    if path.is_file() {
        read_single_file(path)
    } else if path.is_dir() {
        read_directory(path)
    } else {
        Err(PanlabelError::LabelMeLayoutInvalid {
            path: path.to_path_buf(),
            message: "path must be a JSON file or a directory".to_string(),
        })
    }
}

/// Writes an IR dataset as LabelMe JSON.
///
/// - If `path` ends with `.json` and the dataset has exactly 1 image,
///   writes a single LabelMe JSON file.
/// - Otherwise writes a canonical directory layout:
///   `annotations/<stem>.json` + `images/README.txt`.
pub fn write_labelme_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let is_json_ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    if is_json_ext && dataset.images.len() == 1 {
        write_single_file(path, dataset)
    } else if is_json_ext && dataset.images.len() != 1 {
        Err(PanlabelError::LabelMeLayoutInvalid {
            path: path.to_path_buf(),
            message: format!(
                "single-file LabelMe output requires exactly 1 image, got {}; \
                 use a directory path instead",
                dataset.images.len()
            ),
        })
    } else {
        write_directory(path, dataset)
    }
}

/// Parses a single LabelMe JSON string into IR.
pub fn from_labelme_str(json: &str) -> Result<Dataset, PanlabelError> {
    let lm: LabelMeFile =
        serde_json::from_str(json).map_err(|source| PanlabelError::LabelMeJsonParse {
            path: PathBuf::from("<memory>"),
            source,
        })?;
    single_file_to_ir(lm, Path::new("<memory>"))
}

/// Parses a single LabelMe JSON byte slice into IR.
pub fn from_labelme_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    let lm: LabelMeFile =
        serde_json::from_slice(bytes).map_err(|source| PanlabelError::LabelMeJsonParse {
            path: PathBuf::from("<memory>"),
            source,
        })?;
    single_file_to_ir(lm, Path::new("<memory>"))
}

/// Writes a single-image IR dataset to a LabelMe JSON string.
///
/// Fails if the dataset has more than one image.
pub fn to_labelme_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    if dataset.images.len() != 1 {
        return Err(PanlabelError::LabelMeLayoutInvalid {
            path: PathBuf::from("<memory>"),
            message: format!(
                "single-file LabelMe output requires exactly 1 image, got {}",
                dataset.images.len()
            ),
        });
    }

    let lm = ir_to_single_labelme_file(dataset, &dataset.images[0]);
    serde_json::to_string_pretty(&lm).map_err(|source| PanlabelError::LabelMeJsonWrite {
        path: PathBuf::from("<memory>"),
        source,
    })
}

// ============================================================================
// Single-file reading
// ============================================================================

fn read_single_file(path: &Path) -> Result<Dataset, PanlabelError> {
    let contents = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let lm: LabelMeFile =
        serde_json::from_str(&contents).map_err(|source| PanlabelError::LabelMeJsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    single_file_to_ir(lm, path)
}

fn single_file_to_ir(lm: LabelMeFile, source_path: &Path) -> Result<Dataset, PanlabelError> {
    let image_path = lm.image_path.as_deref().unwrap_or("");
    if image_path.is_empty() {
        return Err(PanlabelError::LabelMeLayoutInvalid {
            path: source_path.to_path_buf(),
            message: "missing or empty 'imagePath'".to_string(),
        });
    }

    let width = lm
        .image_width
        .ok_or_else(|| PanlabelError::LabelMeLayoutInvalid {
            path: source_path.to_path_buf(),
            message: "missing 'imageWidth'".to_string(),
        })?;
    let height = lm
        .image_height
        .ok_or_else(|| PanlabelError::LabelMeLayoutInvalid {
            path: source_path.to_path_buf(),
            message: "missing 'imageHeight'".to_string(),
        })?;

    // Derive file_name from imagePath basename
    let file_name = Path::new(image_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(image_path)
        .to_string();

    let mut image = Image::new(ImageId::new(1), &file_name, width, height);
    image
        .attributes
        .insert(ATTR_IMAGE_PATH.to_string(), image_path.to_string());

    // Collect unique labels for categories
    let mut label_set: BTreeSet<String> = BTreeSet::new();
    for shape in &lm.shapes {
        if shape.label.is_empty() {
            return Err(PanlabelError::LabelMeLayoutInvalid {
                path: source_path.to_path_buf(),
                message: "empty shape label".to_string(),
            });
        }
        label_set.insert(shape.label.clone());
    }

    let label_to_cat: BTreeMap<String, CategoryId> = label_set
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), CategoryId::new((i + 1) as u64)))
        .collect();

    let categories: Vec<Category> = label_set
        .iter()
        .enumerate()
        .map(|(i, name)| Category::new((i + 1) as u64, name.clone()))
        .collect();

    let mut annotations = Vec::new();
    for (idx, shape) in lm.shapes.iter().enumerate() {
        let (bbox, is_polygon) = shape_to_bbox(shape, source_path)?;
        let cat_id = label_to_cat[&shape.label];

        let mut ann = Annotation::new(
            AnnotationId::new((idx + 1) as u64),
            ImageId::new(1),
            cat_id,
            bbox,
        );

        if is_polygon {
            ann.attributes
                .insert(ATTR_SHAPE_TYPE.to_string(), "polygon".to_string());
        }

        annotations.push(ann);
    }

    Ok(Dataset {
        images: vec![image],
        categories,
        annotations,
        ..Default::default()
    })
}

// ============================================================================
// Directory reading
// ============================================================================

fn read_directory(path: &Path) -> Result<Dataset, PanlabelError> {
    // Detect layout: separate (annotations/ subdir) or co-located
    let annotations_dir = path.join("annotations");
    let (base_dir, collected) = if annotations_dir.is_dir() {
        (
            annotations_dir.as_path(),
            collect_and_parse_json_files(&annotations_dir)?,
        )
    } else {
        (path, collect_and_parse_json_files(path)?)
    };

    if collected.is_empty() {
        return Err(PanlabelError::LabelMeLayoutInvalid {
            path: path.to_path_buf(),
            message: "no LabelMe JSON files found in directory".to_string(),
        });
    }

    // Derive file_names and build parsed_files
    let mut parsed_files: Vec<(PathBuf, String, LabelMeFile)> = Vec::new();
    for (json_path, lm) in collected {
        let rel = json_path
            .strip_prefix(base_dir)
            .unwrap_or(&json_path)
            .with_extension("");
        let image_path = lm.image_path.as_deref().unwrap_or("");
        let ext = Path::new(image_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("jpg");
        let derived_name = format!("{}.{}", rel.to_string_lossy().replace('\\', "/"), ext);

        parsed_files.push((json_path, derived_name, lm));
    }

    // Sort by derived file_name for deterministic IDs
    parsed_files.sort_by(|a, b| a.1.cmp(&b.1));

    // Check for duplicate derived names
    let mut seen_names: BTreeSet<String> = BTreeSet::new();
    for (json_path, derived_name, _) in &parsed_files {
        if !seen_names.insert(derived_name.clone()) {
            return Err(PanlabelError::LabelMeLayoutInvalid {
                path: json_path.clone(),
                message: format!("duplicate derived image name: '{derived_name}'"),
            });
        }
    }

    // Collect all labels
    let mut label_set: BTreeSet<String> = BTreeSet::new();
    for (json_path, _, lm) in &parsed_files {
        for shape in &lm.shapes {
            if shape.label.is_empty() {
                return Err(PanlabelError::LabelMeLayoutInvalid {
                    path: json_path.clone(),
                    message: "empty shape label".to_string(),
                });
            }
            label_set.insert(shape.label.clone());
        }
    }

    let label_to_cat: BTreeMap<String, CategoryId> = label_set
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), CategoryId::new((i + 1) as u64)))
        .collect();

    let categories: Vec<Category> = label_set
        .iter()
        .enumerate()
        .map(|(i, name)| Category::new((i + 1) as u64, name.clone()))
        .collect();

    let mut images = Vec::new();
    let mut annotations = Vec::new();
    let mut ann_id_counter: u64 = 1;

    for (img_idx, (json_path, derived_name, lm)) in parsed_files.iter().enumerate() {
        let image_id = ImageId::new((img_idx + 1) as u64);

        let width = lm
            .image_width
            .ok_or_else(|| PanlabelError::LabelMeLayoutInvalid {
                path: json_path.clone(),
                message: "missing 'imageWidth'".to_string(),
            })?;
        let height = lm
            .image_height
            .ok_or_else(|| PanlabelError::LabelMeLayoutInvalid {
                path: json_path.clone(),
                message: "missing 'imageHeight'".to_string(),
            })?;

        let mut image = Image::new(image_id, derived_name.clone(), width, height);
        if let Some(image_path) = &lm.image_path {
            image
                .attributes
                .insert(ATTR_IMAGE_PATH.to_string(), image_path.clone());
        }

        images.push(image);

        for shape in &lm.shapes {
            let (bbox, is_polygon) = shape_to_bbox(shape, json_path)?;
            let cat_id = label_to_cat[&shape.label];

            let mut ann =
                Annotation::new(AnnotationId::new(ann_id_counter), image_id, cat_id, bbox);

            if is_polygon {
                ann.attributes
                    .insert(ATTR_SHAPE_TYPE.to_string(), "polygon".to_string());
            }

            annotations.push(ann);
            ann_id_counter += 1;
        }
    }

    Ok(Dataset {
        images,
        categories,
        annotations,
        ..Default::default()
    })
}

/// Collect and parse LabelMe JSON files in a single pass.
///
/// Walks the directory tree, attempts to parse each `.json` file as a
/// `LabelMeFile`, and returns only successfully parsed files. Non-LabelMe
/// JSON files are silently skipped.
fn collect_and_parse_json_files(dir: &Path) -> Result<Vec<(PathBuf, LabelMeFile)>, PanlabelError> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(dir).follow_links(true) {
        let entry = entry.map_err(|source| PanlabelError::LabelMeLayoutInvalid {
            path: dir.to_path_buf(),
            message: format!("failed while traversing directory: {source}"),
        })?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
        {
            if let Ok(contents) = fs::read_to_string(path) {
                if let Ok(lm) = serde_json::from_str::<LabelMeFile>(&contents) {
                    // Verify it's a real LabelMe file (must have shapes field parsed)
                    files.push((path.to_path_buf(), lm));
                }
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

// ============================================================================
// Shape conversion
// ============================================================================

/// Returns (bbox, is_polygon).
fn shape_to_bbox(
    shape: &LabelMeShape,
    source_path: &Path,
) -> Result<(BBoxXYXY<Pixel>, bool), PanlabelError> {
    let shape_type = shape.shape_type.as_deref().unwrap_or("rectangle");

    match shape_type {
        "rectangle" => {
            if shape.points.len() != 2 {
                return Err(PanlabelError::LabelMeLayoutInvalid {
                    path: source_path.to_path_buf(),
                    message: format!(
                        "rectangle shape '{}' must have exactly 2 points, got {}",
                        shape.label,
                        shape.points.len()
                    ),
                });
            }
            let [x1, y1] = shape.points[0];
            let [x2, y2] = shape.points[1];
            let xmin = x1.min(x2);
            let ymin = y1.min(y2);
            let xmax = x1.max(x2);
            let ymax = y1.max(y2);
            Ok((BBoxXYXY::<Pixel>::from_xyxy(xmin, ymin, xmax, ymax), false))
        }
        "polygon" => {
            if shape.points.len() < 3 {
                return Err(PanlabelError::LabelMeLayoutInvalid {
                    path: source_path.to_path_buf(),
                    message: format!(
                        "polygon shape '{}' must have at least 3 points, got {}",
                        shape.label,
                        shape.points.len()
                    ),
                });
            }
            // Compute axis-aligned bounding envelope
            let mut xmin = f64::INFINITY;
            let mut ymin = f64::INFINITY;
            let mut xmax = f64::NEG_INFINITY;
            let mut ymax = f64::NEG_INFINITY;
            for &[x, y] in &shape.points {
                xmin = xmin.min(x);
                ymin = ymin.min(y);
                xmax = xmax.max(x);
                ymax = ymax.max(y);
            }
            Ok((BBoxXYXY::<Pixel>::from_xyxy(xmin, ymin, xmax, ymax), true))
        }
        other => Err(PanlabelError::LabelMeLayoutInvalid {
            path: source_path.to_path_buf(),
            message: format!(
                "unsupported shape type '{}' for shape '{}'; only 'rectangle' and 'polygon' are supported",
                other, shape.label
            ),
        }),
    }
}

// ============================================================================
// Single-file writing
// ============================================================================

fn write_single_file(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let image = &dataset.images[0];
    let lm = ir_to_single_labelme_file(dataset, image);

    let file = fs::File::create(path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &lm).map_err(|source| PanlabelError::LabelMeJsonWrite {
        path: path.to_path_buf(),
        source,
    })
}

fn ir_to_single_labelme_file(dataset: &Dataset, image: &Image) -> LabelMeFile {
    let cat_map: BTreeMap<CategoryId, &str> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();

    let mut anns: Vec<&Annotation> = dataset
        .annotations
        .iter()
        .filter(|a| a.image_id == image.id)
        .collect();
    anns.sort_by_key(|a| a.id);

    let shapes = anns
        .into_iter()
        .map(|ann| {
            let label = cat_map
                .get(&ann.category_id)
                .unwrap_or(&"unknown")
                .to_string();

            LabelMeShape {
                label,
                points: vec![
                    [ann.bbox.xmin(), ann.bbox.ymin()],
                    [ann.bbox.xmax(), ann.bbox.ymax()],
                ],
                shape_type: Some("rectangle".to_string()),
                flags: serde_json::Value::Object(Default::default()),
                group_id: None,
                description: None,
            }
        })
        .collect();

    // Use stored labelme_image_path if available, otherwise file_name
    let image_path = image
        .attributes
        .get(ATTR_IMAGE_PATH)
        .filter(|s| !s.is_empty())
        .cloned()
        .unwrap_or_else(|| image.file_name.clone());

    LabelMeFile {
        version: Some(LABELME_VERSION.to_string()),
        flags: serde_json::Value::Object(Default::default()),
        shapes,
        image_path: Some(image_path),
        image_data: None,
        image_height: Some(image.height),
        image_width: Some(image.width),
    }
}

// ============================================================================
// Directory writing
// ============================================================================

fn write_directory(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    fs::create_dir_all(path).map_err(PanlabelError::Io)?;

    let annotations_dir = path.join("annotations");
    let images_dir = path.join("images");

    fs::create_dir_all(&annotations_dir).map_err(PanlabelError::Io)?;
    fs::create_dir_all(&images_dir).map_err(PanlabelError::Io)?;
    fs::write(images_dir.join("README.txt"), IMAGES_README).map_err(PanlabelError::Io)?;

    let mut images_sorted: Vec<&Image> = dataset.images.iter().collect();
    images_sorted.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    for image in images_sorted {
        let stem = Path::new(&image.file_name).with_extension("");
        let json_rel = format!("{}.json", stem.to_string_lossy().replace('\\', "/"));
        let json_path = annotations_dir.join(&json_rel);

        if let Some(parent) = json_path.parent() {
            fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
        }

        // Reuse the single-file builder, then override imagePath for directory layout
        let mut lm_file = ir_to_single_labelme_file(dataset, image);
        lm_file.image_path = Some(format!("../images/{}", image.file_name));

        let file = fs::File::create(&json_path).map_err(PanlabelError::Io)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &lm_file).map_err(|source| {
            PanlabelError::LabelMeJsonWrite {
                path: json_path.clone(),
                source,
            }
        })?;
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_labelme_json() -> &'static str {
        r#"{
            "version": "5.0.1",
            "flags": {},
            "shapes": [
                {
                    "label": "cat",
                    "points": [[10.0, 20.0], [100.0, 80.0]],
                    "shape_type": "rectangle",
                    "flags": {}
                },
                {
                    "label": "dog",
                    "points": [[50.0, 60.0], [200.0, 150.0], [120.0, 200.0]],
                    "shape_type": "polygon",
                    "flags": {}
                }
            ],
            "imagePath": "img001.jpg",
            "imageHeight": 480,
            "imageWidth": 640,
            "imageData": null
        }"#
    }

    #[test]
    fn parse_single_file_rectangle_and_polygon() {
        let dataset = from_labelme_str(sample_labelme_json()).expect("parse failed");

        assert_eq!(dataset.images.len(), 1);
        assert_eq!(dataset.categories.len(), 2); // cat, dog
        assert_eq!(dataset.annotations.len(), 2);

        let img = &dataset.images[0];
        assert_eq!(img.file_name, "img001.jpg");
        assert_eq!(img.width, 640);
        assert_eq!(img.height, 480);

        // Rectangle: points [[10,20],[100,80]] -> xyxy(10,20,100,80)
        let rect_ann = &dataset.annotations[0];
        assert_eq!(rect_ann.bbox.xmin(), 10.0);
        assert_eq!(rect_ann.bbox.ymin(), 20.0);
        assert_eq!(rect_ann.bbox.xmax(), 100.0);
        assert_eq!(rect_ann.bbox.ymax(), 80.0);
        assert!(rect_ann.attributes.get(ATTR_SHAPE_TYPE).is_none());

        // Polygon: points [[50,60],[200,150],[120,200]] -> envelope xyxy(50,60,200,200)
        let poly_ann = &dataset.annotations[1];
        assert_eq!(poly_ann.bbox.xmin(), 50.0);
        assert_eq!(poly_ann.bbox.ymin(), 60.0);
        assert_eq!(poly_ann.bbox.xmax(), 200.0);
        assert_eq!(poly_ann.bbox.ymax(), 200.0);
        assert_eq!(
            poly_ann.attributes.get(ATTR_SHAPE_TYPE),
            Some(&"polygon".to_string())
        );
    }

    #[test]
    fn missing_image_path_rejected() {
        let json = r#"{
            "shapes": [],
            "imageHeight": 100,
            "imageWidth": 100
        }"#;
        let result = from_labelme_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("imagePath"));
    }

    #[test]
    fn missing_dimensions_rejected() {
        let json = r#"{
            "shapes": [],
            "imagePath": "test.jpg"
        }"#;
        let result = from_labelme_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("imageWidth"));
    }

    #[test]
    fn unsupported_shape_type_rejected() {
        let json = r#"{
            "shapes": [{"label": "test", "points": [[0,0],[1,1],[2,2]], "shape_type": "circle"}],
            "imagePath": "test.jpg",
            "imageHeight": 100,
            "imageWidth": 100
        }"#;
        let result = from_labelme_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("circle"));
    }

    #[test]
    fn rectangle_with_wrong_point_count_rejected() {
        let json = r#"{
            "shapes": [{"label": "test", "points": [[0,0]], "shape_type": "rectangle"}],
            "imagePath": "test.jpg",
            "imageHeight": 100,
            "imageWidth": 100
        }"#;
        let result = from_labelme_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2 points"));
    }

    #[test]
    fn single_file_roundtrip() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "test.jpg", 640, 480)],
            categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
            annotations: vec![
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 80.0),
                ),
                Annotation::new(
                    2u64,
                    1u64,
                    2u64,
                    BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 200.0, 200.0),
                ),
            ],
            ..Default::default()
        };

        let json = to_labelme_string(&dataset).expect("serialize failed");
        let restored = from_labelme_str(&json).expect("parse failed");

        assert_eq!(restored.images.len(), 1);
        assert_eq!(restored.annotations.len(), 2);
        assert_eq!(restored.categories.len(), 2);

        // Check bbox preserved
        assert_eq!(restored.annotations[0].bbox.xmin(), 10.0);
        assert_eq!(restored.annotations[0].bbox.ymin(), 20.0);
        assert_eq!(restored.annotations[0].bbox.xmax(), 100.0);
        assert_eq!(restored.annotations[0].bbox.ymax(), 80.0);
    }

    #[test]
    fn multi_image_single_file_rejected() {
        let dataset = Dataset {
            images: vec![
                Image::new(1u64, "a.jpg", 100, 100),
                Image::new(2u64, "b.jpg", 100, 100),
            ],
            categories: vec![],
            annotations: vec![],
            ..Default::default()
        };

        let result = to_labelme_string(&dataset);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 1 image"));
    }

    #[test]
    fn directory_roundtrip() {
        let dataset = Dataset {
            images: vec![
                Image::new(1u64, "a.jpg", 640, 480),
                Image::new(2u64, "b.png", 800, 600),
            ],
            categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
            annotations: vec![
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 80.0),
                ),
                Annotation::new(
                    2u64,
                    2u64,
                    2u64,
                    BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 200.0, 200.0),
                ),
            ],
            ..Default::default()
        };

        let temp = tempfile::tempdir().expect("create temp dir");
        write_labelme_json(temp.path(), &dataset).expect("write failed");

        // Verify directory structure
        assert!(temp.path().join("annotations").is_dir());
        assert!(temp.path().join("images/README.txt").is_file());
        assert!(temp.path().join("annotations/a.json").is_file());
        assert!(temp.path().join("annotations/b.json").is_file());

        // Read back
        let restored = read_labelme_json(temp.path()).expect("read failed");
        assert_eq!(restored.images.len(), 2);
        assert_eq!(restored.annotations.len(), 2);
        assert_eq!(restored.categories.len(), 2);
    }

    #[test]
    fn missing_shape_type_defaults_to_rectangle() {
        let json = r#"{
            "shapes": [{"label": "box", "points": [[10,20],[30,40]], "flags": {}}],
            "imagePath": "test.jpg",
            "imageHeight": 100,
            "imageWidth": 100
        }"#;
        let dataset = from_labelme_str(json).expect("parse failed");
        assert_eq!(dataset.annotations.len(), 1);
        assert_eq!(dataset.annotations[0].bbox.xmin(), 10.0);
    }
}
