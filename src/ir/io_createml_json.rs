//! CreateML JSON format reader and writer.
//!
//! Apple's CreateML annotation format uses a flat JSON array where each element
//! represents one image with its annotations. Bounding boxes use center-based
//! absolute pixel coordinates: `{x, y, width, height}` where `(x, y)` is the
//! center of the box.
//!
//! # Format Reference
//!
//! ```json
//! [
//!   {
//!     "image": "img001.jpg",
//!     "annotations": [
//!       {
//!         "label": "cat",
//!         "coordinates": { "x": 150.0, "y": 200.0, "width": 80.0, "height": 60.0 }
//!       }
//!     ]
//!   }
//! ]
//! ```
//!
//! Image dimensions are not stored in the JSON — the reader resolves them from
//! local image files relative to the JSON file's parent directory.
//!
//! # Deterministic Output
//!
//! The writer produces deterministic output: image rows are sorted by filename,
//! annotations within each image are sorted by annotation ID.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::model::{Annotation, Category, Dataset, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

// ============================================================================
// CreateML Schema Types (internal to this module)
// ============================================================================

/// One image row in a CreateML JSON array.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CreateMlImageRow {
    pub(crate) image: String,
    #[serde(default)]
    pub(crate) annotations: Vec<CreateMlAnnotation>,
}

/// One annotation within a CreateML image row.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CreateMlAnnotation {
    pub(crate) label: String,
    pub(crate) coordinates: CreateMlCoordinates,
}

/// Center-based absolute pixel coordinates.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CreateMlCoordinates {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) width: f64,
    pub(crate) height: f64,
}

// ============================================================================
// Public API
// ============================================================================

/// Reads a dataset from a CreateML JSON file.
///
/// Image dimensions are resolved by probing local image files relative to
/// the JSON file's parent directory.
pub fn read_createml_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);

    let rows: Vec<CreateMlImageRow> =
        serde_json::from_reader(reader).map_err(|source| PanlabelError::CreateMlJsonParse {
            path: path.to_path_buf(),
            source,
        })?;

    createml_rows_to_ir(rows, base_dir, path)
}

/// Writes a dataset to a CreateML JSON file.
pub fn write_createml_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);

    let rows = ir_to_createml_rows(dataset);

    serde_json::to_writer_pretty(writer, &rows).map_err(|source| PanlabelError::CreateMlJsonWrite {
        path: path.to_path_buf(),
        source,
    })
}

/// Parses CreateML JSON from a byte slice (schema-only, no image resolution).
///
/// Fuzz-only entrypoint: exercises JSON/schema parsing without requiring
/// image files on disk.
#[cfg(feature = "fuzzing")]
pub fn parse_createml_slice(bytes: &[u8]) -> Result<(), serde_json::Error> {
    let _rows: Vec<CreateMlImageRow> = serde_json::from_slice(bytes)?;
    Ok(())
}

/// Reads a dataset from a CreateML JSON string, resolving images from `base_dir`.
pub fn from_createml_str_with_base_dir(
    json: &str,
    base_dir: &Path,
) -> Result<Dataset, PanlabelError> {
    let rows: Vec<CreateMlImageRow> =
        serde_json::from_str(json).map_err(|source| PanlabelError::CreateMlJsonParse {
            path: base_dir.to_path_buf(),
            source,
        })?;

    createml_rows_to_ir(rows, base_dir, base_dir)
}

/// Writes a dataset to a CreateML JSON string.
pub fn to_createml_string(dataset: &Dataset) -> Result<String, serde_json::Error> {
    let rows = ir_to_createml_rows(dataset);
    serde_json::to_string_pretty(&rows)
}

// ============================================================================
// Conversion: CreateML -> IR
// ============================================================================

fn createml_rows_to_ir(
    rows: Vec<CreateMlImageRow>,
    base_dir: &Path,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    // Validate no duplicate image refs
    let mut seen_images: BTreeSet<String> = BTreeSet::new();
    for row in &rows {
        if row.image.is_empty() {
            return Err(PanlabelError::CreateMlJsonInvalid {
                path: source_path.to_path_buf(),
                message: "empty 'image' field".to_string(),
            });
        }
        if !seen_images.insert(row.image.clone()) {
            return Err(PanlabelError::CreateMlJsonInvalid {
                path: source_path.to_path_buf(),
                message: format!("duplicate image entry: '{}'", row.image),
            });
        }
    }

    // Collect all unique labels, sorted for deterministic category IDs
    let mut label_set: BTreeSet<String> = BTreeSet::new();
    for row in &rows {
        for ann in &row.annotations {
            if ann.label.is_empty() {
                return Err(PanlabelError::CreateMlJsonInvalid {
                    path: source_path.to_path_buf(),
                    message: format!("empty annotation label in image '{}'", row.image),
                });
            }
            label_set.insert(ann.label.clone());
        }
    }

    // Build category map: label -> CategoryId (1-based, sorted)
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

    // Sort rows by image name for deterministic image IDs
    let mut sorted_rows = rows;
    sorted_rows.sort_by(|a, b| a.image.cmp(&b.image));

    let mut images = Vec::new();
    let mut annotations = Vec::new();
    let mut ann_id_counter: u64 = 1;

    for (img_idx, row) in sorted_rows.iter().enumerate() {
        let image_id = ImageId::new((img_idx + 1) as u64);

        // Resolve image dimensions from disk
        let (width, height) = resolve_image_dimensions(base_dir, &row.image, source_path)?;

        let file_name = row.image.replace('\\', "/");

        images.push(Image::new(image_id, file_name, width, height));

        for ann_row in &row.annotations {
            let cat_id = label_to_cat[&ann_row.label];
            let bbox = BBoxXYXY::<Pixel>::from_cxcywh(
                ann_row.coordinates.x,
                ann_row.coordinates.y,
                ann_row.coordinates.width,
                ann_row.coordinates.height,
            );

            annotations.push(Annotation::new(
                AnnotationId::new(ann_id_counter),
                image_id,
                cat_id,
                bbox,
            ));
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

/// Resolve image dimensions by probing the filesystem.
///
/// Precedence: `base_dir/<image>` then `base_dir/images/<image>`.
fn resolve_image_dimensions(
    base_dir: &Path,
    image_ref: &str,
    source_path: &Path,
) -> Result<(u32, u32), PanlabelError> {
    // Reject absolute paths and path traversal
    if image_ref.starts_with('/') || image_ref.starts_with('\\') || image_ref.contains("..") {
        return Err(PanlabelError::CreateMlJsonInvalid {
            path: source_path.to_path_buf(),
            message: format!(
                "image reference '{}' must be a relative path without '..'",
                image_ref
            ),
        });
    }

    let candidate1 = base_dir.join(image_ref);
    let candidate2 = base_dir.join("images").join(image_ref);

    // Try candidates directly (no TOCTOU existence check)
    if let Ok(size) = imagesize::size(&candidate1) {
        return Ok((size.width as u32, size.height as u32));
    }
    if let Ok(size) = imagesize::size(&candidate2) {
        return Ok((size.width as u32, size.height as u32));
    }

    Err(PanlabelError::CreateMlImageNotFound {
        path: source_path.to_path_buf(),
        image_ref: image_ref.to_string(),
    })
}

// ============================================================================
// Conversion: IR -> CreateML
// ============================================================================

fn ir_to_createml_rows(dataset: &Dataset) -> Vec<CreateMlImageRow> {
    // Build category lookup
    let cat_map: BTreeMap<CategoryId, &str> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();

    // Group annotations by image_id
    let mut anns_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }

    // Sort images by file_name for deterministic output
    let mut sorted_images: Vec<&Image> = dataset.images.iter().collect();
    sorted_images.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    sorted_images
        .into_iter()
        .map(|img| {
            let mut img_anns: Vec<&Annotation> =
                anns_by_image.get(&img.id).cloned().unwrap_or_default();
            img_anns.sort_by_key(|a| a.id);

            let annotations = img_anns
                .into_iter()
                .map(|ann| {
                    let (cx, cy, w, h) = ann.bbox.to_cxcywh();
                    let label = cat_map
                        .get(&ann.category_id)
                        .unwrap_or(&"unknown")
                        .to_string();

                    CreateMlAnnotation {
                        label,
                        coordinates: CreateMlCoordinates {
                            x: cx,
                            y: cy,
                            width: w,
                            height: h,
                        },
                    }
                })
                .collect();

            CreateMlImageRow {
                image: img.file_name.clone(),
                annotations,
            }
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_createml_json() -> &'static str {
        r#"[
            {
                "image": "img001.jpg",
                "annotations": [
                    {
                        "label": "cat",
                        "coordinates": { "x": 150.0, "y": 200.0, "width": 80.0, "height": 60.0 }
                    },
                    {
                        "label": "dog",
                        "coordinates": { "x": 300.0, "y": 100.0, "width": 120.0, "height": 90.0 }
                    }
                ]
            },
            {
                "image": "img002.jpg",
                "annotations": []
            }
        ]"#
    }

    #[test]
    fn parse_createml_schema_valid() {
        let rows: Vec<CreateMlImageRow> =
            serde_json::from_str(sample_createml_json()).expect("parse failed");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].image, "img001.jpg");
        assert_eq!(rows[0].annotations.len(), 2);
        assert_eq!(rows[0].annotations[0].label, "cat");
        assert_eq!(rows[0].annotations[0].coordinates.x, 150.0);
        assert_eq!(rows[1].image, "img002.jpg");
        assert_eq!(rows[1].annotations.len(), 0);
    }

    #[test]
    fn parse_createml_empty_array() {
        let rows: Vec<CreateMlImageRow> = serde_json::from_str("[]").expect("parse failed");
        assert!(rows.is_empty());
    }

    #[test]
    fn ir_to_createml_roundtrip_string() {
        let dataset = Dataset {
            images: vec![
                Image::new(1u64, "a.jpg", 640, 480),
                Image::new(2u64, "b.jpg", 800, 600),
            ],
            categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
            annotations: vec![
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(110.0, 170.0, 190.0, 230.0),
                ),
                Annotation::new(
                    2u64,
                    1u64,
                    2u64,
                    BBoxXYXY::<Pixel>::from_xyxy(240.0, 55.0, 360.0, 145.0),
                ),
            ],
            ..Default::default()
        };

        let json = to_createml_string(&dataset).expect("serialize failed");
        let rows: Vec<CreateMlImageRow> = serde_json::from_str(&json).unwrap();

        // Two images, sorted by filename
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].image, "a.jpg");
        assert_eq!(rows[1].image, "b.jpg");

        // First image has 2 annotations
        assert_eq!(rows[0].annotations.len(), 2);
        assert_eq!(rows[0].annotations[0].label, "cat");

        // Check center-based coordinates: xyxy(110,170,190,230) -> cx=150, cy=200, w=80, h=60
        let c = &rows[0].annotations[0].coordinates;
        assert!((c.x - 150.0).abs() < 1e-9);
        assert!((c.y - 200.0).abs() < 1e-9);
        assert!((c.width - 80.0).abs() < 1e-9);
        assert!((c.height - 60.0).abs() < 1e-9);

        // Second image has no annotations
        assert_eq!(rows[1].annotations.len(), 0);
    }

    #[test]
    fn writer_deterministic_order() {
        // Images out of alphabetical order
        let dataset = Dataset {
            images: vec![
                Image::new(2u64, "z.jpg", 100, 100),
                Image::new(1u64, "a.jpg", 100, 100),
            ],
            categories: vec![Category::new(1u64, "obj")],
            annotations: vec![
                Annotation::new(
                    2u64,
                    2u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
                ),
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
                ),
            ],
            ..Default::default()
        };

        let json = to_createml_string(&dataset).unwrap();
        let rows: Vec<CreateMlImageRow> = serde_json::from_str(&json).unwrap();
        assert_eq!(rows[0].image, "a.jpg");
        assert_eq!(rows[1].image, "z.jpg");
    }

    #[test]
    fn duplicate_image_rejected() {
        let rows = vec![
            CreateMlImageRow {
                image: "dup.jpg".to_string(),
                annotations: vec![],
            },
            CreateMlImageRow {
                image: "dup.jpg".to_string(),
                annotations: vec![],
            },
        ];

        let result = createml_rows_to_ir(rows, Path::new("."), Path::new("test.json"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("duplicate image entry"));
    }

    #[test]
    fn empty_image_ref_rejected() {
        let rows = vec![CreateMlImageRow {
            image: String::new(),
            annotations: vec![],
        }];

        let result = createml_rows_to_ir(rows, Path::new("."), Path::new("test.json"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("empty 'image' field"));
    }

    #[test]
    fn path_traversal_rejected() {
        let rows = vec![CreateMlImageRow {
            image: "../../../etc/passwd".to_string(),
            annotations: vec![],
        }];

        let result = createml_rows_to_ir(rows, Path::new("."), Path::new("test.json"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("relative path without '..'"));
    }
}
