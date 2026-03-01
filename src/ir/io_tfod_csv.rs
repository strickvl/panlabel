//! TFOD CSV format reader and writer.
//!
//! This module provides bidirectional conversion between TensorFlow Object Detection
//! CSV format and the panlabel IR. TFOD CSV is a simple format commonly used with
//! TensorFlow's Object Detection API.
//!
//! # TFOD CSV Format Reference
//!
//! TFOD CSV uses **normalized coordinates** (0.0 to 1.0) with columns:
//! - `filename`: The image filename
//! - `width`: Image width in pixels
//! - `height`: Image height in pixels
//! - `class`: Category/class name
//! - `xmin`, `ymin`, `xmax`, `ymax`: Normalized bounding box coordinates
//!
//! This differs from our canonical IR format which uses absolute pixel coordinates.
//!
//! # Deterministic Output
//!
//! The writer produces deterministic output by sorting rows by annotation ID.
//! This ensures reproducible builds and meaningful diffs.
//!
//! # Format Limitations
//!
//! TFOD CSV cannot represent:
//! - Dataset-level metadata (info, licenses)
//! - Images without annotations
//! - Annotation attributes (confidence, iscrowd, etc.)
//!
//! Converting from IR to TFOD may be lossy if the dataset contains these features.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Normalized};
use crate::error::PanlabelError;

// ============================================================================
// TFOD CSV Schema Type (internal to this module)
// ============================================================================

/// A single row in the TFOD CSV format.
#[derive(Debug, Serialize, Deserialize)]
struct TfodRow {
    filename: String,
    width: u32,
    height: u32,
    #[serde(rename = "class")]
    class_name: String,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
}

// ============================================================================
// Public API
// ============================================================================

/// Reads a dataset from a TFOD CSV file.
///
/// # Arguments
/// * `path` - Path to the TFOD CSV file
///
/// # Errors
/// Returns an error if the file cannot be read, parsed, or contains
/// inconsistent data (e.g., same filename with different dimensions).
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use panlabel::ir::io_tfod_csv::read_tfod_csv;
///
/// let dataset = read_tfod_csv(Path::new("annotations.csv"))?;
/// # Ok::<(), panlabel::PanlabelError>(())
/// ```
pub fn read_tfod_csv(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);

    let mut csv_reader = csv::Reader::from_reader(reader);
    let mut rows = Vec::new();

    for result in csv_reader.deserialize() {
        let row: TfodRow = result.map_err(|source| PanlabelError::TfodCsvParse {
            path: path.to_path_buf(),
            source,
        })?;
        rows.push(row);
    }

    tfod_to_ir(rows, path)
}

/// Writes a dataset to a TFOD CSV file.
///
/// The output is deterministic: rows are sorted by annotation ID to ensure
/// reproducible output and meaningful diffs.
///
/// # Arguments
/// * `path` - Path to the output file
/// * `dataset` - The dataset to write
///
/// # Errors
/// Returns an error if the file cannot be written or if the dataset contains
/// annotations referencing non-existent images or categories.
///
/// # Notes
/// - Images without annotations will not appear in the output (TFOD limitation)
/// - Dataset metadata (info, licenses) is not preserved
/// - Annotation attributes (confidence, etc.) are not preserved
pub fn write_tfod_csv(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);

    let rows = ir_to_tfod(dataset, path)?;

    let mut csv_writer = csv::Writer::from_writer(writer);
    for row in rows {
        csv_writer
            .serialize(&row)
            .map_err(|source| PanlabelError::TfodCsvWrite {
                path: path.to_path_buf(),
                source,
            })?;
    }

    csv_writer
        .into_inner()
        .map_err(|e| PanlabelError::Io(e.into_error()))?
        .flush()
        .map_err(PanlabelError::Io)?;

    Ok(())
}

/// Reads a dataset from a TFOD CSV string.
///
/// Useful for testing without file I/O.
pub fn from_tfod_csv_str(csv_str: &str) -> Result<Dataset, PanlabelError> {
    from_tfod_csv_slice(csv_str.as_bytes())
}

/// Reads a dataset from TFOD CSV bytes.
///
/// Useful for fuzzing and processing raw bytes without requiring UTF-8 upfront.
pub fn from_tfod_csv_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    let mut csv_reader = csv::Reader::from_reader(bytes);
    let mut rows = Vec::new();
    let dummy_path = Path::new("<bytes>");

    for result in csv_reader.deserialize() {
        let row: TfodRow = result.map_err(|source| PanlabelError::TfodCsvParse {
            path: dummy_path.to_path_buf(),
            source,
        })?;
        rows.push(row);
    }

    tfod_to_ir(rows, dummy_path)
}

/// Writes a dataset to a TFOD CSV string.
///
/// Useful for testing without file I/O.
pub fn to_tfod_csv_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let dummy_path = Path::new("<string>");
    let rows = ir_to_tfod(dataset, dummy_path)?;

    let mut csv_writer = csv::Writer::from_writer(Vec::new());
    for row in rows {
        csv_writer
            .serialize(&row)
            .map_err(|source| PanlabelError::TfodCsvWrite {
                path: dummy_path.to_path_buf(),
                source,
            })?;
    }

    let bytes = csv_writer
        .into_inner()
        .map_err(|e| PanlabelError::Io(e.into_error()))?;

    String::from_utf8(bytes).map_err(|e| PanlabelError::TfodCsvInvalid {
        path: dummy_path.to_path_buf(),
        message: format!("Invalid UTF-8 in output: {}", e),
    })
}

// ============================================================================
// Conversion: TFOD CSV -> IR
// ============================================================================

/// Converts TFOD CSV rows to the panlabel IR.
///
/// # ID Assignment Policy (for determinism)
///
/// - **Images**: IDs are assigned in lexicographic order of filenames (1, 2, 3, ...)
/// - **Categories**: IDs are assigned in lexicographic order of class names (1, 2, 3, ...)
/// - **Annotations**: IDs are assigned in file row order (preserves input ordering)
fn tfod_to_ir(rows: Vec<TfodRow>, path: &Path) -> Result<Dataset, PanlabelError> {
    // Build image map: filename -> (width, height)
    // Use BTreeMap for deterministic iteration order
    let mut image_dims: BTreeMap<String, (u32, u32)> = BTreeMap::new();

    for row in &rows {
        if let Some(&(existing_w, existing_h)) = image_dims.get(&row.filename) {
            // Validate consistency: same filename should have same dimensions
            if existing_w != row.width || existing_h != row.height {
                return Err(PanlabelError::TfodCsvInvalid {
                    path: path.to_path_buf(),
                    message: format!(
                        "Inconsistent dimensions for '{}': ({}, {}) vs ({}, {})",
                        row.filename, existing_w, existing_h, row.width, row.height
                    ),
                });
            }
        } else {
            image_dims.insert(row.filename.clone(), (row.width, row.height));
        }
    }

    // Build category map: class_name -> CategoryId
    // Sorted by class name for deterministic ID assignment
    let mut category_names: Vec<String> = rows.iter().map(|r| r.class_name.clone()).collect();
    category_names.sort();
    category_names.dedup();

    let category_map: BTreeMap<String, CategoryId> = category_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), CategoryId::new((i + 1) as u64)))
        .collect();

    // Build image map: filename -> ImageId
    // Sorted by filename for deterministic ID assignment
    let image_map: BTreeMap<String, ImageId> = image_dims
        .keys()
        .enumerate()
        .map(|(i, name)| (name.clone(), ImageId::new((i + 1) as u64)))
        .collect();

    // Build images
    let images: Vec<Image> = image_dims
        .iter()
        .map(|(filename, &(width, height))| {
            let id = image_map[filename];
            Image::new(id, filename.clone(), width, height)
        })
        .collect();

    // Build categories
    let categories: Vec<Category> = category_names
        .iter()
        .map(|name| {
            let id = category_map[name];
            Category::new(id, name.clone())
        })
        .collect();

    // Build annotations (preserve row order for stable roundtrips)
    let annotations: Vec<Annotation> = rows
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let image_id = image_map[&row.filename];
            let category_id = category_map[&row.class_name];

            // Convert normalized bbox to pixel coordinates
            let bbox_norm =
                BBoxXYXY::<Normalized>::from_xyxy(row.xmin, row.ymin, row.xmax, row.ymax);
            let bbox_px = bbox_norm.to_pixel(row.width as f64, row.height as f64);

            Annotation::new(
                AnnotationId::new((i + 1) as u64),
                image_id,
                category_id,
                bbox_px,
            )
        })
        .collect();

    Ok(Dataset {
        info: DatasetInfo::default(),
        licenses: vec![],
        images,
        categories,
        annotations,
    })
}

// ============================================================================
// Conversion: IR -> TFOD CSV
// ============================================================================

/// Converts the panlabel IR to TFOD CSV rows.
///
/// Rows are sorted by annotation ID for deterministic output.
fn ir_to_tfod(dataset: &Dataset, path: &Path) -> Result<Vec<TfodRow>, PanlabelError> {
    // Build lookup maps
    let image_lookup: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();

    let category_lookup: BTreeMap<CategoryId, &Category> =
        dataset.categories.iter().map(|cat| (cat.id, cat)).collect();

    // Convert annotations to rows
    let mut rows: Vec<(AnnotationId, TfodRow)> = Vec::with_capacity(dataset.annotations.len());

    for ann in &dataset.annotations {
        // Look up image
        let image =
            image_lookup
                .get(&ann.image_id)
                .ok_or_else(|| PanlabelError::TfodCsvInvalid {
                    path: path.to_path_buf(),
                    message: format!(
                        "Annotation {} references non-existent image {}",
                        ann.id.as_u64(),
                        ann.image_id.as_u64()
                    ),
                })?;

        // Look up category
        let category =
            category_lookup
                .get(&ann.category_id)
                .ok_or_else(|| PanlabelError::TfodCsvInvalid {
                    path: path.to_path_buf(),
                    message: format!(
                        "Annotation {} references non-existent category {}",
                        ann.id.as_u64(),
                        ann.category_id.as_u64()
                    ),
                })?;

        // Convert pixel bbox to normalized coordinates
        let bbox_norm = ann
            .bbox
            .to_normalized(image.width as f64, image.height as f64);

        rows.push((
            ann.id,
            TfodRow {
                filename: image.file_name.clone(),
                width: image.width,
                height: image.height,
                class_name: category.name.clone(),
                xmin: bbox_norm.xmin(),
                ymin: bbox_norm.ymin(),
                xmax: bbox_norm.xmax(),
                ymax: bbox_norm.ymax(),
            },
        ));
    }

    // Sort by annotation ID for deterministic output
    rows.sort_by_key(|(id, _)| *id);

    Ok(rows.into_iter().map(|(_, row)| row).collect())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Pixel;

    fn sample_tfod_csv() -> &'static str {
        "filename,width,height,class,xmin,ymin,xmax,ymax\n\
         image001.jpg,640,480,person,0.1,0.2,0.5,0.8\n\
         image001.jpg,640,480,car,0.3,0.1,0.7,0.4\n\
         image002.jpg,800,600,dog,0.2,0.3,0.6,0.9\n"
    }

    #[test]
    fn test_tfod_to_ir_basic() {
        let dataset = from_tfod_csv_str(sample_tfod_csv()).expect("parse failed");

        assert_eq!(dataset.images.len(), 2);
        assert_eq!(dataset.categories.len(), 3); // car, dog, person (sorted)
        assert_eq!(dataset.annotations.len(), 3);

        // Check images are sorted by filename
        assert_eq!(dataset.images[0].file_name, "image001.jpg");
        assert_eq!(dataset.images[0].width, 640);
        assert_eq!(dataset.images[0].height, 480);
        assert_eq!(dataset.images[1].file_name, "image002.jpg");
        assert_eq!(dataset.images[1].width, 800);
        assert_eq!(dataset.images[1].height, 600);

        // Check categories are sorted by name
        assert_eq!(dataset.categories[0].name, "car");
        assert_eq!(dataset.categories[1].name, "dog");
        assert_eq!(dataset.categories[2].name, "person");

        // Check first annotation bbox conversion (normalized -> pixel)
        // xmin=0.1, ymin=0.2, xmax=0.5, ymax=0.8 on 640x480 image
        let ann = &dataset.annotations[0];
        assert!((ann.bbox.xmin() - 64.0).abs() < 0.001); // 0.1 * 640
        assert!((ann.bbox.ymin() - 96.0).abs() < 0.001); // 0.2 * 480
        assert!((ann.bbox.xmax() - 320.0).abs() < 0.001); // 0.5 * 640
        assert!((ann.bbox.ymax() - 384.0).abs() < 0.001); // 0.8 * 480
    }

    #[test]
    fn test_ir_to_tfod_bbox_conversion() {
        // Create IR dataset with pixel-space bbox
        let dataset = Dataset {
            images: vec![Image::new(1u64, "test.jpg", 640, 480)],
            categories: vec![Category::new(1u64, "person")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(64.0, 96.0, 320.0, 384.0),
            )],
            ..Default::default()
        };

        let csv_str = to_tfod_csv_string(&dataset).expect("serialize failed");

        // Parse back and verify normalized coordinates
        let restored = from_tfod_csv_str(&csv_str).expect("parse failed");
        let ann = &restored.annotations[0];

        // Should round-trip back to same pixel coords
        assert!((ann.bbox.xmin() - 64.0).abs() < 0.001);
        assert!((ann.bbox.ymin() - 96.0).abs() < 0.001);
        assert!((ann.bbox.xmax() - 320.0).abs() < 0.001);
        assert!((ann.bbox.ymax() - 384.0).abs() < 0.001);
    }

    #[test]
    fn test_roundtrip_preserves_data() {
        let original = from_tfod_csv_str(sample_tfod_csv()).expect("parse failed");

        let csv_str = to_tfod_csv_string(&original).expect("serialize failed");
        let restored = from_tfod_csv_str(&csv_str).expect("parse failed");

        // Verify counts match
        assert_eq!(original.images.len(), restored.images.len());
        assert_eq!(original.categories.len(), restored.categories.len());
        assert_eq!(original.annotations.len(), restored.annotations.len());

        // Check bbox is preserved through round-trip (within floating point tolerance)
        for (orig_ann, rest_ann) in original.annotations.iter().zip(restored.annotations.iter()) {
            assert!((orig_ann.bbox.xmin() - rest_ann.bbox.xmin()).abs() < 0.01);
            assert!((orig_ann.bbox.ymin() - rest_ann.bbox.ymin()).abs() < 0.01);
            assert!((orig_ann.bbox.xmax() - rest_ann.bbox.xmax()).abs() < 0.01);
            assert!((orig_ann.bbox.ymax() - rest_ann.bbox.ymax()).abs() < 0.01);
        }
    }

    #[test]
    fn test_from_slice_roundtrip() {
        let original = from_tfod_csv_str(sample_tfod_csv()).expect("parse failed");
        let csv_str = to_tfod_csv_string(&original).expect("serialize failed");
        let restored = from_tfod_csv_slice(csv_str.as_bytes()).expect("parse failed");

        assert_eq!(original.images, restored.images);
        assert_eq!(original.categories, restored.categories);
        assert_eq!(original.annotations.len(), restored.annotations.len());
    }

    #[test]
    fn test_deterministic_output() {
        // Create dataset with annotations out of order
        let dataset = Dataset {
            images: vec![
                Image::new(2u64, "b.jpg", 100, 100),
                Image::new(1u64, "a.jpg", 100, 100),
            ],
            categories: vec![Category::new(1u64, "cat")],
            annotations: vec![
                Annotation::new(
                    3u64,
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
                Annotation::new(
                    2u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
                ),
            ],
            ..Default::default()
        };

        let csv_str = to_tfod_csv_string(&dataset).expect("serialize failed");
        let lines: Vec<&str> = csv_str.lines().collect();

        // First line is header
        assert!(lines[0].starts_with("filename,"));

        // Rows should be sorted by annotation ID, so:
        // ID 1 -> a.jpg, ID 2 -> a.jpg, ID 3 -> b.jpg
        assert!(lines[1].starts_with("a.jpg,"));
        assert!(lines[2].starts_with("a.jpg,"));
        assert!(lines[3].starts_with("b.jpg,"));
    }

    #[test]
    fn test_inconsistent_dimensions_error() {
        let bad_csv = "filename,width,height,class,xmin,ymin,xmax,ymax\n\
                       image.jpg,640,480,cat,0.1,0.1,0.5,0.5\n\
                       image.jpg,800,600,dog,0.2,0.2,0.6,0.6\n"; // Different dims!

        let result = from_tfod_csv_str(bad_csv);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_image_error() {
        let dataset = Dataset {
            images: vec![], // No images!
            categories: vec![Category::new(1u64, "cat")],
            annotations: vec![Annotation::new(
                1u64,
                1u64, // References non-existent image
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
            )],
            ..Default::default()
        };

        let result = to_tfod_csv_string(&dataset);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_category_error() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "test.jpg", 100, 100)],
            categories: vec![], // No categories!
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64, // References non-existent category
                BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
            )],
            ..Default::default()
        };

        let result = to_tfod_csv_string(&dataset);
        assert!(result.is_err());
    }
}
