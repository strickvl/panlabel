//! Udacity Self-Driving Car CSV format reader and writer.
//!
//! This module provides bidirectional conversion between the Udacity
//! Self-Driving Car Dataset CSV format and the panlabel IR.
//!
//! # Format Reference
//!
//! The CSV has 8 columns with a header:
//! `filename,width,height,class,xmin,ymin,xmax,ymax`
//!
//! - `filename`: Image file path
//! - `width`, `height`: Image dimensions in pixels
//! - `class`: Category/class name
//! - `xmin`, `ymin`, `xmax`, `ymax`: **Absolute pixel** bounding box coordinates
//!
//! This is structurally identical to TFOD CSV but uses absolute pixel
//! coordinates instead of normalized (0–1) coordinates.
//!
//! # Deterministic Output
//!
//! The writer produces deterministic output by sorting rows by annotation ID.
//!
//! # Format Limitations
//!
//! Udacity CSV cannot represent:
//! - Dataset-level metadata (info, licenses)
//! - Images without annotations
//! - Annotation attributes (confidence, iscrowd, etc.)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

// ============================================================================
// Row type
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct UdacityRow {
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

/// Reads a dataset from a Udacity Self-Driving Car CSV file.
pub fn read_udacity_csv(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);

    let mut csv_reader = csv::Reader::from_reader(reader);
    let mut rows = Vec::new();

    for result in csv_reader.deserialize() {
        let row: UdacityRow = result.map_err(|source| PanlabelError::UdacityCsvParse {
            path: path.to_path_buf(),
            source,
        })?;
        rows.push(row);
    }

    udacity_to_ir(rows, path)
}

/// Writes a dataset to a Udacity Self-Driving Car CSV file.
pub fn write_udacity_csv(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);

    let rows = ir_to_udacity(dataset, path)?;

    let mut csv_writer = csv::Writer::from_writer(writer);
    for row in rows {
        csv_writer
            .serialize(&row)
            .map_err(|source| PanlabelError::UdacityCsvWrite {
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

/// Reads a dataset from a Udacity CSV string.
pub fn from_udacity_csv_str(csv_str: &str) -> Result<Dataset, PanlabelError> {
    from_udacity_csv_slice(csv_str.as_bytes())
}

/// Reads a dataset from Udacity CSV bytes.
pub fn from_udacity_csv_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    let mut csv_reader = csv::Reader::from_reader(bytes);
    let mut rows = Vec::new();
    let dummy_path = Path::new("<bytes>");

    for result in csv_reader.deserialize() {
        let row: UdacityRow = result.map_err(|source| PanlabelError::UdacityCsvParse {
            path: dummy_path.to_path_buf(),
            source,
        })?;
        rows.push(row);
    }

    udacity_to_ir(rows, dummy_path)
}

/// Writes a dataset to a Udacity CSV string.
pub fn to_udacity_csv_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let dummy_path = Path::new("<string>");
    let rows = ir_to_udacity(dataset, dummy_path)?;

    let mut csv_writer = csv::Writer::from_writer(Vec::new());
    for row in rows {
        csv_writer
            .serialize(&row)
            .map_err(|source| PanlabelError::UdacityCsvWrite {
                path: dummy_path.to_path_buf(),
                source,
            })?;
    }

    let bytes = csv_writer
        .into_inner()
        .map_err(|e| PanlabelError::Io(e.into_error()))?;

    String::from_utf8(bytes).map_err(|e| PanlabelError::UdacityCsvInvalid {
        path: dummy_path.to_path_buf(),
        message: format!("Invalid UTF-8 in output: {e}"),
    })
}

// ============================================================================
// Conversion: Udacity CSV -> IR
// ============================================================================

fn udacity_to_ir(rows: Vec<UdacityRow>, path: &Path) -> Result<Dataset, PanlabelError> {
    let mut image_dims: BTreeMap<String, (u32, u32)> = BTreeMap::new();

    for row in &rows {
        if let Some(&(existing_w, existing_h)) = image_dims.get(&row.filename) {
            if existing_w != row.width || existing_h != row.height {
                return Err(PanlabelError::UdacityCsvInvalid {
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

    let mut category_names: Vec<String> = rows.iter().map(|r| r.class_name.clone()).collect();
    category_names.sort();
    category_names.dedup();

    let category_map: BTreeMap<String, CategoryId> = category_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), CategoryId::new((i + 1) as u64)))
        .collect();

    let image_map: BTreeMap<String, ImageId> = image_dims
        .keys()
        .enumerate()
        .map(|(i, name)| (name.clone(), ImageId::new((i + 1) as u64)))
        .collect();

    let images: Vec<Image> = image_dims
        .iter()
        .map(|(filename, &(width, height))| {
            let id = image_map[filename];
            Image::new(id, filename.clone(), width, height)
        })
        .collect();

    let categories: Vec<Category> = category_names
        .iter()
        .map(|name| {
            let id = category_map[name];
            Category::new(id, name.clone())
        })
        .collect();

    // Absolute pixel coords go directly into IR (no normalization)
    let annotations: Vec<Annotation> = rows
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let image_id = image_map[&row.filename];
            let category_id = category_map[&row.class_name];
            let bbox = BBoxXYXY::<Pixel>::from_xyxy(row.xmin, row.ymin, row.xmax, row.ymax);

            Annotation::new(
                AnnotationId::new((i + 1) as u64),
                image_id,
                category_id,
                bbox,
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
// Conversion: IR -> Udacity CSV
// ============================================================================

fn ir_to_udacity(dataset: &Dataset, path: &Path) -> Result<Vec<UdacityRow>, PanlabelError> {
    let image_lookup: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();

    let category_lookup: BTreeMap<CategoryId, &Category> =
        dataset.categories.iter().map(|cat| (cat.id, cat)).collect();

    let mut rows: Vec<(AnnotationId, UdacityRow)> = Vec::with_capacity(dataset.annotations.len());

    for ann in &dataset.annotations {
        let image =
            image_lookup
                .get(&ann.image_id)
                .ok_or_else(|| PanlabelError::UdacityCsvInvalid {
                    path: path.to_path_buf(),
                    message: format!(
                        "Annotation {} references non-existent image {}",
                        ann.id.as_u64(),
                        ann.image_id.as_u64()
                    ),
                })?;

        let category = category_lookup.get(&ann.category_id).ok_or_else(|| {
            PanlabelError::UdacityCsvInvalid {
                path: path.to_path_buf(),
                message: format!(
                    "Annotation {} references non-existent category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            }
        })?;

        // Absolute pixel coords — no normalization
        rows.push((
            ann.id,
            UdacityRow {
                filename: image.file_name.clone(),
                width: image.width,
                height: image.height,
                class_name: category.name.clone(),
                xmin: ann.bbox.xmin(),
                ymin: ann.bbox.ymin(),
                xmax: ann.bbox.xmax(),
                ymax: ann.bbox.ymax(),
            },
        ));
    }

    rows.sort_by_key(|(id, _)| *id);

    Ok(rows.into_iter().map(|(_, row)| row).collect())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_udacity_csv() -> &'static str {
        "filename,width,height,class,xmin,ymin,xmax,ymax\n\
         image001.jpg,640,480,car,100,50,300,200\n\
         image001.jpg,640,480,pedestrian,400,100,500,400\n\
         image002.jpg,800,600,truck,10,20,500,400\n"
    }

    #[test]
    fn test_udacity_to_ir_basic() {
        let dataset = from_udacity_csv_str(sample_udacity_csv()).expect("parse failed");

        assert_eq!(dataset.images.len(), 2);
        assert_eq!(dataset.categories.len(), 3); // car, pedestrian, truck
        assert_eq!(dataset.annotations.len(), 3);

        assert_eq!(dataset.images[0].file_name, "image001.jpg");
        assert_eq!(dataset.images[0].width, 640);
        assert_eq!(dataset.images[1].file_name, "image002.jpg");

        // Categories sorted alphabetically
        assert_eq!(dataset.categories[0].name, "car");
        assert_eq!(dataset.categories[1].name, "pedestrian");
        assert_eq!(dataset.categories[2].name, "truck");

        // Absolute pixel coords preserved directly
        let ann = &dataset.annotations[0];
        assert!((ann.bbox.xmin() - 100.0).abs() < 0.001);
        assert!((ann.bbox.ymin() - 50.0).abs() < 0.001);
        assert!((ann.bbox.xmax() - 300.0).abs() < 0.001);
        assert!((ann.bbox.ymax() - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_roundtrip_preserves_data() {
        let original = from_udacity_csv_str(sample_udacity_csv()).expect("parse failed");
        let csv_str = to_udacity_csv_string(&original).expect("serialize failed");
        let restored = from_udacity_csv_str(&csv_str).expect("parse failed");

        assert_eq!(original.images.len(), restored.images.len());
        assert_eq!(original.categories.len(), restored.categories.len());
        assert_eq!(original.annotations.len(), restored.annotations.len());

        for (orig, rest) in original.annotations.iter().zip(restored.annotations.iter()) {
            assert!((orig.bbox.xmin() - rest.bbox.xmin()).abs() < 0.01);
            assert!((orig.bbox.ymin() - rest.bbox.ymin()).abs() < 0.01);
            assert!((orig.bbox.xmax() - rest.bbox.xmax()).abs() < 0.01);
            assert!((orig.bbox.ymax() - rest.bbox.ymax()).abs() < 0.01);
        }
    }

    #[test]
    fn test_deterministic_output() {
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
            ],
            ..Default::default()
        };

        let csv_str = to_udacity_csv_string(&dataset).expect("serialize failed");
        let lines: Vec<&str> = csv_str.lines().collect();

        assert!(lines[0].starts_with("filename,"));
        // Sorted by annotation ID: ID 1 -> a.jpg, ID 3 -> b.jpg
        assert!(lines[1].starts_with("a.jpg,"));
        assert!(lines[2].starts_with("b.jpg,"));
    }

    #[test]
    fn test_inconsistent_dimensions_error() {
        let bad_csv = "filename,width,height,class,xmin,ymin,xmax,ymax\n\
                       image.jpg,640,480,cat,10,10,50,50\n\
                       image.jpg,800,600,dog,20,20,60,60\n";
        assert!(from_udacity_csv_str(bad_csv).is_err());
    }

    #[test]
    fn test_missing_image_error() {
        let dataset = Dataset {
            images: vec![],
            categories: vec![Category::new(1u64, "cat")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
            )],
            ..Default::default()
        };
        assert!(to_udacity_csv_string(&dataset).is_err());
    }
}
