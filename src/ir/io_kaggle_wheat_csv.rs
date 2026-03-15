//! Kaggle Global Wheat Detection CSV format reader and writer.
//!
//! This module provides bidirectional conversion between the Kaggle Wheat
//! Detection CSV format and the panlabel IR.
//!
//! # Format Reference
//!
//! The CSV has 5 columns with a header:
//! `image_id,width,height,bbox,source`
//!
//! - `image_id`: Image identifier (no extension)
//! - `width`, `height`: Image dimensions in pixels
//! - `bbox`: Bounding box as a string `[x, y, width, height]` in absolute pixels
//! - `source`: Data source identifier
//!
//! # Single-Class Format
//!
//! Kaggle Wheat CSV is inherently single-class — the format has no label
//! column. All annotations are implicitly `wheat_head`. Converting a
//! multi-class dataset to this format will collapse all categories.
//!
//! # Deterministic Output
//!
//! The writer produces deterministic output by sorting rows by annotation ID.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

/// The implied single class name for all Kaggle Wheat annotations.
pub const WHEAT_HEAD_CLASS: &str = "wheat_head";
/// Attribute key for the data source on images.
pub const ATTR_SOURCE: &str = "kaggle_wheat_source";

// ============================================================================
// Public API
// ============================================================================

/// Reads a dataset from a Kaggle Wheat CSV file.
pub fn read_kaggle_wheat_csv(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    from_kaggle_wheat_csv_reader(reader, Path::new(&path.to_path_buf()))
}

/// Reads a dataset from any reader implementing `Read`.
fn from_kaggle_wheat_csv_reader<R: std::io::Read>(
    reader: R,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(reader);

    let mut rows = Vec::new();
    let mut is_first = true;
    let mut row_num: usize = 0;

    for result in csv_reader.records() {
        row_num += 1;
        let record = result.map_err(|source| PanlabelError::KaggleWheatCsvParse {
            path: source_path.to_path_buf(),
            source,
        })?;

        if record.len() != 5 {
            return Err(PanlabelError::KaggleWheatCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {}: expected 5 columns, got {}", row_num, record.len()),
            });
        }

        let col0 = record.get(0).unwrap_or("");
        let col1 = record.get(1).unwrap_or("");
        let col2 = record.get(2).unwrap_or("");
        let col3 = record.get(3).unwrap_or("");
        let col4 = record.get(4).unwrap_or("");

        // Skip header row
        if is_first {
            is_first = false;
            if col0 == "image_id" && col1 == "width" && col2 == "height" && col3 == "bbox" {
                continue;
            }
        }

        if col0.is_empty() {
            return Err(PanlabelError::KaggleWheatCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {}: empty image_id field", row_num),
            });
        }

        let width: u32 = col1
            .parse()
            .map_err(|_| PanlabelError::KaggleWheatCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {}: invalid width '{}'", row_num, col1),
            })?;

        let height: u32 = col2
            .parse()
            .map_err(|_| PanlabelError::KaggleWheatCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {}: invalid height '{}'", row_num, col2),
            })?;

        let (bx, by, bw, bh) = parse_bbox_string(col3, row_num, source_path)?;

        rows.push(KaggleRow {
            image_id: col0.to_string(),
            width,
            height,
            bx,
            by,
            bw,
            bh,
            source: col4.to_string(),
        });
    }

    kaggle_to_ir(rows, source_path)
}

/// Writes a dataset to a Kaggle Wheat CSV file.
pub fn write_kaggle_wheat_csv(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let csv_string = to_kaggle_wheat_csv_string(dataset)?;
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(csv_string.as_bytes())
        .map_err(PanlabelError::Io)?;
    writer.flush().map_err(PanlabelError::Io)?;
    Ok(())
}

/// Reads a dataset from a Kaggle Wheat CSV string.
pub fn from_kaggle_wheat_csv_str(csv: &str) -> Result<Dataset, PanlabelError> {
    from_kaggle_wheat_csv_slice(csv.as_bytes())
}

/// Reads a dataset from Kaggle Wheat CSV bytes.
pub fn from_kaggle_wheat_csv_slice(bytes: &[u8]) -> Result<Dataset, PanlabelError> {
    from_kaggle_wheat_csv_reader(bytes, Path::new("<bytes>"))
}

/// Writes a dataset to a Kaggle Wheat CSV string.
pub fn to_kaggle_wheat_csv_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let dummy_path = Path::new("<string>");

    let image_lookup: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();

    let mut tagged: Vec<(AnnotationId, &Annotation)> =
        dataset.annotations.iter().map(|a| (a.id, a)).collect();
    tagged.sort_by_key(|(id, _)| *id);

    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());

    // Write header
    csv_writer
        .write_record(["image_id", "width", "height", "bbox", "source"])
        .map_err(|source| PanlabelError::KaggleWheatCsvWrite {
            path: dummy_path.to_path_buf(),
            source,
        })?;

    for (_, ann) in tagged {
        let image = image_lookup.get(&ann.image_id).ok_or_else(|| {
            PanlabelError::KaggleWheatCsvInvalid {
                path: dummy_path.to_path_buf(),
                message: format!(
                    "Annotation {} references non-existent image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            }
        })?;

        let (_, _, w, h) = ann.bbox.to_xywh();
        let bbox_str = format!("[{}, {}, {}, {}]", ann.bbox.xmin(), ann.bbox.ymin(), w, h);

        let source = image
            .attributes
            .get(ATTR_SOURCE)
            .map(|s| s.as_str())
            .unwrap_or("");

        csv_writer
            .write_record([
                &image.file_name,
                &image.width.to_string(),
                &image.height.to_string(),
                &bbox_str,
                source,
            ])
            .map_err(|source| PanlabelError::KaggleWheatCsvWrite {
                path: dummy_path.to_path_buf(),
                source,
            })?;
    }

    let bytes = csv_writer
        .into_inner()
        .map_err(|e| PanlabelError::Io(e.into_error()))?;

    String::from_utf8(bytes).map_err(|e| PanlabelError::KaggleWheatCsvInvalid {
        path: dummy_path.to_path_buf(),
        message: format!("Invalid UTF-8 in output: {e}"),
    })
}

// ============================================================================
// Bbox string parsing
// ============================================================================

/// Parses a bbox string like `[x, y, width, height]` into four f64 values.
fn parse_bbox_string(
    s: &str,
    row_num: usize,
    path: &Path,
) -> Result<(f64, f64, f64, f64), PanlabelError> {
    let trimmed = s.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(PanlabelError::KaggleWheatCsvInvalid {
            path: path.to_path_buf(),
            message: format!(
                "row {}: bbox must be enclosed in brackets, got '{}'",
                row_num, s
            ),
        });
    }

    let inner = &trimmed[1..trimmed.len() - 1];
    let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();

    if parts.len() != 4 {
        return Err(PanlabelError::KaggleWheatCsvInvalid {
            path: path.to_path_buf(),
            message: format!(
                "row {}: bbox must have exactly 4 values, got {}",
                row_num,
                parts.len()
            ),
        });
    }

    let parse_f64 = |idx: usize, label: &str| -> Result<f64, PanlabelError> {
        parts[idx]
            .parse::<f64>()
            .map_err(|_| PanlabelError::KaggleWheatCsvInvalid {
                path: path.to_path_buf(),
                message: format!(
                    "row {}: invalid bbox {} value '{}'",
                    row_num, label, parts[idx]
                ),
            })
    };

    Ok((
        parse_f64(0, "x")?,
        parse_f64(1, "y")?,
        parse_f64(2, "width")?,
        parse_f64(3, "height")?,
    ))
}

// ============================================================================
// Internal row type
// ============================================================================

struct KaggleRow {
    image_id: String,
    width: u32,
    height: u32,
    bx: f64,
    by: f64,
    bw: f64,
    bh: f64,
    source: String,
}

// ============================================================================
// Conversion: Kaggle CSV -> IR
// ============================================================================

fn kaggle_to_ir(rows: Vec<KaggleRow>, path: &Path) -> Result<Dataset, PanlabelError> {
    // Validate dimension consistency
    let mut image_dims: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    let mut image_source: BTreeMap<String, String> = BTreeMap::new();

    for row in &rows {
        if let Some(&(ew, eh)) = image_dims.get(&row.image_id) {
            if ew != row.width || eh != row.height {
                return Err(PanlabelError::KaggleWheatCsvInvalid {
                    path: path.to_path_buf(),
                    message: format!(
                        "Inconsistent dimensions for '{}': ({}, {}) vs ({}, {})",
                        row.image_id, ew, eh, row.width, row.height
                    ),
                });
            }
        } else {
            image_dims.insert(row.image_id.clone(), (row.width, row.height));
            if !row.source.is_empty() {
                image_source.insert(row.image_id.clone(), row.source.clone());
            }
        }
    }

    // Single category
    let category = Category::new(CategoryId::new(1), WHEAT_HEAD_CLASS.to_string());
    let category_id = category.id;

    // Build image map (BTreeMap gives lexicographic order)
    let image_map: BTreeMap<String, ImageId> = image_dims
        .keys()
        .enumerate()
        .map(|(i, name)| (name.clone(), ImageId::new((i + 1) as u64)))
        .collect();

    let images: Vec<Image> = image_dims
        .iter()
        .map(|(image_id, &(width, height))| {
            let id = image_map[image_id];
            let mut img = Image::new(id, image_id.clone(), width, height);
            if let Some(source) = image_source.get(image_id) {
                img.attributes
                    .insert(ATTR_SOURCE.to_string(), source.clone());
            }
            img
        })
        .collect();

    let annotations: Vec<Annotation> = rows
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let image_id = image_map[&row.image_id];
            let bbox = BBoxXYXY::<Pixel>::from_xywh(row.bx, row.by, row.bw, row.bh);

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
        categories: vec![category],
        annotations,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_kaggle_csv() -> &'static str {
        "image_id,width,height,bbox,source\n\
         b6ab77fd7,1024,1024,\"[834.0, 222.0, 56.0, 36.0]\",usask_1\n\
         b6ab77fd7,1024,1024,\"[226.0, 548.0, 130.0, 58.0]\",usask_1\n\
         b21ccd7b0,1024,1024,\"[432.0, 104.0, 72.0, 80.0]\",arvalis_1\n"
    }

    #[test]
    fn test_kaggle_to_ir_basic() {
        let dataset = from_kaggle_wheat_csv_str(sample_kaggle_csv()).expect("parse failed");

        assert_eq!(dataset.images.len(), 2);
        assert_eq!(dataset.categories.len(), 1);
        assert_eq!(dataset.categories[0].name, "wheat_head");
        assert_eq!(dataset.annotations.len(), 3);

        // First annotation: [834, 222, 56, 36] -> xywh -> xyxy: (834, 222, 890, 258)
        let ann = &dataset.annotations[0];
        assert!((ann.bbox.xmin() - 834.0).abs() < 0.001);
        assert!((ann.bbox.ymin() - 222.0).abs() < 0.001);
        assert!((ann.bbox.xmax() - 890.0).abs() < 0.001);
        assert!((ann.bbox.ymax() - 258.0).abs() < 0.001);
    }

    #[test]
    fn test_source_attribute_roundtrip() {
        let dataset = from_kaggle_wheat_csv_str(sample_kaggle_csv()).expect("parse failed");

        // Image b21ccd7b0 should have source "arvalis_1"
        let img = dataset
            .images
            .iter()
            .find(|i| i.file_name == "b21ccd7b0")
            .unwrap();
        assert_eq!(img.attributes.get(ATTR_SOURCE).unwrap(), "arvalis_1");
    }

    #[test]
    fn test_roundtrip() {
        let original = from_kaggle_wheat_csv_str(sample_kaggle_csv()).expect("parse failed");
        let csv_str = to_kaggle_wheat_csv_string(&original).expect("serialize failed");
        let restored = from_kaggle_wheat_csv_str(&csv_str).expect("parse failed");

        assert_eq!(original.images.len(), restored.images.len());
        assert_eq!(original.annotations.len(), restored.annotations.len());

        for (orig, rest) in original.annotations.iter().zip(restored.annotations.iter()) {
            assert!((orig.bbox.xmin() - rest.bbox.xmin()).abs() < 0.01);
            assert!((orig.bbox.ymin() - rest.bbox.ymin()).abs() < 0.01);
            assert!((orig.bbox.xmax() - rest.bbox.xmax()).abs() < 0.01);
            assert!((orig.bbox.ymax() - rest.bbox.ymax()).abs() < 0.01);
        }
    }

    #[test]
    fn test_bbox_string_tolerance() {
        // Various spacing styles
        let csv = "image_id,width,height,bbox,source\n\
                   img,100,100,\"[10,20,30,40]\",src\n";
        let d = from_kaggle_wheat_csv_str(csv).expect("parse failed");
        assert_eq!(d.annotations.len(), 1);

        let csv2 = "image_id,width,height,bbox,source\n\
                    img,100,100,\"[ 10 , 20 , 30 , 40 ]\",src\n";
        let d2 = from_kaggle_wheat_csv_str(csv2).expect("parse failed");
        assert_eq!(d2.annotations.len(), 1);
    }

    #[test]
    fn test_inconsistent_dimensions_error() {
        let bad = "image_id,width,height,bbox,source\n\
                   img,100,100,\"[0,0,10,10]\",s\n\
                   img,200,200,\"[0,0,10,10]\",s\n";
        assert!(from_kaggle_wheat_csv_str(bad).is_err());
    }

    #[test]
    fn test_bad_bbox_string() {
        let bad = "image_id,width,height,bbox,source\n\
                   img,100,100,\"not a bbox\",s\n";
        assert!(from_kaggle_wheat_csv_str(bad).is_err());
    }
}
