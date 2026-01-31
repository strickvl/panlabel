//! COCO JSON format reader and writer.
//!
//! This module provides bidirectional conversion between COCO JSON format
//! and the panlabel IR. COCO is one of the most widely used formats for
//! object detection datasets.
//!
//! # COCO Format Reference
//!
//! COCO bounding boxes use `[x, y, width, height]` format where:
//! - `(x, y)` is the top-left corner in absolute pixel coordinates
//! - `width` and `height` are the dimensions
//!
//! This differs from our canonical IR format which uses XYXY (xmin, ymin, xmax, ymax).
//!
//! # Deterministic Output
//!
//! The writer produces deterministic output by sorting all lists by ID.
//! This ensures reproducible builds and meaningful diffs.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image, License};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, LicenseId, Pixel};
use crate::error::PanlabelError;

// ============================================================================
// COCO Schema Types (internal to this module)
// ============================================================================

/// Top-level COCO dataset structure.
#[derive(Debug, Serialize, Deserialize)]
struct CocoDataset {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    info: Option<CocoInfo>,

    #[serde(default)]
    licenses: Vec<CocoLicense>,

    images: Vec<CocoImage>,

    annotations: Vec<CocoAnnotation>,

    categories: Vec<CocoCategory>,
}

/// COCO dataset info block.
#[derive(Debug, Default, Serialize, Deserialize)]
struct CocoInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    year: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    version: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    contributor: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    url: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    date_created: Option<String>,
}

/// COCO license entry.
#[derive(Debug, Serialize, Deserialize)]
struct CocoLicense {
    id: u64,
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    url: Option<String>,
}

/// COCO image entry.
#[derive(Debug, Serialize, Deserialize)]
struct CocoImage {
    id: u64,
    width: u32,
    height: u32,
    file_name: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    license: Option<u64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    date_captured: Option<String>,
}

/// COCO category entry.
#[derive(Debug, Serialize, Deserialize)]
struct CocoCategory {
    id: u64,
    name: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    supercategory: Option<String>,
}

/// COCO annotation entry.
#[derive(Debug, Serialize, Deserialize)]
struct CocoAnnotation {
    id: u64,
    image_id: u64,
    category_id: u64,

    /// COCO bbox format: [x, y, width, height] with (x,y) as top-left corner
    bbox: [f64; 4],

    #[serde(default, skip_serializing_if = "Option::is_none")]
    area: Option<f64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    iscrowd: Option<u8>,

    /// Segmentation data (polygons or RLE). We accept but ignore for detection.
    #[serde(default)]
    segmentation: serde_json::Value,

    /// Score/confidence for detection results
    #[serde(default, skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
}

// ============================================================================
// Public API
// ============================================================================

/// Reads a dataset from a COCO JSON file.
///
/// # Arguments
/// * `path` - Path to the COCO JSON file
///
/// # Errors
/// Returns an error if the file cannot be read or parsed.
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use panlabel::ir::io_coco_json::read_coco_json;
///
/// let dataset = read_coco_json(Path::new("annotations.json"))?;
/// # Ok::<(), panlabel::PanlabelError>(())
/// ```
pub fn read_coco_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);

    let coco: CocoDataset =
        serde_json::from_reader(reader).map_err(|source| PanlabelError::CocoJsonParse {
            path: path.to_path_buf(),
            source,
        })?;

    Ok(coco_to_ir(coco))
}

/// Writes a dataset to a COCO JSON file.
///
/// The output is deterministic: all lists are sorted by ID to ensure
/// reproducible output and meaningful diffs.
///
/// # Arguments
/// * `path` - Path to the output file
/// * `dataset` - The dataset to write
///
/// # Errors
/// Returns an error if the file cannot be written.
pub fn write_coco_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);

    let coco = ir_to_coco(dataset);

    serde_json::to_writer_pretty(writer, &coco).map_err(|source| PanlabelError::CocoJsonWrite {
        path: path.to_path_buf(),
        source,
    })
}

/// Reads a dataset from a COCO JSON string.
///
/// Useful for testing without file I/O.
pub fn from_coco_str(json: &str) -> Result<Dataset, serde_json::Error> {
    let coco: CocoDataset = serde_json::from_str(json)?;
    Ok(coco_to_ir(coco))
}

/// Reads a dataset from a COCO JSON byte slice.
///
/// Useful for fuzzing and processing raw bytes without UTF-8 validation overhead.
pub fn from_coco_slice(bytes: &[u8]) -> Result<Dataset, serde_json::Error> {
    let coco: CocoDataset = serde_json::from_slice(bytes)?;
    Ok(coco_to_ir(coco))
}

/// Writes a dataset to a COCO JSON string.
///
/// Useful for testing without file I/O.
pub fn to_coco_string(dataset: &Dataset) -> Result<String, serde_json::Error> {
    let coco = ir_to_coco(dataset);
    serde_json::to_string_pretty(&coco)
}

// ============================================================================
// Conversion: COCO -> IR
// ============================================================================

fn coco_to_ir(coco: CocoDataset) -> Dataset {
    // Convert info
    let info = if let Some(coco_info) = coco.info {
        DatasetInfo {
            name: None, // COCO doesn't have a "name" field in info
            version: coco_info.version,
            description: coco_info.description,
            url: coco_info.url,
            year: coco_info.year,
            contributor: coco_info.contributor,
            date_created: coco_info.date_created,
        }
    } else {
        DatasetInfo::default()
    };

    // Convert licenses
    let licenses = coco
        .licenses
        .into_iter()
        .map(|l| License {
            id: LicenseId::new(l.id),
            name: l.name,
            url: l.url,
        })
        .collect();

    // Convert images
    let images = coco
        .images
        .into_iter()
        .map(|img| Image {
            id: ImageId::new(img.id),
            file_name: img.file_name,
            width: img.width,
            height: img.height,
            license_id: img.license.map(LicenseId::new),
            date_captured: img.date_captured,
        })
        .collect();

    // Convert categories
    let categories = coco
        .categories
        .into_iter()
        .map(|cat| Category {
            id: CategoryId::new(cat.id),
            name: cat.name,
            supercategory: cat.supercategory,
        })
        .collect();

    // Convert annotations
    let annotations = coco
        .annotations
        .into_iter()
        .map(|ann| {
            let [x, y, w, h] = ann.bbox;
            let bbox = BBoxXYXY::<Pixel>::from_xywh(x, y, w, h);

            let mut annotation = Annotation::new(
                AnnotationId::new(ann.id),
                ImageId::new(ann.image_id),
                CategoryId::new(ann.category_id),
                bbox,
            );

            // Map score to confidence
            if let Some(score) = ann.score {
                annotation.confidence = Some(score);
            }

            // Store iscrowd as attribute if present
            if let Some(iscrowd) = ann.iscrowd {
                annotation
                    .attributes
                    .insert("iscrowd".to_string(), iscrowd.to_string());
            }

            // Store area as attribute if present (for round-trip preservation)
            if let Some(area) = ann.area {
                annotation
                    .attributes
                    .insert("area".to_string(), format!("{:.6}", area));
            }

            annotation
        })
        .collect();

    Dataset {
        info,
        licenses,
        images,
        categories,
        annotations,
    }
}

// ============================================================================
// Conversion: IR -> COCO
// ============================================================================

fn ir_to_coco(dataset: &Dataset) -> CocoDataset {
    // Convert info (always include if any field is set)
    let info = Some(CocoInfo {
        year: dataset.info.year,
        version: dataset.info.version.clone(),
        description: dataset.info.description.clone(),
        contributor: dataset.info.contributor.clone(),
        url: dataset.info.url.clone(),
        date_created: dataset.info.date_created.clone(),
    });

    // Convert and sort licenses by ID for deterministic output
    let mut licenses: Vec<CocoLicense> = dataset
        .licenses
        .iter()
        .map(|l| CocoLicense {
            id: l.id.as_u64(),
            name: l.name.clone(),
            url: l.url.clone(),
        })
        .collect();
    licenses.sort_by_key(|l| l.id);

    // Convert and sort images by ID
    let mut images: Vec<CocoImage> = dataset
        .images
        .iter()
        .map(|img| CocoImage {
            id: img.id.as_u64(),
            width: img.width,
            height: img.height,
            file_name: img.file_name.clone(),
            license: img.license_id.map(|l| l.as_u64()),
            date_captured: img.date_captured.clone(),
        })
        .collect();
    images.sort_by_key(|i| i.id);

    // Convert and sort categories by ID
    let mut categories: Vec<CocoCategory> = dataset
        .categories
        .iter()
        .map(|cat| CocoCategory {
            id: cat.id.as_u64(),
            name: cat.name.clone(),
            supercategory: cat.supercategory.clone(),
        })
        .collect();
    categories.sort_by_key(|c| c.id);

    // Convert and sort annotations by ID
    let mut annotations: Vec<CocoAnnotation> = dataset
        .annotations
        .iter()
        .map(|ann| {
            let (x, y, w, h) = ann.bbox.to_xywh();

            // Try to use stored area, otherwise compute from bbox
            let area = ann
                .attributes
                .get("area")
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or_else(|| ann.bbox.area());

            // Try to use stored iscrowd, otherwise default to 0
            let iscrowd = ann
                .attributes
                .get("iscrowd")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(0);

            CocoAnnotation {
                id: ann.id.as_u64(),
                image_id: ann.image_id.as_u64(),
                category_id: ann.category_id.as_u64(),
                bbox: [x, y, w, h],
                area: Some(area),
                iscrowd: Some(iscrowd),
                segmentation: serde_json::Value::Array(vec![]), // Empty for detection-only
                score: ann.confidence,
            }
        })
        .collect();
    annotations.sort_by_key(|a| a.id);

    CocoDataset {
        info,
        licenses,
        images,
        annotations,
        categories,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_coco_json() -> &'static str {
        r#"{
            "info": {
                "year": 2024,
                "version": "1.0",
                "description": "Test dataset"
            },
            "licenses": [
                {"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}
            ],
            "images": [
                {"id": 1, "width": 640, "height": 480, "file_name": "image001.jpg", "license": 1}
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "human"}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10.0, 20.0, 90.0, 60.0],
                    "area": 5400.0,
                    "iscrowd": 0
                }
            ]
        }"#
    }

    #[test]
    fn test_coco_to_ir_basic() {
        let dataset = from_coco_str(sample_coco_json()).expect("parse failed");

        assert_eq!(dataset.images.len(), 1);
        assert_eq!(dataset.categories.len(), 1);
        assert_eq!(dataset.annotations.len(), 1);
        assert_eq!(dataset.licenses.len(), 1);

        // Check info conversion
        assert_eq!(dataset.info.year, Some(2024));
        assert_eq!(dataset.info.version, Some("1.0".to_string()));
        assert_eq!(dataset.info.description, Some("Test dataset".to_string()));

        // Check license conversion
        assert_eq!(dataset.licenses[0].id.as_u64(), 1);
        assert_eq!(dataset.licenses[0].name, "CC BY 4.0");

        // Check image conversion
        let img = &dataset.images[0];
        assert_eq!(img.id.as_u64(), 1);
        assert_eq!(img.file_name, "image001.jpg");
        assert_eq!(img.width, 640);
        assert_eq!(img.height, 480);
        assert_eq!(img.license_id, Some(LicenseId::new(1)));

        // Check category conversion
        let cat = &dataset.categories[0];
        assert_eq!(cat.id.as_u64(), 1);
        assert_eq!(cat.name, "person");
        assert_eq!(cat.supercategory, Some("human".to_string()));

        // Check annotation conversion (XYWH -> XYXY)
        let ann = &dataset.annotations[0];
        assert_eq!(ann.id.as_u64(), 1);
        assert_eq!(ann.image_id.as_u64(), 1);
        assert_eq!(ann.category_id.as_u64(), 1);

        // COCO [10, 20, 90, 60] should become XYXY [10, 20, 100, 80]
        assert_eq!(ann.bbox.xmin(), 10.0);
        assert_eq!(ann.bbox.ymin(), 20.0);
        assert_eq!(ann.bbox.xmax(), 100.0); // x + width = 10 + 90
        assert_eq!(ann.bbox.ymax(), 80.0); // y + height = 20 + 60
    }

    #[test]
    fn test_ir_to_coco_bbox_conversion() {
        // Create IR dataset with XYXY bbox
        let dataset = Dataset {
            images: vec![Image::new(1u64, "test.jpg", 640, 480)],
            categories: vec![Category::new(1u64, "dog")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 80.0),
            )],
            ..Default::default()
        };

        let json = to_coco_string(&dataset).expect("serialize failed");

        // Parse back and verify COCO format
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let bbox = &parsed["annotations"][0]["bbox"];

        // XYXY [10, 20, 100, 80] should become XYWH [10, 20, 90, 60]
        assert_eq!(bbox[0], 10.0); // x
        assert_eq!(bbox[1], 20.0); // y
        assert_eq!(bbox[2], 90.0); // width
        assert_eq!(bbox[3], 60.0); // height
    }

    #[test]
    fn test_roundtrip_preserves_data() {
        let original = from_coco_str(sample_coco_json()).expect("parse failed");

        let json = to_coco_string(&original).expect("serialize failed");
        let restored = from_coco_str(&json).expect("parse failed");

        // Verify key data is preserved
        assert_eq!(original.images.len(), restored.images.len());
        assert_eq!(original.categories.len(), restored.categories.len());
        assert_eq!(original.annotations.len(), restored.annotations.len());

        // Check bbox is preserved through round-trip
        let orig_bbox = &original.annotations[0].bbox;
        let rest_bbox = &restored.annotations[0].bbox;
        assert_eq!(orig_bbox.xmin(), rest_bbox.xmin());
        assert_eq!(orig_bbox.ymin(), rest_bbox.ymin());
        assert_eq!(orig_bbox.xmax(), rest_bbox.xmax());
        assert_eq!(orig_bbox.ymax(), rest_bbox.ymax());
    }

    #[test]
    fn test_deterministic_output() {
        // Create dataset with IDs out of order
        let dataset = Dataset {
            images: vec![
                Image::new(3u64, "c.jpg", 100, 100),
                Image::new(1u64, "a.jpg", 100, 100),
                Image::new(2u64, "b.jpg", 100, 100),
            ],
            categories: vec![Category::new(2u64, "cat"), Category::new(1u64, "dog")],
            annotations: vec![
                Annotation::new(
                    2u64,
                    1u64,
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

        let json = to_coco_string(&dataset).expect("serialize failed");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Verify images are sorted by ID
        assert_eq!(parsed["images"][0]["id"], 1);
        assert_eq!(parsed["images"][1]["id"], 2);
        assert_eq!(parsed["images"][2]["id"], 3);

        // Verify categories are sorted by ID
        assert_eq!(parsed["categories"][0]["id"], 1);
        assert_eq!(parsed["categories"][1]["id"], 2);

        // Verify annotations are sorted by ID
        assert_eq!(parsed["annotations"][0]["id"], 1);
        assert_eq!(parsed["annotations"][1]["id"], 2);
    }

    #[test]
    fn test_confidence_to_score_mapping() {
        let mut ann = Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(0.0, 0.0, 10.0, 10.0),
        );
        ann.confidence = Some(0.95);

        let dataset = Dataset {
            images: vec![Image::new(1u64, "test.jpg", 100, 100)],
            categories: vec![Category::new(1u64, "test")],
            annotations: vec![ann],
            ..Default::default()
        };

        let json = to_coco_string(&dataset).expect("serialize failed");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["annotations"][0]["score"], 0.95);
    }

    #[test]
    fn test_iscrowd_attribute_roundtrip() {
        let coco_with_crowd = r#"{
            "images": [{"id": 1, "width": 100, "height": 100, "file_name": "test.jpg"}],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [{
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 50, 50],
                "area": 2500,
                "iscrowd": 1
            }]
        }"#;

        let dataset = from_coco_str(coco_with_crowd).expect("parse failed");
        assert_eq!(
            dataset.annotations[0].attributes.get("iscrowd"),
            Some(&"1".to_string())
        );

        // Write back and verify iscrowd is preserved
        let json = to_coco_string(&dataset).expect("serialize failed");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["annotations"][0]["iscrowd"], 1);
    }
}
