//! Google Cloud AutoML Vision CSV format reader and writer.
//!
//! This module provides bidirectional conversion between the Google Cloud
//! AutoML Vision object detection CSV format and the panlabel IR.
//!
//! # Format Reference
//!
//! AutoML Vision CSV rows have a sparse layout. The canonical 11-column form is:
//! `set,path,label,xmin,ymin,,,xmax,ymax,,`
//!
//! - `set`: ML split (TRAIN, VALIDATION, TEST, UNASSIGNED)
//! - `path`: Image path (local or `gs://bucket/path`)
//! - `label`: Class label name
//! - `xmin`, `ymin`: Top-left normalized coordinates (0–1)
//! - `xmax`, `ymax`: Bottom-right normalized coordinates (0–1)
//! - Empty columns (5, 6, 9, 10) are placeholders
//!
//! The reader also accepts 9-column rows (without trailing placeholders)
//! and an optional header row.
//!
//! # Deterministic Output
//!
//! The writer emits headerless 11-column rows sorted by annotation ID.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Normalized};
use crate::error::PanlabelError;

// ============================================================================
// Internal row type
// ============================================================================

/// Attribute key for ML split on images (TRAIN, VALIDATION, TEST, UNASSIGNED).
pub const ATTR_ML_USE: &str = "automl_ml_use";
/// Attribute key for the original image URI on images.
pub const ATTR_IMAGE_URI: &str = "automl_image_uri";

struct AutoMlRow {
    ml_use: String,
    image_uri: String,
    label: String,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
}

// ============================================================================
// Public API
// ============================================================================

/// Reads a dataset from an AutoML Vision CSV file.
pub fn read_automl_vision_csv(path: &Path) -> Result<Dataset, PanlabelError> {
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let rows = parse_csv_rows(reader, path)?;
    automl_to_ir(rows, base_dir, path)
}

/// Writes a dataset to an AutoML Vision CSV file.
pub fn write_automl_vision_csv(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let csv_string = to_automl_vision_csv_string(dataset)?;
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(csv_string.as_bytes())
        .map_err(PanlabelError::Io)?;
    writer.flush().map_err(PanlabelError::Io)?;
    Ok(())
}

/// Reads a dataset from an AutoML Vision CSV string, resolving images from `base_dir`.
pub fn from_automl_vision_csv_str_with_base_dir(
    csv: &str,
    base_dir: &Path,
) -> Result<Dataset, PanlabelError> {
    let dummy_path = base_dir.join("<string>");
    let rows = parse_csv_rows(csv.as_bytes(), &dummy_path)?;
    automl_to_ir(rows, base_dir, &dummy_path)
}

/// Parses AutoML Vision CSV bytes, exercising CSV/schema parsing only.
#[cfg(feature = "fuzzing")]
pub fn parse_automl_vision_csv_slice(bytes: &[u8]) -> Result<(), csv::Error> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(bytes);
    for result in rdr.records() {
        let _record = result?;
    }
    Ok(())
}

/// Writes a dataset to an AutoML Vision CSV string (headerless, 11-column).
pub fn to_automl_vision_csv_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let dummy_path = Path::new("<string>");

    let image_lookup: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();
    let category_lookup: BTreeMap<CategoryId, &Category> =
        dataset.categories.iter().map(|cat| (cat.id, cat)).collect();

    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());

    let mut tagged: Vec<(AnnotationId, &Annotation)> =
        dataset.annotations.iter().map(|a| (a.id, a)).collect();
    tagged.sort_by_key(|(id, _)| *id);

    for (_, ann) in tagged {
        let image = image_lookup.get(&ann.image_id).ok_or_else(|| {
            PanlabelError::AutoMlVisionCsvInvalid {
                path: dummy_path.to_path_buf(),
                message: format!(
                    "Annotation {} references non-existent image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            }
        })?;

        let category = category_lookup.get(&ann.category_id).ok_or_else(|| {
            PanlabelError::AutoMlVisionCsvInvalid {
                path: dummy_path.to_path_buf(),
                message: format!(
                    "Annotation {} references non-existent category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            }
        })?;

        let bbox_norm = ann
            .bbox
            .to_normalized(image.width as f64, image.height as f64);

        let ml_use = image
            .attributes
            .get(ATTR_ML_USE)
            .map(|s| s.as_str())
            .unwrap_or("UNASSIGNED");

        let image_uri = image
            .attributes
            .get(ATTR_IMAGE_URI)
            .map(|s| s.as_str())
            .unwrap_or(&image.file_name);

        // 11-column sparse layout: set,path,label,xmin,ymin,,,xmax,ymax,,
        csv_writer
            .write_record([
                ml_use,
                image_uri,
                &category.name,
                &bbox_norm.xmin().to_string(),
                &bbox_norm.ymin().to_string(),
                "",
                "",
                &bbox_norm.xmax().to_string(),
                &bbox_norm.ymax().to_string(),
                "",
                "",
            ])
            .map_err(|source| PanlabelError::AutoMlVisionCsvWrite {
                path: dummy_path.to_path_buf(),
                source,
            })?;
    }

    let bytes = csv_writer
        .into_inner()
        .map_err(|e| PanlabelError::Io(e.into_error()))?;

    String::from_utf8(bytes).map_err(|e| PanlabelError::AutoMlVisionCsvInvalid {
        path: dummy_path.to_path_buf(),
        message: format!("Invalid UTF-8 in output: {e}"),
    })
}

// ============================================================================
// CSV Parsing
// ============================================================================

fn parse_csv_rows<R: std::io::Read>(
    reader: R,
    source_path: &Path,
) -> Result<Vec<AutoMlRow>, PanlabelError> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(reader);

    let mut rows = Vec::new();
    let mut is_first = true;
    let mut row_num: usize = 0;

    for result in csv_reader.records() {
        row_num += 1;
        let record = result.map_err(|source| PanlabelError::AutoMlVisionCsvParse {
            path: source_path.to_path_buf(),
            source,
        })?;

        let ncols = record.len();
        if ncols != 9 && ncols != 11 {
            return Err(PanlabelError::AutoMlVisionCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {}: expected 9 or 11 columns, got {}", row_num, ncols),
            });
        }

        let col0 = record.get(0).unwrap_or("");
        let col1 = record.get(1).unwrap_or("");
        let col2 = record.get(2).unwrap_or("");

        // Skip header row (check common header aliases)
        if is_first {
            is_first = false;
            let c0 = col0.to_ascii_lowercase();
            if c0 == "set" || c0 == "ml_use" {
                continue;
            }
        }

        if col1.is_empty() {
            return Err(PanlabelError::AutoMlVisionCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {}: empty image path field", row_num),
            });
        }

        let parse_coord = |idx: usize, label: &str| -> Result<f64, PanlabelError> {
            record.get(idx).unwrap_or("").parse::<f64>().map_err(|_| {
                PanlabelError::AutoMlVisionCsvInvalid {
                    path: source_path.to_path_buf(),
                    message: format!(
                        "row {}: invalid {} value '{}'",
                        row_num,
                        label,
                        record.get(idx).unwrap_or("")
                    ),
                }
            })
        };

        // Coordinates at fixed positions: xmin=3, ymin=4, xmax=7, ymax=8
        let xmin = parse_coord(3, "xmin")?;
        let ymin = parse_coord(4, "ymin")?;
        let xmax = parse_coord(7, "xmax")?;
        let ymax = parse_coord(8, "ymax")?;

        rows.push(AutoMlRow {
            ml_use: col0.to_string(),
            image_uri: col1.to_string(),
            label: col2.to_string(),
            xmin,
            ymin,
            xmax,
            ymax,
        });
    }

    Ok(rows)
}

// ============================================================================
// Conversion: AutoML CSV -> IR
// ============================================================================

fn automl_to_ir(
    rows: Vec<AutoMlRow>,
    base_dir: &Path,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    // Collect unique image URIs
    let mut image_uris: BTreeSet<String> = BTreeSet::new();
    let mut uri_ml_use: BTreeMap<String, String> = BTreeMap::new();
    for row in &rows {
        if image_uris.insert(row.image_uri.clone()) && !row.ml_use.is_empty() {
            uri_ml_use.insert(row.image_uri.clone(), row.ml_use.clone());
        }
    }

    // Resolve image dimensions
    let mut dim_cache: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    for uri in &image_uris {
        let dims = resolve_image_dimensions(base_dir, uri, source_path)?;
        dim_cache.insert(uri.clone(), dims);
    }

    // Build image map
    let image_map: BTreeMap<String, ImageId> = image_uris
        .iter()
        .enumerate()
        .map(|(i, uri)| (uri.clone(), ImageId::new((i + 1) as u64)))
        .collect();

    let images: Vec<Image> = image_uris
        .iter()
        .map(|uri| {
            let id = image_map[uri];
            let (width, height) = dim_cache[uri];
            // Use basename as file_name for display
            let file_name = uri_to_filename(uri);
            let mut img = Image::new(id, file_name, width, height);
            img.attributes
                .insert(ATTR_IMAGE_URI.to_string(), uri.clone());
            if let Some(ml_use) = uri_ml_use.get(uri) {
                img.attributes
                    .insert(ATTR_ML_USE.to_string(), ml_use.clone());
            }
            img
        })
        .collect();

    // Collect categories
    let label_names: BTreeSet<String> = rows.iter().map(|r| r.label.clone()).collect();

    let category_map: BTreeMap<String, CategoryId> = label_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), CategoryId::new((i + 1) as u64)))
        .collect();

    let categories: Vec<Category> = label_names
        .iter()
        .map(|name| Category::new(category_map[name], name.clone()))
        .collect();

    // Build annotations
    let mut annotations = Vec::new();
    for (i, row) in rows.into_iter().enumerate() {
        let image_id = image_map[&row.image_uri];
        let category_id = category_map[&row.label];
        let (width, height) = dim_cache[&row.image_uri];

        let bbox_norm = BBoxXYXY::<Normalized>::from_xyxy(row.xmin, row.ymin, row.xmax, row.ymax);
        let bbox_px = bbox_norm.to_pixel(width as f64, height as f64);

        annotations.push(Annotation::new(
            AnnotationId::new((i + 1) as u64),
            image_id,
            category_id,
            bbox_px,
        ));
    }

    Ok(Dataset {
        info: DatasetInfo::default(),
        licenses: vec![],
        images,
        categories,
        annotations,
    })
}

// ============================================================================
// Image dimension resolution
// ============================================================================

const IMAGE_EXTENSIONS: &[&str] = &[".jpg", ".jpeg", ".png", ".bmp", ".webp"];

fn resolve_image_dimensions(
    base_dir: &Path,
    image_uri: &str,
    source_path: &Path,
) -> Result<(u32, u32), PanlabelError> {
    let candidates = build_image_candidates(base_dir, image_uri);

    for candidate in &candidates {
        if candidate.exists() {
            let size = imagesize::size(candidate).map_err(|source| {
                PanlabelError::AutoMlVisionImageDimensionRead {
                    path: candidate.clone(),
                    source,
                }
            })?;
            return Ok((size.width as u32, size.height as u32));
        }
    }

    Err(PanlabelError::AutoMlVisionImageNotFound {
        path: source_path.to_path_buf(),
        image_ref: image_uri.to_string(),
    })
}

fn build_image_candidates(base_dir: &Path, image_uri: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    // If it's a GCS URI like gs://bucket/path/to/file.jpg
    if let Some(stripped) = image_uri.strip_prefix("gs://") {
        // Extract path after bucket name
        if let Some(slash_pos) = stripped.find('/') {
            let gcs_path = &stripped[slash_pos + 1..];
            candidates.push(base_dir.join(gcs_path));
            candidates.push(base_dir.join("images").join(gcs_path));
            // Also try just the basename
            if let Some(basename) = Path::new(gcs_path).file_name() {
                candidates.push(base_dir.join(basename));
                candidates.push(base_dir.join("images").join(basename));
            }
        }
    } else {
        // Local path
        let ref_path = Path::new(image_uri);
        let has_extension = ref_path.extension().is_some();

        if has_extension {
            candidates.push(base_dir.join(image_uri));
            candidates.push(base_dir.join("images").join(image_uri));
        } else {
            for ext in IMAGE_EXTENSIONS {
                let with_ext = format!("{image_uri}{ext}");
                candidates.push(base_dir.join(&with_ext));
                candidates.push(base_dir.join("images").join(&with_ext));
            }
        }
    }

    candidates
}

fn uri_to_filename(uri: &str) -> String {
    if let Some(stripped) = uri.strip_prefix("gs://") {
        if let Some(slash_pos) = stripped.find('/') {
            return stripped[slash_pos + 1..].to_string();
        }
    }
    // For local paths, use as-is
    uri.to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Pixel;

    #[test]
    fn test_parse_11_column_row() {
        let csv = "TRAIN,image.jpg,Cat,0.1,0.2,,,0.5,0.8,,\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ml_use, "TRAIN");
        assert_eq!(rows[0].image_uri, "image.jpg");
        assert_eq!(rows[0].label, "Cat");
        assert!((rows[0].xmin - 0.1).abs() < 1e-9);
        assert!((rows[0].ymin - 0.2).abs() < 1e-9);
        assert!((rows[0].xmax - 0.5).abs() < 1e-9);
        assert!((rows[0].ymax - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_parse_9_column_row() {
        let csv = "TRAIN,image.jpg,Cat,0.1,0.2,,,0.5,0.8\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");

        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_header_skipped() {
        let csv = "set,path,label,xmin,ymin,,,xmax,ymax,,\n\
                   TRAIN,img.jpg,Dog,0.0,0.0,,,0.5,0.5,,\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_wrong_column_count() {
        let csv = "a,b,c\n";
        let result = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"));
        assert!(result.is_err());
    }

    #[test]
    fn test_gcs_uri_candidates() {
        let candidates =
            build_image_candidates(Path::new("/data"), "gs://mybucket/path/to/image.jpg");
        assert!(candidates.contains(&PathBuf::from("/data/path/to/image.jpg")));
        assert!(candidates.contains(&PathBuf::from("/data/images/path/to/image.jpg")));
        assert!(candidates.contains(&PathBuf::from("/data/image.jpg")));
        assert!(candidates.contains(&PathBuf::from("/data/images/image.jpg")));
    }

    #[test]
    fn test_uri_to_filename() {
        assert_eq!(
            uri_to_filename("gs://bucket/path/to/img.jpg"),
            "path/to/img.jpg"
        );
        assert_eq!(uri_to_filename("local/path.jpg"), "local/path.jpg");
    }

    #[test]
    fn test_writer_output() {
        let mut img = Image::new(1u64, "photo.jpg", 640, 480);
        img.attributes
            .insert(ATTR_ML_USE.to_string(), "TRAIN".to_string());
        img.attributes
            .insert(ATTR_IMAGE_URI.to_string(), "photo.jpg".to_string());

        let dataset = Dataset {
            images: vec![img],
            categories: vec![Category::new(1u64, "Cat")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(64.0, 96.0, 320.0, 384.0),
            )],
            ..Default::default()
        };

        let csv = to_automl_vision_csv_string(&dataset).expect("serialize failed");
        let lines: Vec<&str> = csv.lines().collect();

        assert_eq!(lines.len(), 1); // headerless
        assert!(lines[0].starts_with("TRAIN,photo.jpg,Cat,"));
    }
}
