//! OpenImages CSV format reader and writer.
//!
//! This module provides bidirectional conversion between the Google
//! OpenImages CSV annotation format and the panlabel IR.
//!
//! # Format Reference
//!
//! The CSV has 8 columns (minimal) or 13 columns (extended) with a header:
//! `ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax[,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside]`
//!
//! - Coordinates are **normalized** (0.0 to 1.0)
//! - Note the column order: XMin, **XMax**, YMin, **YMax** (not XMin, YMin, XMax, YMax)
//! - Image dimensions are resolved from local image files
//!
//! # Deterministic Output
//!
//! The writer always emits the full 8-column format (without trailing flags)
//! with rows sorted by annotation ID.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Normalized};
use crate::error::PanlabelError;

// ============================================================================
// Constants
// ============================================================================

/// Attribute key for the OpenImages source field on annotations.
pub const ATTR_SOURCE: &str = "openimages_source";
/// Attribute key for the original ImageID on images.
pub const ATTR_IMAGE_ID: &str = "openimages_image_id";

const HEADER_8: [&str; 8] = [
    "ImageID",
    "Source",
    "LabelName",
    "Confidence",
    "XMin",
    "XMax",
    "YMin",
    "YMax",
];

// ============================================================================
// Internal row type
// ============================================================================

struct OpenImagesRow {
    image_id: String,
    source: String,
    label_name: String,
    confidence: f64,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
}

// ============================================================================
// Public API
// ============================================================================

/// Reads a dataset from an OpenImages CSV file.
///
/// Image dimensions are resolved from image files on disk relative to
/// the CSV file's parent directory.
pub fn read_openimages_csv(path: &Path) -> Result<Dataset, PanlabelError> {
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let rows = parse_csv_rows(reader, path)?;
    openimages_to_ir(rows, base_dir, path)
}

/// Writes a dataset to an OpenImages CSV file.
pub fn write_openimages_csv(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let csv_string = to_openimages_csv_string(dataset)?;
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(csv_string.as_bytes())
        .map_err(PanlabelError::Io)?;
    writer.flush().map_err(PanlabelError::Io)?;
    Ok(())
}

/// Reads a dataset from an OpenImages CSV string, resolving images from `base_dir`.
pub fn from_openimages_csv_str_with_base_dir(
    csv: &str,
    base_dir: &Path,
) -> Result<Dataset, PanlabelError> {
    let dummy_path = base_dir.join("<string>");
    let rows = parse_csv_rows(csv.as_bytes(), &dummy_path)?;
    openimages_to_ir(rows, base_dir, &dummy_path)
}

/// Parses OpenImages CSV bytes, exercising CSV/schema parsing only.
///
/// Fuzz-only entrypoint: validates that the bytes parse as valid CSV rows
/// without requiring image files on disk.
#[cfg(feature = "fuzzing")]
pub fn parse_openimages_csv_slice(bytes: &[u8]) -> Result<(), csv::Error> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(bytes);
    for result in rdr.records() {
        let _record = result?;
    }
    Ok(())
}

/// Writes a dataset to an OpenImages CSV string (with header).
pub fn to_openimages_csv_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let dummy_path = Path::new("<string>");

    let image_lookup: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();
    let category_lookup: BTreeMap<CategoryId, &Category> =
        dataset.categories.iter().map(|cat| (cat.id, cat)).collect();

    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());

    // Write header
    csv_writer
        .write_record(HEADER_8)
        .map_err(|source| PanlabelError::OpenImagesCsvWrite {
            path: dummy_path.to_path_buf(),
            source,
        })?;

    // Sort annotations by ID for deterministic output
    let mut tagged: Vec<(AnnotationId, &Annotation)> =
        dataset.annotations.iter().map(|a| (a.id, a)).collect();
    tagged.sort_by_key(|(id, _)| *id);

    for (_, ann) in tagged {
        let image =
            image_lookup
                .get(&ann.image_id)
                .ok_or_else(|| PanlabelError::OpenImagesCsvInvalid {
                    path: dummy_path.to_path_buf(),
                    message: format!(
                        "Annotation {} references non-existent image {}",
                        ann.id.as_u64(),
                        ann.image_id.as_u64()
                    ),
                })?;

        let category = category_lookup.get(&ann.category_id).ok_or_else(|| {
            PanlabelError::OpenImagesCsvInvalid {
                path: dummy_path.to_path_buf(),
                message: format!(
                    "Annotation {} references non-existent category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            }
        })?;

        // Normalize pixel bbox
        let bbox_norm = ann
            .bbox
            .to_normalized(image.width as f64, image.height as f64);

        let source = ann
            .attributes
            .get(ATTR_SOURCE)
            .map(|s| s.as_str())
            .unwrap_or("xclick");

        let confidence = ann.confidence.unwrap_or(1.0);

        // Note: OpenImages column order is XMin, XMax, YMin, YMax
        csv_writer
            .write_record([
                &derive_image_id(image),
                source,
                &category.name,
                &confidence.to_string(),
                &bbox_norm.xmin().to_string(),
                &bbox_norm.xmax().to_string(),
                &bbox_norm.ymin().to_string(),
                &bbox_norm.ymax().to_string(),
            ])
            .map_err(|source| PanlabelError::OpenImagesCsvWrite {
                path: dummy_path.to_path_buf(),
                source,
            })?;
    }

    let bytes = csv_writer
        .into_inner()
        .map_err(|e| PanlabelError::Io(e.into_error()))?;

    String::from_utf8(bytes).map_err(|e| PanlabelError::OpenImagesCsvInvalid {
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
) -> Result<Vec<OpenImagesRow>, PanlabelError> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(reader);

    let mut rows = Vec::new();
    let mut is_first = true;
    let mut row_num: usize = 0;

    for result in csv_reader.records() {
        row_num += 1;
        let record = result.map_err(|source| PanlabelError::OpenImagesCsvParse {
            path: source_path.to_path_buf(),
            source,
        })?;

        // Accept 8-column (minimal) or 13-column (extended) rows
        let ncols = record.len();
        if ncols != 8 && ncols != 13 {
            return Err(PanlabelError::OpenImagesCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {}: expected 8 or 13 columns, got {}", row_num, ncols),
            });
        }

        let col0 = record.get(0).unwrap_or("");
        let col1 = record.get(1).unwrap_or("");
        let col2 = record.get(2).unwrap_or("");
        let _col3 = record.get(3).unwrap_or("");

        // Skip header row
        if is_first {
            is_first = false;
            if col0.eq_ignore_ascii_case("ImageID") && col2.eq_ignore_ascii_case("LabelName") {
                continue;
            }
        }

        if col0.is_empty() {
            return Err(PanlabelError::OpenImagesCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {}: empty ImageID field", row_num),
            });
        }

        let parse_f64 = |idx: usize, label: &str| -> Result<f64, PanlabelError> {
            record.get(idx).unwrap_or("").parse::<f64>().map_err(|_| {
                PanlabelError::OpenImagesCsvInvalid {
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

        let confidence = parse_f64(3, "Confidence")?;
        // Note: OpenImages order is XMin(4), XMax(5), YMin(6), YMax(7)
        let xmin = parse_f64(4, "XMin")?;
        let xmax = parse_f64(5, "XMax")?;
        let ymin = parse_f64(6, "YMin")?;
        let ymax = parse_f64(7, "YMax")?;

        rows.push(OpenImagesRow {
            image_id: col0.to_string(),
            source: col1.to_string(),
            label_name: col2.to_string(),
            confidence,
            xmin,
            xmax,
            ymin,
            ymax,
        });
    }

    Ok(rows)
}

// ============================================================================
// Conversion: OpenImages CSV -> IR
// ============================================================================

fn openimages_to_ir(
    rows: Vec<OpenImagesRow>,
    base_dir: &Path,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    // Collect unique image IDs (sorted for deterministic ID assignment)
    let mut image_ids: BTreeSet<String> = BTreeSet::new();
    for row in &rows {
        image_ids.insert(row.image_id.clone());
    }

    // Resolve image dimensions from disk
    let mut dim_cache: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    for img_id in &image_ids {
        let dims = resolve_image_dimensions(base_dir, img_id, source_path)?;
        dim_cache.insert(img_id.clone(), dims);
    }

    // Build image map
    let image_map: BTreeMap<String, ImageId> = image_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), ImageId::new((i + 1) as u64)))
        .collect();

    let images: Vec<Image> = image_ids
        .iter()
        .map(|img_id| {
            let id = image_map[img_id];
            let (width, height) = dim_cache[img_id];
            let mut img = Image::new(id, img_id.clone(), width, height);
            img.attributes
                .insert(ATTR_IMAGE_ID.to_string(), img_id.clone());
            img
        })
        .collect();

    // Collect unique label names
    let label_names: BTreeSet<String> = rows.iter().map(|r| r.label_name.clone()).collect();

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
        let image_id = image_map[&row.image_id];
        let category_id = category_map[&row.label_name];
        let (width, height) = dim_cache[&row.image_id];

        // Convert normalized coords to pixel
        let bbox_norm = BBoxXYXY::<Normalized>::from_xyxy(row.xmin, row.ymin, row.xmax, row.ymax);
        let bbox_px = bbox_norm.to_pixel(width as f64, height as f64);

        let mut ann = Annotation::new(
            AnnotationId::new((i + 1) as u64),
            image_id,
            category_id,
            bbox_px,
        );
        ann.confidence = Some(row.confidence);
        if !row.source.is_empty() {
            ann.attributes.insert(ATTR_SOURCE.to_string(), row.source);
        }

        annotations.push(ann);
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
    image_ref: &str,
    source_path: &Path,
) -> Result<(u32, u32), PanlabelError> {
    let candidates = build_image_candidates(base_dir, image_ref);

    for candidate in &candidates {
        if candidate.exists() {
            let size = imagesize::size(candidate).map_err(|source| {
                PanlabelError::OpenImagesImageDimensionRead {
                    path: candidate.clone(),
                    source,
                }
            })?;
            return Ok((size.width as u32, size.height as u32));
        }
    }

    Err(PanlabelError::OpenImagesImageNotFound {
        path: source_path.to_path_buf(),
        image_ref: image_ref.to_string(),
    })
}

fn build_image_candidates(base_dir: &Path, image_ref: &str) -> Vec<PathBuf> {
    let ref_path = Path::new(image_ref);
    let has_extension = ref_path.extension().is_some();
    let mut candidates = Vec::new();

    if has_extension {
        // Exact match locations
        candidates.push(base_dir.join(image_ref));
        candidates.push(base_dir.join("images").join(image_ref));
    } else {
        // Try with common image extensions
        for ext in IMAGE_EXTENSIONS {
            let with_ext = format!("{image_ref}{ext}");
            candidates.push(base_dir.join(&with_ext));
            candidates.push(base_dir.join("images").join(&with_ext));
        }
    }

    candidates
}

fn derive_image_id(image: &Image) -> String {
    if let Some(id) = image.attributes.get(ATTR_IMAGE_ID) {
        return id.clone();
    }
    // Fall back to file stem
    Path::new(&image.file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&image.file_name)
        .to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Pixel;

    #[test]
    fn test_parse_8_column_row() {
        let csv = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax\n\
                   abc123,xclick,Cat,1.0,0.1,0.5,0.2,0.8\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].image_id, "abc123");
        assert_eq!(rows[0].label_name, "Cat");
        assert!((rows[0].confidence - 1.0).abs() < 1e-9);
        assert!((rows[0].xmin - 0.1).abs() < 1e-9);
        assert!((rows[0].xmax - 0.5).abs() < 1e-9);
        assert!((rows[0].ymin - 0.2).abs() < 1e-9);
        assert!((rows[0].ymax - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_header_skipped() {
        let csv = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax\n\
                   id1,src,Dog,0.9,0.0,0.5,0.0,0.5\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_writer_output() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "abc.jpg", 640, 480)],
            categories: vec![Category::new(1u64, "Cat")],
            annotations: vec![{
                let mut ann = Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(64.0, 96.0, 320.0, 384.0),
                );
                ann.confidence = Some(0.95);
                ann
            }],
            ..Default::default()
        };

        let csv = to_openimages_csv_string(&dataset).expect("serialize failed");
        let lines: Vec<&str> = csv.lines().collect();

        assert_eq!(lines.len(), 2); // header + 1 row
        assert!(lines[0].starts_with("ImageID,"));

        // Image ID derived from file stem
        assert!(lines[1].starts_with("abc,"));
        // Confidence preserved
        assert!(lines[1].contains("0.95"));
    }

    #[test]
    fn test_wrong_column_count() {
        let csv = "a,b,c\n";
        let result = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"));
        assert!(result.is_err());
    }

    #[test]
    fn test_image_candidates_with_extension() {
        let candidates = build_image_candidates(Path::new("/data"), "photo.jpg");
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0], PathBuf::from("/data/photo.jpg"));
        assert_eq!(candidates[1], PathBuf::from("/data/images/photo.jpg"));
    }

    #[test]
    fn test_image_candidates_without_extension() {
        let candidates = build_image_candidates(Path::new("/data"), "abc123");
        // 5 extensions × 2 locations = 10
        assert_eq!(candidates.len(), 10);
        assert_eq!(candidates[0], PathBuf::from("/data/abc123.jpg"));
        assert_eq!(candidates[1], PathBuf::from("/data/images/abc123.jpg"));
    }
}
