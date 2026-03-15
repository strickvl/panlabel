//! RetinaNet Keras CSV format reader and writer.
//!
//! This module provides bidirectional conversion between the RetinaNet Keras CSV
//! annotation format and the panlabel IR. RetinaNet CSV is a simple 6-column format
//! used with the keras-retinanet project.
//!
//! # RetinaNet CSV Format Reference
//!
//! Each row is: `path,x1,y1,x2,y2,class_name`
//!
//! - `path`: Image file path (relative or absolute)
//! - `x1`, `y1`, `x2`, `y2`: Absolute pixel bounding box coordinates
//! - `class_name`: Category/class name
//!
//! Unannotated images are represented as `path,,,,,` (all-empty row).
//!
//! The format does **not** include image dimensions; the reader resolves them
//! from image files on disk using `imagesize`.
//!
//! # Deterministic Output
//!
//! The writer produces deterministic output: images are sorted by filename,
//! annotations within each image are sorted by annotation ID.
//!
//! # Format Limitations
//!
//! RetinaNet CSV cannot represent:
//! - Dataset-level metadata (info, licenses)
//! - Annotation attributes (confidence, iscrowd, etc.)
//!
//! Unlike TFOD CSV, RetinaNet CSV **can** represent unannotated images.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

// ============================================================================
// Constants
// ============================================================================

/// The expected header columns (used for optional header detection).
const HEADER_COLUMNS: [&str; 6] = ["path", "x1", "y1", "x2", "y2", "class_name"];

// ============================================================================
// Internal row representation
// ============================================================================

/// A parsed row from a RetinaNet CSV file.
///
/// Either an annotation row (all fields present) or an empty row (image path
/// only, no bbox/class).
#[derive(Debug)]
enum RetinanetRow {
    /// Image with an annotation: path, x1, y1, x2, y2, class_name.
    Annotation {
        path: String,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        class_name: String,
    },
    /// Unannotated image: `path,,,,,`.
    Empty { path: String },
}

// ============================================================================
// Public API
// ============================================================================

/// Reads a dataset from a RetinaNet CSV file.
///
/// Image dimensions are resolved from the image files on disk, relative to
/// the CSV file's parent directory.
///
/// # Arguments
/// * `path` - Path to the RetinaNet CSV file
///
/// # Errors
/// Returns an error if the file cannot be read, parsed, or if referenced
/// images cannot be found on disk.
pub fn read_retinanet_csv(path: &Path) -> Result<Dataset, PanlabelError> {
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);

    let rows = parse_csv_rows(reader, path)?;
    retinanet_to_ir(rows, base_dir, path)
}

/// Writes a dataset to a RetinaNet CSV file.
///
/// The output is headerless and deterministic: images are sorted by filename,
/// annotations within each image are sorted by annotation ID. Unannotated
/// images emit exactly one `path,,,,,` row.
pub fn write_retinanet_csv(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let csv_string = to_retinanet_csv_string(dataset)?;

    let file = File::create(path).map_err(PanlabelError::Io)?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(csv_string.as_bytes())
        .map_err(PanlabelError::Io)?;
    writer.flush().map_err(PanlabelError::Io)?;

    Ok(())
}

/// Reads a dataset from a RetinaNet CSV string, resolving images from `base_dir`.
///
/// Useful for testing without file I/O.
pub fn from_retinanet_csv_str_with_base_dir(
    csv: &str,
    base_dir: &Path,
) -> Result<Dataset, PanlabelError> {
    let dummy_path = base_dir.join("<string>");
    let reader = std::io::Cursor::new(csv.as_bytes());
    let rows = parse_csv_rows(reader, &dummy_path)?;
    retinanet_to_ir(rows, base_dir, &dummy_path)
}

/// Parses a RetinaNet CSV byte slice, exercising CSV/schema parsing only.
///
/// Fuzz-only entrypoint: validates that the bytes parse as valid CSV rows
/// without requiring image files on disk.
#[cfg(feature = "fuzzing")]
pub fn parse_retinanet_csv_slice(bytes: &[u8]) -> Result<(), csv::Error> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(bytes);

    for result in rdr.records() {
        let _record = result?;
    }
    Ok(())
}

/// Writes a dataset to a RetinaNet CSV string (headerless).
///
/// Images are sorted by filename, annotations within each image by annotation
/// ID. Unannotated images emit one `path,,,,,` row.
pub fn to_retinanet_csv_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let dummy_path = Path::new("<string>");

    // Build lookup maps
    let category_lookup: BTreeMap<CategoryId, &Category> =
        dataset.categories.iter().map(|cat| (cat.id, cat)).collect();

    // Group annotations by image_id
    let mut anns_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }

    // Sort images by file_name for deterministic output
    let mut sorted_images: Vec<&Image> = dataset.images.iter().collect();
    sorted_images.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());

    for img in sorted_images {
        match anns_by_image.get(&img.id) {
            Some(anns) if !anns.is_empty() => {
                // Sort annotations by ID for deterministic output
                let mut sorted_anns: Vec<&Annotation> = anns.clone();
                sorted_anns.sort_by_key(|a| a.id);

                for ann in sorted_anns {
                    let category = category_lookup.get(&ann.category_id).ok_or_else(|| {
                        PanlabelError::RetinanetCsvInvalid {
                            path: dummy_path.to_path_buf(),
                            message: format!(
                                "Annotation {} references non-existent category {}",
                                ann.id.as_u64(),
                                ann.category_id.as_u64()
                            ),
                        }
                    })?;

                    csv_writer
                        .write_record([
                            &img.file_name,
                            &ann.bbox.xmin().to_string(),
                            &ann.bbox.ymin().to_string(),
                            &ann.bbox.xmax().to_string(),
                            &ann.bbox.ymax().to_string(),
                            &category.name,
                        ])
                        .map_err(|source| PanlabelError::RetinanetCsvWrite {
                            path: dummy_path.to_path_buf(),
                            source,
                        })?;
                }
            }
            _ => {
                // Unannotated image: emit `path,,,,,`
                csv_writer
                    .write_record([&img.file_name, "", "", "", "", ""])
                    .map_err(|source| PanlabelError::RetinanetCsvWrite {
                        path: dummy_path.to_path_buf(),
                        source,
                    })?;
            }
        }
    }

    let bytes = csv_writer
        .into_inner()
        .map_err(|e| PanlabelError::Io(e.into_error()))?;

    String::from_utf8(bytes).map_err(|e| PanlabelError::RetinanetCsvInvalid {
        path: dummy_path.to_path_buf(),
        message: format!("Invalid UTF-8 in output: {e}"),
    })
}

// ============================================================================
// CSV Parsing
// ============================================================================

/// Parses raw CSV records into `RetinanetRow` values, handling optional header
/// detection, empty rows, annotation rows, and partial-row rejection.
fn parse_csv_rows<R: std::io::Read>(
    reader: R,
    source_path: &Path,
) -> Result<Vec<RetinanetRow>, PanlabelError> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(reader);

    let mut rows = Vec::new();
    let mut is_first = true;
    let mut row_num: usize = 0;

    for result in csv_reader.records() {
        row_num += 1;
        let record = result.map_err(|source| PanlabelError::RetinanetCsvParse {
            path: source_path.to_path_buf(),
            source,
        })?;

        // Need exactly 6 fields
        if record.len() != 6 {
            return Err(PanlabelError::RetinanetCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!(
                    "row {}: expected 6 columns, got {} in row: {:?}",
                    row_num,
                    record.len(),
                    record.iter().collect::<Vec<_>>()
                ),
            });
        }

        let col0 = record.get(0).unwrap_or("");
        let col1 = record.get(1).unwrap_or("");
        let col2 = record.get(2).unwrap_or("");
        let col3 = record.get(3).unwrap_or("");
        let col4 = record.get(4).unwrap_or("");
        let col5 = record.get(5).unwrap_or("");

        // Skip optional header row if it exactly matches expected column names
        if is_first {
            is_first = false;
            if col0 == HEADER_COLUMNS[0]
                && col1 == HEADER_COLUMNS[1]
                && col2 == HEADER_COLUMNS[2]
                && col3 == HEADER_COLUMNS[3]
                && col4 == HEADER_COLUMNS[4]
                && col5 == HEADER_COLUMNS[5]
            {
                continue;
            }
        }

        // Path must be non-empty
        if col0.is_empty() {
            return Err(PanlabelError::RetinanetCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {}: empty path field", row_num),
            });
        }

        let bbox_fields = [col1, col2, col3, col4, col5];
        let all_empty = bbox_fields.iter().all(|f| f.is_empty());
        let all_present = bbox_fields.iter().all(|f| !f.is_empty());

        if all_empty {
            // Unannotated image row: `path,,,,,`
            rows.push(RetinanetRow::Empty {
                path: col0.to_string(),
            });
        } else if all_present {
            // Annotation row: parse coordinates
            let x1: f64 = col1
                .parse()
                .map_err(|_| PanlabelError::RetinanetCsvInvalid {
                    path: source_path.to_path_buf(),
                    message: format!(
                        "row {}: invalid x1 value '{}' for image '{}'",
                        row_num, col1, col0
                    ),
                })?;
            let y1: f64 = col2
                .parse()
                .map_err(|_| PanlabelError::RetinanetCsvInvalid {
                    path: source_path.to_path_buf(),
                    message: format!(
                        "row {}: invalid y1 value '{}' for image '{}'",
                        row_num, col2, col0
                    ),
                })?;
            let x2: f64 = col3
                .parse()
                .map_err(|_| PanlabelError::RetinanetCsvInvalid {
                    path: source_path.to_path_buf(),
                    message: format!(
                        "row {}: invalid x2 value '{}' for image '{}'",
                        row_num, col3, col0
                    ),
                })?;
            let y2: f64 = col4
                .parse()
                .map_err(|_| PanlabelError::RetinanetCsvInvalid {
                    path: source_path.to_path_buf(),
                    message: format!(
                        "row {}: invalid y2 value '{}' for image '{}'",
                        row_num, col4, col0
                    ),
                })?;

            rows.push(RetinanetRow::Annotation {
                path: col0.to_string(),
                x1,
                y1,
                x2,
                y2,
                class_name: col5.to_string(),
            });
        } else {
            // Partial row: some bbox fields present, some empty
            return Err(PanlabelError::RetinanetCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!(
                    "row {}: partial annotation row for image '{}': some bbox/class fields are empty while others are present",
                    row_num, col0
                ),
            });
        }
    }

    Ok(rows)
}

// ============================================================================
// Conversion: RetinaNet CSV -> IR
// ============================================================================

/// Converts parsed RetinaNet CSV rows to the panlabel IR.
///
/// # ID Assignment Policy (for determinism)
///
/// - **Images**: IDs are assigned in lexicographic order of paths (1, 2, 3, ...)
/// - **Categories**: IDs are assigned in lexicographic order of class names (1, 2, 3, ...)
/// - **Annotations**: IDs are assigned in row order (preserves input ordering)
fn retinanet_to_ir(
    rows: Vec<RetinanetRow>,
    base_dir: &Path,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    // Collect unique image paths (preserving order of first appearance for
    // annotation ID stability, but we sort them for image ID assignment)
    let mut image_paths: Vec<String> = Vec::new();
    let mut seen_paths = std::collections::BTreeSet::new();

    for row in &rows {
        let p = match row {
            RetinanetRow::Annotation { path, .. } => path,
            RetinanetRow::Empty { path } => path,
        };
        if seen_paths.insert(p.clone()) {
            image_paths.push(p.clone());
        }
    }

    // Sort for deterministic image ID assignment
    image_paths.sort();

    // Resolve image dimensions from disk, cached per-image
    let mut dim_cache: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    for img_path in &image_paths {
        let dims = resolve_image_dimensions(base_dir, img_path, source_path)?;
        dim_cache.insert(img_path.clone(), dims);
    }

    // Build image map: path -> ImageId
    let image_map: BTreeMap<String, ImageId> = image_paths
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), ImageId::new((i + 1) as u64)))
        .collect();

    // Build images
    let images: Vec<Image> = image_paths
        .iter()
        .map(|p| {
            let id = image_map[p];
            let (width, height) = dim_cache[p];
            Image::new(id, p.clone(), width, height)
        })
        .collect();

    // Collect unique category names (BTreeSet gives sorted + dedup directly)
    let category_names: BTreeSet<String> = rows
        .iter()
        .filter_map(|row| match row {
            RetinanetRow::Annotation { class_name, .. } => Some(class_name.clone()),
            RetinanetRow::Empty { .. } => None,
        })
        .collect();

    let category_map: BTreeMap<String, CategoryId> = category_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), CategoryId::new((i + 1) as u64)))
        .collect();

    let categories: Vec<Category> = category_names
        .iter()
        .map(|name| {
            let id = category_map[name];
            Category::new(id, name.clone())
        })
        .collect();

    // Build annotations (row order for stable IDs)
    let mut annotations = Vec::new();
    let mut ann_id_counter: u64 = 1;

    for row in rows {
        if let RetinanetRow::Annotation {
            path,
            x1,
            y1,
            x2,
            y2,
            class_name,
        } = row
        {
            let image_id = image_map[&path];
            let category_id = category_map[&class_name];
            let bbox = BBoxXYXY::<Pixel>::from_xyxy(x1, y1, x2, y2);

            annotations.push(Annotation::new(
                AnnotationId::new(ann_id_counter),
                image_id,
                category_id,
                bbox,
            ));
            ann_id_counter += 1;
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

// ============================================================================
// Image dimension resolution
// ============================================================================

/// Resolves image dimensions by probing the filesystem.
///
/// If `image_ref` is an absolute path, uses it directly. Otherwise resolves
/// against `base_dir`. Caches should be maintained by the caller.
fn resolve_image_dimensions(
    base_dir: &Path,
    image_ref: &str,
    source_path: &Path,
) -> Result<(u32, u32), PanlabelError> {
    let image_path = if Path::new(image_ref).is_absolute() {
        PathBuf::from(image_ref)
    } else {
        base_dir.join(image_ref)
    };

    let size = imagesize::size(&image_path).map_err(|source| {
        // If the file doesn't exist at all, report it as not found
        if !image_path.exists() {
            return PanlabelError::RetinanetImageNotFound {
                path: source_path.to_path_buf(),
                image_ref: image_ref.to_string(),
            };
        }
        PanlabelError::RetinanetImageDimensionRead {
            path: image_path.clone(),
            source,
        }
    })?;

    Ok((size.width as u32, size.height as u32))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Writer / to_retinanet_csv_string tests ---

    #[test]
    fn test_write_annotated_images() {
        let dataset = Dataset {
            images: vec![
                Image::new(1u64, "img_a.jpg", 640, 480),
                Image::new(2u64, "img_b.jpg", 800, 600),
            ],
            categories: vec![Category::new(1u64, "cat"), Category::new(2u64, "dog")],
            annotations: vec![
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 200.0),
                ),
                Annotation::new(
                    2u64,
                    1u64,
                    2u64,
                    BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 150.0, 250.0),
                ),
                Annotation::new(
                    3u64,
                    2u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(5.0, 10.0, 55.0, 110.0),
                ),
            ],
            ..Default::default()
        };

        let csv = to_retinanet_csv_string(&dataset).expect("serialize failed");
        let lines: Vec<&str> = csv.lines().collect();

        // 3 annotation rows, no header
        assert_eq!(lines.len(), 3);

        // Images sorted by file_name: img_a first, then img_b
        assert!(lines[0].starts_with("img_a.jpg,"));
        assert!(lines[1].starts_with("img_a.jpg,"));
        assert!(lines[2].starts_with("img_b.jpg,"));

        // Annotations within img_a sorted by ann.id: 1 then 2
        assert!(lines[0].contains(",cat"));
        assert!(lines[1].contains(",dog"));
    }

    #[test]
    fn test_write_unannotated_image() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "empty.jpg", 100, 100)],
            categories: vec![],
            annotations: vec![],
            ..Default::default()
        };

        let csv = to_retinanet_csv_string(&dataset).expect("serialize failed");
        let lines: Vec<&str> = csv.lines().collect();

        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "empty.jpg,,,,,");
    }

    #[test]
    fn test_write_mixed_annotated_and_unannotated() {
        let dataset = Dataset {
            images: vec![
                Image::new(1u64, "annotated.jpg", 640, 480),
                Image::new(2u64, "empty.jpg", 100, 100),
            ],
            categories: vec![Category::new(1u64, "person")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 200.0),
            )],
            ..Default::default()
        };

        let csv = to_retinanet_csv_string(&dataset).expect("serialize failed");
        let lines: Vec<&str> = csv.lines().collect();

        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("annotated.jpg,10"));
        assert_eq!(lines[1], "empty.jpg,,,,,");
    }

    #[test]
    fn test_write_deterministic_order() {
        // Images out of alphabetical order, annotations out of ID order
        let dataset = Dataset {
            images: vec![
                Image::new(2u64, "z.jpg", 100, 100),
                Image::new(1u64, "a.jpg", 100, 100),
            ],
            categories: vec![Category::new(1u64, "obj")],
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
                    BBoxXYXY::<Pixel>::from_xyxy(5.0, 5.0, 15.0, 15.0),
                ),
            ],
            ..Default::default()
        };

        let csv = to_retinanet_csv_string(&dataset).expect("serialize failed");
        let lines: Vec<&str> = csv.lines().collect();

        // a.jpg first (ann IDs 1, 2), then z.jpg (ann ID 3)
        assert!(lines[0].starts_with("a.jpg,"));
        assert!(lines[1].starts_with("a.jpg,"));
        assert!(lines[2].starts_with("z.jpg,"));
    }

    #[test]
    fn test_write_missing_category_error() {
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

        let result = to_retinanet_csv_string(&dataset);
        assert!(result.is_err());
    }

    // --- CSV parsing tests (parse_csv_rows) ---

    #[test]
    fn test_parse_annotation_row() {
        let csv = "img.jpg,10,20,100,200,person\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");

        assert_eq!(rows.len(), 1);
        match &rows[0] {
            RetinanetRow::Annotation {
                path,
                x1,
                y1,
                x2,
                y2,
                class_name,
            } => {
                assert_eq!(path, "img.jpg");
                assert_eq!(*x1, 10.0);
                assert_eq!(*y1, 20.0);
                assert_eq!(*x2, 100.0);
                assert_eq!(*y2, 200.0);
                assert_eq!(class_name, "person");
            }
            RetinanetRow::Empty { .. } => panic!("expected annotation row"),
        }
    }

    #[test]
    fn test_parse_empty_row() {
        let csv = "img.jpg,,,,,\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");

        assert_eq!(rows.len(), 1);
        match &rows[0] {
            RetinanetRow::Empty { path } => assert_eq!(path, "img.jpg"),
            RetinanetRow::Annotation { .. } => panic!("expected empty row"),
        }
    }

    #[test]
    fn test_parse_header_skipped() {
        let csv = "path,x1,y1,x2,y2,class_name\nimg.jpg,10,20,100,200,cat\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");

        // Header should be skipped, leaving only the data row
        assert_eq!(rows.len(), 1);
        match &rows[0] {
            RetinanetRow::Annotation { path, .. } => assert_eq!(path, "img.jpg"),
            RetinanetRow::Empty { .. } => panic!("expected annotation row"),
        }
    }

    #[test]
    fn test_parse_no_header() {
        let csv = "img.jpg,10,20,100,200,cat\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");

        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_parse_partial_row_rejected() {
        // x1 present but y1 empty -- partial
        let csv = "img.jpg,10,,100,200,cat\n";
        let result = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"));

        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("partial annotation row"));
    }

    #[test]
    fn test_parse_empty_path_rejected() {
        let csv = ",10,20,100,200,cat\n";
        let result = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"));

        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("empty path field"));
    }

    #[test]
    fn test_parse_invalid_coordinate_rejected() {
        let csv = "img.jpg,abc,20,100,200,cat\n";
        let result = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"));

        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("invalid x1 value"));
    }

    #[test]
    fn test_parse_multiple_images() {
        let csv = "a.jpg,10,20,100,200,cat\na.jpg,50,60,150,250,dog\nb.jpg,,,,,\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");

        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_parse_float_coordinates() {
        let csv = "img.jpg,10.5,20.3,100.7,200.9,person\n";
        let rows = parse_csv_rows(std::io::Cursor::new(csv.as_bytes()), Path::new("test.csv"))
            .expect("parse failed");

        match &rows[0] {
            RetinanetRow::Annotation { x1, y1, x2, y2, .. } => {
                assert!((x1 - 10.5).abs() < 1e-9);
                assert!((y1 - 20.3).abs() < 1e-9);
                assert!((x2 - 100.7).abs() < 1e-9);
                assert!((y2 - 200.9).abs() < 1e-9);
            }
            _ => panic!("expected annotation row"),
        }
    }
}
