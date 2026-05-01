//! Microsoft VoTT CSV reader and writer.
//!
//! VoTT CSV is a headered six-column object-detection export:
//! `image,xmin,ymin,xmax,ymax,label`.
//!
//! Coordinates are absolute pixel-space XYXY values. The format does not store
//! image dimensions, so the reader probes the referenced image files from disk.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::io_adapter_common::{is_safe_relative_image_ref, normalize_path_separators};
use super::model::{Annotation, Category, Dataset, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

const HEADER: [&str; 6] = ["image", "xmin", "ymin", "xmax", "ymax", "label"];

#[derive(Debug)]
struct VottCsvRow {
    image: String,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    label: String,
}

pub fn read_vott_csv(path: &Path) -> Result<Dataset, PanlabelError> {
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let file = File::open(path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let rows = parse_csv_rows(reader, path)?;
    vott_rows_to_ir(rows, base_dir, path)
}

pub fn write_vott_csv(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let csv_string = to_vott_csv_string(dataset)?;
    let file = File::create(path).map_err(PanlabelError::Io)?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(csv_string.as_bytes())
        .map_err(PanlabelError::Io)?;
    writer.flush().map_err(PanlabelError::Io)?;
    Ok(())
}

pub fn from_vott_csv_str_with_base_dir(
    csv: &str,
    base_dir: &Path,
) -> Result<Dataset, PanlabelError> {
    let dummy_path = base_dir.join("<string>");
    let rows = parse_csv_rows(csv.as_bytes(), &dummy_path)?;
    vott_rows_to_ir(rows, base_dir, &dummy_path)
}

pub fn to_vott_csv_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    let dummy_path = Path::new("<string>");
    let image_lookup: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();
    let category_lookup: BTreeMap<CategoryId, &Category> =
        dataset.categories.iter().map(|cat| (cat.id, cat)).collect();

    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());
    csv_writer
        .write_record(HEADER)
        .map_err(|source| PanlabelError::VottCsvWrite {
            path: dummy_path.to_path_buf(),
            source,
        })?;

    let mut rows = Vec::with_capacity(dataset.annotations.len());
    for ann in &dataset.annotations {
        let image = image_lookup.get(&ann.image_id).copied().ok_or_else(|| {
            PanlabelError::VottCsvInvalid {
                path: dummy_path.to_path_buf(),
                message: format!(
                    "Annotation {} references non-existent image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            }
        })?;
        rows.push((image, ann));
    }
    rows.sort_by(|(image_a, ann_a), (image_b, ann_b)| {
        image_a
            .file_name
            .cmp(&image_b.file_name)
            .then_with(|| ann_a.id.cmp(&ann_b.id))
    });

    for (image, ann) in rows {
        let category =
            category_lookup
                .get(&ann.category_id)
                .ok_or_else(|| PanlabelError::VottCsvInvalid {
                    path: dummy_path.to_path_buf(),
                    message: format!(
                        "Annotation {} references non-existent category {}",
                        ann.id.as_u64(),
                        ann.category_id.as_u64()
                    ),
                })?;

        csv_writer
            .write_record([
                &image.file_name,
                &ann.bbox.xmin().to_string(),
                &ann.bbox.ymin().to_string(),
                &ann.bbox.xmax().to_string(),
                &ann.bbox.ymax().to_string(),
                &category.name,
            ])
            .map_err(|source| PanlabelError::VottCsvWrite {
                path: dummy_path.to_path_buf(),
                source,
            })?;
    }

    let bytes = csv_writer
        .into_inner()
        .map_err(|e| PanlabelError::Io(e.into_error()))?;
    String::from_utf8(bytes).map_err(|e| PanlabelError::VottCsvInvalid {
        path: dummy_path.to_path_buf(),
        message: format!("Invalid UTF-8 in output: {e}"),
    })
}

#[cfg(feature = "fuzzing")]
pub fn parse_vott_csv_slice(bytes: &[u8]) -> Result<(), csv::Error> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(bytes);
    for result in rdr.records() {
        let _record = result?;
    }
    Ok(())
}

fn parse_csv_rows<R: std::io::Read>(
    reader: R,
    source_path: &Path,
) -> Result<Vec<VottCsvRow>, PanlabelError> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(reader);
    let mut records = csv_reader.records();

    let header = records
        .next()
        .ok_or_else(|| PanlabelError::VottCsvInvalid {
            path: source_path.to_path_buf(),
            message: "CSV file is empty; expected header image,xmin,ymin,xmax,ymax,label"
                .to_string(),
        })?
        .map_err(|source| PanlabelError::VottCsvParse {
            path: source_path.to_path_buf(),
            source,
        })?;

    if header.len() != 6
        || !HEADER.iter().enumerate().all(|(i, expected)| {
            header
                .get(i)
                .map(|actual| actual.eq_ignore_ascii_case(expected))
                .unwrap_or(false)
        })
    {
        return Err(PanlabelError::VottCsvInvalid {
            path: source_path.to_path_buf(),
            message: format!(
                "expected header image,xmin,ymin,xmax,ymax,label, got {:?}",
                header.iter().collect::<Vec<_>>()
            ),
        });
    }

    let mut rows = Vec::new();
    for (idx, result) in records.enumerate() {
        let row_num = idx + 2;
        let record = result.map_err(|source| PanlabelError::VottCsvParse {
            path: source_path.to_path_buf(),
            source,
        })?;
        if record.len() != 6 {
            return Err(PanlabelError::VottCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {row_num}: expected 6 columns, got {}", record.len()),
            });
        }

        let image = record.get(0).unwrap_or("").to_string();
        let label = record.get(5).unwrap_or("").to_string();
        if image.is_empty() {
            return Err(PanlabelError::VottCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {row_num}: empty image field"),
            });
        }
        if label.is_empty() {
            return Err(PanlabelError::VottCsvInvalid {
                path: source_path.to_path_buf(),
                message: format!("row {row_num}: empty label field for image '{image}'"),
            });
        }

        let parse_f64 = |col_idx: usize, name: &str| -> Result<f64, PanlabelError> {
            let raw = record.get(col_idx).unwrap_or("");
            raw.parse::<f64>()
                .map_err(|_| PanlabelError::VottCsvInvalid {
                    path: source_path.to_path_buf(),
                    message: format!(
                        "row {row_num}: invalid {name} value '{raw}' for image '{image}'"
                    ),
                })
        };

        let xmin = parse_f64(1, "xmin")?;
        let ymin = parse_f64(2, "ymin")?;
        let xmax = parse_f64(3, "xmax")?;
        let ymax = parse_f64(4, "ymax")?;

        rows.push(VottCsvRow {
            image,
            xmin,
            ymin,
            xmax,
            ymax,
            label,
        });
    }

    Ok(rows)
}

fn vott_rows_to_ir(
    rows: Vec<VottCsvRow>,
    base_dir: &Path,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    let mut image_refs = BTreeSet::new();
    let mut label_set = BTreeSet::new();
    for row in &rows {
        validate_relative_image_ref(&row.image, source_path)?;
        image_refs.insert(row.image.clone());
        label_set.insert(row.label.clone());
    }

    let image_map: BTreeMap<String, ImageId> = image_refs
        .iter()
        .enumerate()
        .map(|(idx, image)| (image.clone(), ImageId::new((idx + 1) as u64)))
        .collect();
    let category_map: BTreeMap<String, CategoryId> = label_set
        .iter()
        .enumerate()
        .map(|(idx, label)| (label.clone(), CategoryId::new((idx + 1) as u64)))
        .collect();

    let mut images = Vec::new();
    for image_ref in &image_refs {
        let (width, height) = resolve_image_dimensions(base_dir, image_ref, source_path)?;
        images.push(Image::new(
            image_map[image_ref],
            normalize_path_separators(image_ref),
            width,
            height,
        ));
    }

    let categories: Vec<Category> = label_set
        .iter()
        .enumerate()
        .map(|(idx, label)| Category::new((idx + 1) as u64, label.clone()))
        .collect();

    let mut sorted_rows = rows;
    sorted_rows.sort_by(|a, b| {
        a.image
            .cmp(&b.image)
            .then_with(|| a.label.cmp(&b.label))
            .then_with(|| a.xmin.total_cmp(&b.xmin))
            .then_with(|| a.ymin.total_cmp(&b.ymin))
            .then_with(|| a.xmax.total_cmp(&b.xmax))
            .then_with(|| a.ymax.total_cmp(&b.ymax))
    });

    let annotations = sorted_rows
        .into_iter()
        .enumerate()
        .map(|(idx, row)| {
            Annotation::new(
                AnnotationId::new((idx + 1) as u64),
                image_map[&row.image],
                category_map[&row.label],
                BBoxXYXY::<Pixel>::from_xyxy(row.xmin, row.ymin, row.xmax, row.ymax),
            )
        })
        .collect();

    Ok(Dataset {
        images,
        categories,
        annotations,
        ..Default::default()
    })
}

fn validate_relative_image_ref(image_ref: &str, source_path: &Path) -> Result<(), PanlabelError> {
    if !is_safe_relative_image_ref(image_ref) {
        return Err(PanlabelError::VottCsvInvalid {
            path: source_path.to_path_buf(),
            message: format!(
                "image reference '{}' must be a relative path without parent-directory components",
                image_ref
            ),
        });
    }
    Ok(())
}

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
    let images_dir_path = base_dir.join("images").join(image_ref);

    if let Ok(size) = imagesize::size(&image_path) {
        return Ok((size.width as u32, size.height as u32));
    }
    if let Ok(size) = imagesize::size(&images_dir_path) {
        return Ok((size.width as u32, size.height as u32));
    }

    Err(PanlabelError::VottCsvImageNotFound {
        path: source_path.to_path_buf(),
        image_ref: image_ref.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_vott_header() {
        let result = parse_csv_rows("img.bmp,1,2,3,4,cat\n".as_bytes(), Path::new("test.csv"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected header"));
    }

    #[test]
    fn writes_header_and_rows() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "img.bmp", 100, 50)],
            categories: vec![Category::new(1u64, "cat")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(1.0, 2.0, 30.0, 40.0),
            )],
            ..Default::default()
        };
        let csv = to_vott_csv_string(&dataset).expect("serialize");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "image,xmin,ymin,xmax,ymax,label");
        assert_eq!(lines[1], "img.bmp,1,2,30,40,cat");
    }
}
