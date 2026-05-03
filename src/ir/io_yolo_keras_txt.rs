//! Shared YOLO Keras / YOLOv4 PyTorch absolute-coordinate TXT reader and writer.
//!
//! Both public formats use the same grammar:
//!
//! ```text
//! <image_ref> [xmin,ymin,xmax,ymax,class_id ...]
//! ```
//!
//! Coordinates are absolute pixel-space XYXY values. A row containing only the
//! image reference represents an unannotated image and is preserved on write.
//! Class IDs are zero-based in the source file and map to panlabel category IDs
//! by adding one.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::io_adapter_common::has_extension;
use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};
use crate::error::PanlabelError;

const CLASS_FILE_CANDIDATES: [&str; 4] = [
    "classes.txt",
    "class_names.txt",
    "classes.names",
    "obj.names",
];
const IMAGE_SUBDIR: &str = "images";

pub(crate) const YOLO_KERAS_ANNOTATION_CANDIDATES: [&str; 5] = [
    "yolo_keras.txt",
    "yolo-keras.txt",
    "annotations.txt",
    "train_annotations.txt",
    "train.txt",
];

pub(crate) const YOLOV4_PYTORCH_ANNOTATION_CANDIDATES: [&str; 6] = [
    "yolov4_pytorch.txt",
    "yolov4-pytorch.txt",
    "yolov4_train.txt",
    "train_annotation.txt",
    "train_annotations.txt",
    "train.txt",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum YoloKerasTxtProfile {
    YoloKeras,
    YoloV4Pytorch,
}

impl YoloKerasTxtProfile {
    fn public_name(self) -> &'static str {
        match self {
            Self::YoloKeras => "yolo-keras",
            Self::YoloV4Pytorch => "yolov4-pytorch",
        }
    }

    fn canonical_annotation_file(self) -> &'static str {
        match self {
            Self::YoloKeras => "yolo_keras.txt",
            Self::YoloV4Pytorch => "yolov4_pytorch.txt",
        }
    }

    fn annotation_candidates(self) -> &'static [&'static str] {
        match self {
            Self::YoloKeras => &YOLO_KERAS_ANNOTATION_CANDIDATES,
            Self::YoloV4Pytorch => &YOLOV4_PYTORCH_ANNOTATION_CANDIDATES,
        }
    }
}

#[derive(Clone, Debug)]
struct ParsedBox {
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    class_id: usize,
}

#[derive(Clone, Debug)]
struct ParsedRow {
    image_ref: String,
    boxes: Vec<ParsedBox>,
}

#[derive(Clone, Debug)]
struct InputLayout {
    annotation_path: PathBuf,
    base_dir: PathBuf,
    class_file: Option<PathBuf>,
}

/// Read a YOLO Keras TXT annotation file or directory into IR.
pub fn read_yolo_keras_txt(path: &Path) -> Result<Dataset, PanlabelError> {
    read_shared(path, YoloKerasTxtProfile::YoloKeras)
}

/// Write an IR dataset as YOLO Keras TXT.
pub fn write_yolo_keras_txt(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    write_shared(path, dataset, YoloKerasTxtProfile::YoloKeras)
}

/// Read a YOLOv4 PyTorch TXT annotation file or directory into IR.
pub fn read_yolov4_pytorch_txt(path: &Path) -> Result<Dataset, PanlabelError> {
    read_shared(path, YoloKerasTxtProfile::YoloV4Pytorch)
}

/// Write an IR dataset as YOLOv4 PyTorch TXT.
pub fn write_yolov4_pytorch_txt(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    write_shared(path, dataset, YoloKerasTxtProfile::YoloV4Pytorch)
}

/// Sniff whether a file looks like the shared absolute-coordinate TXT grammar.
///
/// This is intentionally grammar-only. The two public formats are identical at
/// the row level, so CLI auto-detection decides whether a file name is specific
/// enough to select one format or should remain ambiguous.
pub fn looks_like_yolo_keras_txt_file(path: &Path) -> Result<bool, PanlabelError> {
    if !path.is_file() {
        return Ok(false);
    }
    let content = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    looks_like_yolo_keras_txt_content(&content, path)
}

pub fn looks_like_yolo_keras_txt_content(
    content: &str,
    source_path: &Path,
) -> Result<bool, PanlabelError> {
    let mut saw_row = false;
    for (line_idx, raw_line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        saw_row = true;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() || tokens[0].contains(',') {
            return Ok(false);
        }
        for token in tokens.iter().skip(1) {
            if parse_box_token(token, source_path, line_num).is_err() {
                return Ok(false);
            }
        }
    }
    Ok(saw_row)
}

#[cfg(feature = "fuzzing")]
pub fn parse_yolo_keras_txt_str(input: &str) -> Result<(), PanlabelError> {
    let _rows = parse_annotation_txt(input, Path::new("<fuzz>"))?;
    Ok(())
}

/// Serialize to the shared TXT content string. Used by tests and both writers.
pub fn to_yolo_keras_txt_string(dataset: &Dataset) -> Result<String, PanlabelError> {
    format_annotation_lines(dataset, Path::new("<string>"))
}

fn read_shared(path: &Path, profile: YoloKerasTxtProfile) -> Result<Dataset, PanlabelError> {
    let layout = discover_input_layout(path, profile)?;
    let content = fs::read_to_string(&layout.annotation_path).map_err(PanlabelError::Io)?;
    let rows = parse_annotation_txt(&content, &layout.annotation_path)?;
    rows_to_ir(rows, &layout)
}

fn write_shared(
    path: &Path,
    dataset: &Dataset,
    profile: YoloKerasTxtProfile,
) -> Result<(), PanlabelError> {
    let (annotation_path, class_path) = output_paths(path, profile)?;
    if let Some(parent) = annotation_path.parent() {
        fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
    }

    let annotation_text = format_annotation_lines(dataset, &annotation_path)?;
    let class_text = format_class_file(dataset);

    let mut annotation_file = fs::File::create(&annotation_path).map_err(PanlabelError::Io)?;
    annotation_file
        .write_all(annotation_text.as_bytes())
        .map_err(PanlabelError::Io)?;
    annotation_file.flush().map_err(PanlabelError::Io)?;

    if let Some(parent) = class_path.parent() {
        fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
    }
    fs::write(class_path, class_text).map_err(PanlabelError::Io)?;
    Ok(())
}

fn discover_input_layout(
    path: &Path,
    profile: YoloKerasTxtProfile,
) -> Result<InputLayout, PanlabelError> {
    if path.is_file() {
        let base_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let class_file = find_class_file(&base_dir);
        return Ok(InputLayout {
            annotation_path: path.to_path_buf(),
            base_dir,
            class_file,
        });
    }

    if !path.is_dir() {
        return Err(PanlabelError::YoloKerasTxtInvalid {
            path: path.to_path_buf(),
            message: format!(
                "{} input must be an annotation .txt file or a directory containing one",
                profile.public_name()
            ),
        });
    }

    let matches: Vec<PathBuf> = profile
        .annotation_candidates()
        .iter()
        .map(|name| path.join(name))
        .filter(|candidate| candidate.is_file())
        .collect();

    let annotation_path = match matches.as_slice() {
        [single] => single.clone(),
        [] => {
            return Err(PanlabelError::YoloKerasTxtInvalid {
                path: path.to_path_buf(),
                message: format!(
                    "could not find {} annotation file; expected one of: {}",
                    profile.public_name(),
                    profile.annotation_candidates().join(", ")
                ),
            });
        }
        many => {
            return Err(PanlabelError::YoloKerasTxtInvalid {
                path: path.to_path_buf(),
                message: format!(
                    "multiple candidate annotation files found: {}. Pass the desired .txt file directly.",
                    many.iter()
                        .map(|p| p.file_name().and_then(|n| n.to_str()).unwrap_or("<unknown>"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            });
        }
    };

    Ok(InputLayout {
        annotation_path,
        base_dir: path.to_path_buf(),
        class_file: find_class_file(path),
    })
}

fn find_class_file(base_dir: &Path) -> Option<PathBuf> {
    CLASS_FILE_CANDIDATES
        .iter()
        .map(|name| base_dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn output_paths(
    path: &Path,
    profile: YoloKerasTxtProfile,
) -> Result<(PathBuf, PathBuf), PanlabelError> {
    let looks_like_txt = has_extension(path, "txt");

    if looks_like_txt {
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        Ok((path.to_path_buf(), base_dir.join("classes.txt")))
    } else {
        fs::create_dir_all(path).map_err(PanlabelError::Io)?;
        Ok((
            path.join(profile.canonical_annotation_file()),
            path.join("classes.txt"),
        ))
    }
}

fn parse_annotation_txt(
    content: &str,
    source_path: &Path,
) -> Result<Vec<ParsedRow>, PanlabelError> {
    let mut rows = Vec::new();
    for (line_idx, raw_line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        let image_ref = tokens[0].to_string();
        if image_ref.contains(',') {
            return Err(PanlabelError::YoloKerasTxtParse {
                path: source_path.to_path_buf(),
                line: line_num,
                message: "first token must be an image reference, not a comma-separated box"
                    .to_string(),
            });
        }

        let boxes = tokens
            .iter()
            .skip(1)
            .map(|token| parse_box_token(token, source_path, line_num))
            .collect::<Result<Vec<_>, _>>()?;

        rows.push(ParsedRow { image_ref, boxes });
    }
    Ok(rows)
}

fn parse_box_token(
    token: &str,
    source_path: &Path,
    line: usize,
) -> Result<ParsedBox, PanlabelError> {
    let fields: Vec<&str> = token.split(',').collect();
    if fields.len() != 5 {
        return Err(PanlabelError::YoloKerasTxtParse {
            path: source_path.to_path_buf(),
            line,
            message: format!(
                "box token '{token}' must have exactly 5 comma-separated fields: xmin,ymin,xmax,ymax,class_id"
            ),
        });
    }

    let parse_f64 = |idx: usize, label: &str| -> Result<f64, PanlabelError> {
        let value = fields[idx]
            .parse::<f64>()
            .map_err(|_| PanlabelError::YoloKerasTxtParse {
                path: source_path.to_path_buf(),
                line,
                message: format!(
                    "invalid {label} value '{}' in box token '{token}'",
                    fields[idx]
                ),
            })?;
        if !value.is_finite() {
            return Err(PanlabelError::YoloKerasTxtParse {
                path: source_path.to_path_buf(),
                line,
                message: format!("{label} value '{}' is not finite", fields[idx]),
            });
        }
        Ok(value)
    };

    let xmin = parse_f64(0, "xmin")?;
    let ymin = parse_f64(1, "ymin")?;
    let xmax = parse_f64(2, "xmax")?;
    let ymax = parse_f64(3, "ymax")?;
    let class_id = fields[4]
        .parse::<usize>()
        .map_err(|_| PanlabelError::YoloKerasTxtParse {
            path: source_path.to_path_buf(),
            line,
            message: format!(
                "invalid class_id value '{}' in box token '{token}'",
                fields[4]
            ),
        })?;

    if xmax < xmin || ymax < ymin {
        return Err(PanlabelError::YoloKerasTxtParse {
            path: source_path.to_path_buf(),
            line,
            message: format!("malformed box '{token}': expected xmin <= xmax and ymin <= ymax"),
        });
    }

    Ok(ParsedBox {
        xmin,
        ymin,
        xmax,
        ymax,
        class_id,
    })
}

fn rows_to_ir(rows: Vec<ParsedRow>, layout: &InputLayout) -> Result<Dataset, PanlabelError> {
    let mut image_refs = BTreeSet::new();
    let mut class_ids = BTreeSet::new();
    for row in &rows {
        image_refs.insert(row.image_ref.clone());
        for bbox in &row.boxes {
            class_ids.insert(bbox.class_id);
        }
    }

    let class_names = match &layout.class_file {
        Some(path) => read_class_names(path)?,
        None => Vec::new(),
    };

    let max_class_id = class_ids
        .iter()
        .copied()
        .chain((0..class_names.len()).collect::<Vec<_>>())
        .max();

    let categories: Vec<Category> = match max_class_id {
        Some(max_id) => (0..=max_id)
            .map(|class_id| {
                let name = class_names
                    .get(class_id)
                    .filter(|name| !name.trim().is_empty())
                    .cloned()
                    .unwrap_or_else(|| format!("class_{class_id}"));
                Category::new(CategoryId::new(class_id as u64 + 1), name)
            })
            .collect(),
        None => Vec::new(),
    };

    let mut dimensions = BTreeMap::new();
    for image_ref in &image_refs {
        dimensions.insert(
            image_ref.clone(),
            resolve_image_dimensions(&layout.base_dir, image_ref, &layout.annotation_path)?,
        );
    }

    let images: Vec<Image> = image_refs
        .iter()
        .enumerate()
        .map(|(idx, image_ref)| {
            let (width, height) = dimensions[image_ref];
            Image::new((idx + 1) as u64, image_ref.clone(), width, height)
        })
        .collect();

    let image_id_by_ref: BTreeMap<String, ImageId> = images
        .iter()
        .map(|image| (image.file_name.clone(), image.id))
        .collect();

    let mut annotations = Vec::new();
    let mut next_ann_id = 1u64;
    for row in rows {
        let image_id = image_id_by_ref[&row.image_ref];
        for bbox in row.boxes {
            annotations.push(Annotation::new(
                AnnotationId::new(next_ann_id),
                image_id,
                CategoryId::new(bbox.class_id as u64 + 1),
                BBoxXYXY::<Pixel>::from_xyxy(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax),
            ));
            next_ann_id += 1;
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

fn read_class_names(path: &Path) -> Result<Vec<String>, PanlabelError> {
    let content = fs::read_to_string(path).map_err(PanlabelError::Io)?;
    let names = content
        .lines()
        .map(str::trim)
        .map(ToOwned::to_owned)
        .collect();
    Ok(names)
}

fn resolve_image_dimensions(
    base_dir: &Path,
    image_ref: &str,
    source_path: &Path,
) -> Result<(u32, u32), PanlabelError> {
    let mut candidates = Vec::new();
    let image_ref_path = Path::new(image_ref);
    if image_ref_path.is_absolute() {
        candidates.push(image_ref_path.to_path_buf());
    } else {
        candidates.push(base_dir.join(image_ref_path));
        candidates.push(base_dir.join(IMAGE_SUBDIR).join(image_ref_path));
    }

    let existing = candidates.iter().find(|candidate| candidate.is_file());
    let Some(image_path) = existing else {
        return Err(PanlabelError::YoloKerasTxtImageNotFound {
            path: source_path.to_path_buf(),
            image_ref: image_ref.to_string(),
            searched: candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
        });
    };

    let size = imagesize::size(image_path).map_err(|source| {
        PanlabelError::YoloKerasTxtImageDimensionRead {
            path: image_path.clone(),
            source,
        }
    })?;

    Ok((size.width as u32, size.height as u32))
}

fn format_annotation_lines(dataset: &Dataset, error_path: &Path) -> Result<String, PanlabelError> {
    let image_by_id: BTreeMap<ImageId, &Image> = dataset
        .images
        .iter()
        .map(|image| (image.id, image))
        .collect();
    let mut categories_sorted: Vec<&Category> = dataset.categories.iter().collect();
    categories_sorted.sort_by_key(|category| category.id);
    let class_id_by_category: BTreeMap<CategoryId, usize> = categories_sorted
        .iter()
        .enumerate()
        .map(|(idx, category)| (category.id, idx))
        .collect();

    let mut annotations_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        if !image_by_id.contains_key(&ann.image_id) {
            return Err(PanlabelError::YoloKerasTxtWriteError {
                path: error_path.to_path_buf(),
                message: format!(
                    "annotation {} references missing image {}",
                    ann.id.as_u64(),
                    ann.image_id.as_u64()
                ),
            });
        }
        if !class_id_by_category.contains_key(&ann.category_id) {
            return Err(PanlabelError::YoloKerasTxtWriteError {
                path: error_path.to_path_buf(),
                message: format!(
                    "annotation {} references missing category {}",
                    ann.id.as_u64(),
                    ann.category_id.as_u64()
                ),
            });
        }
        annotations_by_image
            .entry(ann.image_id)
            .or_default()
            .push(ann);
    }

    let mut images_sorted: Vec<&Image> = dataset.images.iter().collect();
    images_sorted.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    let mut output = String::new();
    for image in images_sorted {
        validate_write_image_ref(&image.file_name, error_path)?;
        let mut line = image.file_name.clone();
        let mut anns = annotations_by_image.remove(&image.id).unwrap_or_default();
        anns.sort_by_key(|ann| ann.id);

        for ann in anns {
            let class_id = class_id_by_category[&ann.category_id];
            line.push(' ');
            line.push_str(&format!(
                "{},{},{},{},{}",
                format_number(ann.bbox.xmin()),
                format_number(ann.bbox.ymin()),
                format_number(ann.bbox.xmax()),
                format_number(ann.bbox.ymax()),
                class_id,
            ));
        }

        output.push_str(&line);
        output.push('\n');
    }

    Ok(output)
}

fn validate_write_image_ref(image_ref: &str, error_path: &Path) -> Result<(), PanlabelError> {
    if image_ref.is_empty() || image_ref.contains(',') || image_ref.chars().any(char::is_whitespace)
    {
        return Err(PanlabelError::YoloKerasTxtWriteError {
            path: error_path.to_path_buf(),
            message: format!(
                "image file_name '{image_ref}' cannot be represented in YOLO Keras-style TXT because image_ref may not be empty or contain whitespace/commas"
            ),
        });
    }
    Ok(())
}

fn format_class_file(dataset: &Dataset) -> String {
    let mut categories_sorted: Vec<&Category> = dataset.categories.iter().collect();
    categories_sorted.sort_by_key(|category| category.id);
    let mut output = String::new();
    for category in categories_sorted {
        output.push_str(&category.name);
        output.push('\n');
    }
    output
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_empty_and_annotated_rows() {
        let txt = "img1.bmp 10,20,30,40,0 1,2,3,4,1\nempty.bmp\n";
        let rows = parse_annotation_txt(txt, Path::new("train.txt")).expect("parse");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].boxes.len(), 2);
        assert_eq!(rows[1].boxes.len(), 0);
    }

    #[test]
    fn rejects_malformed_box() {
        let err = parse_annotation_txt("img.bmp 30,20,10,40,0\n", Path::new("train.txt"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("malformed box"));
    }

    #[test]
    fn writer_is_deterministic_and_preserves_empty_rows() {
        let dataset = Dataset {
            images: vec![
                Image::new(2u64, "z.bmp", 10, 10),
                Image::new(1u64, "a.bmp", 10, 10),
            ],
            categories: vec![Category::new(10u64, "cat")],
            annotations: vec![Annotation::new(
                2u64,
                2u64,
                10u64,
                BBoxXYXY::<Pixel>::from_xyxy(1.0, 2.0, 3.0, 4.0),
            )],
            ..Default::default()
        };

        let txt = to_yolo_keras_txt_string(&dataset).expect("write string");
        assert_eq!(txt, "a.bmp\nz.bmp 1,2,3,4,0\n");
    }
}
