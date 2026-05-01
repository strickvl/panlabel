//! IBM Cloud Annotations JSON reader and writer.
//!
//! Cloud Annotations localization exports use a `_annotations.json` file with
//! normalized bounding boxes under an image-keyed `annotations` object:
//!
//! ```json
//! {
//!   "version": "1.0",
//!   "type": "localization",
//!   "labels": ["cat"],
//!   "annotations": {
//!     "image.jpg": [{ "x": 0.1, "y": 0.2, "x2": 0.4, "y2": 0.6, "label": "cat" }]
//!   }
//! }
//! ```
//!
//! Coordinates are normalized. The reader resolves image dimensions from local
//! image files so the IR can store canonical pixel-space XYXY boxes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::io_adapter_common::{
    is_safe_relative_image_ref, normalize_path_separators, write_images_readme,
};
use super::model::{Annotation, Category, Dataset, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Normalized};
use crate::error::PanlabelError;

const ANNOTATIONS_FILE_NAME: &str = "_annotations.json";
const IMAGES_README: &str = "Panlabel wrote _annotations.json only. Copy your image files here if a downstream tool expects a self-contained Cloud Annotations directory.\n";

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CloudAnnotationsFile {
    pub(crate) version: String,
    #[serde(rename = "type")]
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) labels: Vec<String>,
    #[serde(default)]
    pub(crate) annotations: BTreeMap<String, Vec<CloudAnnotation>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CloudAnnotation {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) x2: f64,
    pub(crate) y2: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<String>,
    pub(crate) label: String,
}

pub fn read_cloud_annotations_json(path: &Path) -> Result<Dataset, PanlabelError> {
    let annotation_path = annotation_file_path(path);
    let base_dir = annotation_path.parent().unwrap_or_else(|| Path::new("."));

    let file = File::open(&annotation_path).map_err(PanlabelError::Io)?;
    let reader = BufReader::new(file);
    let parsed: CloudAnnotationsFile = serde_json::from_reader(reader).map_err(|source| {
        PanlabelError::CloudAnnotationsJsonParse {
            path: annotation_path.clone(),
            source,
        }
    })?;

    cloud_annotations_to_ir(parsed, base_dir, &annotation_path)
}

pub fn write_cloud_annotations_json(path: &Path, dataset: &Dataset) -> Result<(), PanlabelError> {
    let annotation_path = output_annotation_file_path(path);
    if let Some(parent) = annotation_path.parent() {
        fs::create_dir_all(parent).map_err(PanlabelError::Io)?;
    }

    if path.extension().and_then(|ext| ext.to_str()).is_none() || path.is_dir() {
        write_images_readme(
            annotation_path.parent().unwrap_or_else(|| Path::new(".")),
            IMAGES_README,
        )?;
    }

    let file = File::create(&annotation_path).map_err(PanlabelError::Io)?;
    let writer = BufWriter::new(file);
    let output = ir_to_cloud_annotations(dataset);

    serde_json::to_writer_pretty(writer, &output).map_err(|source| {
        PanlabelError::CloudAnnotationsJsonWrite {
            path: annotation_path,
            source,
        }
    })
}

pub fn from_cloud_annotations_str_with_base_dir(
    json: &str,
    base_dir: &Path,
) -> Result<Dataset, PanlabelError> {
    let parsed: CloudAnnotationsFile =
        serde_json::from_str(json).map_err(|source| PanlabelError::CloudAnnotationsJsonParse {
            path: base_dir.join(ANNOTATIONS_FILE_NAME),
            source,
        })?;
    cloud_annotations_to_ir(parsed, base_dir, &base_dir.join(ANNOTATIONS_FILE_NAME))
}

pub fn to_cloud_annotations_string(dataset: &Dataset) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&ir_to_cloud_annotations(dataset))
}

#[cfg(feature = "fuzzing")]
pub fn parse_cloud_annotations_slice(bytes: &[u8]) -> Result<(), serde_json::Error> {
    let _parsed: CloudAnnotationsFile = serde_json::from_slice(bytes)?;
    Ok(())
}

fn annotation_file_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join(ANNOTATIONS_FILE_NAME)
    } else {
        path.to_path_buf()
    }
}

fn output_annotation_file_path(path: &Path) -> PathBuf {
    if path.extension().and_then(|ext| ext.to_str()).is_some() {
        path.to_path_buf()
    } else {
        path.join(ANNOTATIONS_FILE_NAME)
    }
}

fn cloud_annotations_to_ir(
    parsed: CloudAnnotationsFile,
    base_dir: &Path,
    source_path: &Path,
) -> Result<Dataset, PanlabelError> {
    if parsed.kind != "localization" {
        return Err(PanlabelError::CloudAnnotationsJsonInvalid {
            path: source_path.to_path_buf(),
            message: format!(
                "unsupported type '{}' (expected 'localization')",
                parsed.kind
            ),
        });
    }

    let mut labels = Vec::new();
    let mut seen_labels = BTreeSet::new();
    for label in parsed.labels {
        if label.is_empty() {
            return Err(PanlabelError::CloudAnnotationsJsonInvalid {
                path: source_path.to_path_buf(),
                message: "labels array contains an empty label".to_string(),
            });
        }
        if seen_labels.insert(label.clone()) {
            labels.push(label);
        }
    }

    let mut extra_labels = BTreeSet::new();
    for (image_ref, anns) in &parsed.annotations {
        if image_ref.is_empty() {
            return Err(PanlabelError::CloudAnnotationsJsonInvalid {
                path: source_path.to_path_buf(),
                message: "annotations object contains an empty image key".to_string(),
            });
        }
        for ann in anns {
            if ann.label.is_empty() {
                return Err(PanlabelError::CloudAnnotationsJsonInvalid {
                    path: source_path.to_path_buf(),
                    message: format!("empty annotation label in image '{image_ref}'"),
                });
            }
            if !seen_labels.contains(&ann.label) {
                extra_labels.insert(ann.label.clone());
            }
        }
    }
    for label in extra_labels {
        seen_labels.insert(label.clone());
        labels.push(label);
    }

    let label_to_cat: BTreeMap<String, CategoryId> = labels
        .iter()
        .enumerate()
        .map(|(idx, label)| (label.clone(), CategoryId::new((idx + 1) as u64)))
        .collect();
    let categories: Vec<Category> = labels
        .iter()
        .enumerate()
        .map(|(idx, label)| Category::new((idx + 1) as u64, label.clone()))
        .collect();

    let mut images = Vec::new();
    let mut annotations = Vec::new();
    let mut ann_id = 1u64;

    for (image_idx, (image_ref, anns)) in parsed.annotations.iter().enumerate() {
        validate_relative_image_ref(image_ref, source_path)?;
        let (width, height) = resolve_image_dimensions(base_dir, image_ref, source_path)?;
        let image_id = ImageId::new((image_idx + 1) as u64);
        images.push(Image::new(
            image_id,
            normalize_path_separators(image_ref),
            width,
            height,
        ));

        for ann in anns {
            let normalized = BBoxXYXY::<Normalized>::from_xyxy(ann.x, ann.y, ann.x2, ann.y2);
            let bbox = normalized.to_pixel(width as f64, height as f64);
            annotations.push(Annotation::new(
                AnnotationId::new(ann_id),
                image_id,
                label_to_cat[&ann.label],
                bbox,
            ));
            ann_id += 1;
        }
    }

    Ok(Dataset {
        images,
        categories,
        annotations,
        ..Default::default()
    })
}

fn ir_to_cloud_annotations(dataset: &Dataset) -> CloudAnnotationsFile {
    let category_lookup: BTreeMap<CategoryId, &Category> =
        dataset.categories.iter().map(|cat| (cat.id, cat)).collect();
    let image_lookup: BTreeMap<ImageId, &Image> =
        dataset.images.iter().map(|img| (img.id, img)).collect();

    let mut sorted_categories: Vec<&Category> = dataset.categories.iter().collect();
    sorted_categories.sort_by_key(|category| category.id);
    let labels = sorted_categories
        .into_iter()
        .map(|category| category.name.clone())
        .collect();

    let mut anns_by_image: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }

    let mut sorted_images: Vec<&Image> = dataset.images.iter().collect();
    sorted_images.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    let mut annotations = BTreeMap::new();
    for image in sorted_images {
        let mut image_annotations = Vec::new();
        let mut anns = anns_by_image.get(&image.id).cloned().unwrap_or_default();
        anns.sort_by_key(|ann| ann.id);
        for ann in anns {
            let Some(category) = category_lookup.get(&ann.category_id) else {
                continue;
            };
            let Some(image_for_ann) = image_lookup.get(&ann.image_id) else {
                continue;
            };
            let bbox = ann
                .bbox
                .to_normalized(image_for_ann.width as f64, image_for_ann.height as f64);
            image_annotations.push(CloudAnnotation {
                x: bbox.xmin(),
                y: bbox.ymin(),
                x2: bbox.xmax(),
                y2: bbox.ymax(),
                id: Some(format!("panlabel-{}", ann.id.as_u64())),
                label: category.name.clone(),
            });
        }
        annotations.insert(image.file_name.clone(), image_annotations);
    }

    CloudAnnotationsFile {
        version: "1.0".to_string(),
        kind: "localization".to_string(),
        labels,
        annotations,
    }
}

fn validate_relative_image_ref(image_ref: &str, source_path: &Path) -> Result<(), PanlabelError> {
    if !is_safe_relative_image_ref(image_ref) {
        return Err(PanlabelError::CloudAnnotationsJsonInvalid {
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
    let candidate1 = base_dir.join(image_ref);
    let candidate2 = base_dir.join("images").join(image_ref);

    if let Ok(size) = imagesize::size(&candidate1) {
        return Ok((size.width as u32, size.height as u32));
    }
    if let Ok(size) = imagesize::size(&candidate2) {
        return Ok((size.width as u32, size.height as u32));
    }

    Err(PanlabelError::CloudAnnotationsImageNotFound {
        path: source_path.to_path_buf(),
        image_ref: image_ref.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Pixel;

    #[test]
    fn schema_parse_accepts_localization_file() {
        let json = r#"{
            "version": "1.0",
            "type": "localization",
            "labels": ["cat"],
            "annotations": {"img.bmp": [{"x": 0.1, "y": 0.2, "x2": 0.3, "y2": 0.4, "label": "cat"}]}
        }"#;
        let parsed: CloudAnnotationsFile = serde_json::from_str(json).expect("parse");
        assert_eq!(parsed.kind, "localization");
        assert_eq!(parsed.labels, vec!["cat"]);
        assert_eq!(parsed.annotations["img.bmp"][0].label, "cat");
    }

    #[test]
    fn writer_uses_normalized_coordinates() {
        let dataset = Dataset {
            images: vec![Image::new(1u64, "img.bmp", 200, 100)],
            categories: vec![Category::new(1u64, "box")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(20.0, 25.0, 100.0, 75.0),
            )],
            ..Default::default()
        };

        let json = to_cloud_annotations_string(&dataset).expect("serialize");
        let parsed: CloudAnnotationsFile = serde_json::from_str(&json).unwrap();
        let ann = &parsed.annotations["img.bmp"][0];
        assert!((ann.x - 0.1).abs() < 1e-9);
        assert!((ann.y - 0.25).abs() < 1e-9);
        assert!((ann.x2 - 0.5).abs() < 1e-9);
        assert!((ann.y2 - 0.75).abs() < 1e-9);
    }
}
