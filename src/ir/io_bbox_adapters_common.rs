use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::model::{Annotation, Category, Dataset, DatasetInfo, Image};
use super::{AnnotationId, BBoxXYXY, CategoryId, ImageId, Pixel};

pub(crate) const IMAGE_EXTENSIONS: &[&str] = &[".jpg", ".jpeg", ".png", ".bmp", ".webp"];

#[derive(Clone, Debug)]
pub(crate) struct RawAnn {
    pub image: String,
    pub category: String,
    pub bbox: BBoxXYXY<Pixel>,
    pub confidence: Option<f64>,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub(crate) struct RawImage {
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    pub attributes: BTreeMap<String, String>,
}

pub(crate) fn dataset_from_raw(
    mut images: Vec<RawImage>,
    anns: Vec<RawAnn>,
    categories_hint: Vec<(String, Option<String>)>,
    info: DatasetInfo,
) -> Dataset {
    let mut image_names: BTreeSet<String> =
        images.iter().map(|img| img.file_name.clone()).collect();
    for ann in &anns {
        image_names.insert(ann.image.clone());
    }

    let mut images_by_name: BTreeMap<String, RawImage> = BTreeMap::new();
    for img in images.drain(..) {
        images_by_name.insert(img.file_name.clone(), img);
    }
    for name in &image_names {
        images_by_name
            .entry(name.clone())
            .or_insert_with(|| RawImage {
                file_name: name.clone(),
                width: inferred_width_for(name, &anns),
                height: inferred_height_for(name, &anns),
                attributes: BTreeMap::new(),
            });
    }

    let mut category_meta: BTreeMap<String, Option<String>> = BTreeMap::new();
    for (name, supercategory) in categories_hint {
        category_meta.entry(name).or_insert(supercategory);
    }
    for ann in &anns {
        category_meta.entry(ann.category.clone()).or_insert(None);
    }

    let image_id_by_name: BTreeMap<String, ImageId> = images_by_name
        .keys()
        .enumerate()
        .map(|(idx, name)| (name.clone(), ImageId::new((idx + 1) as u64)))
        .collect();
    let category_id_by_name: BTreeMap<String, CategoryId> = category_meta
        .keys()
        .enumerate()
        .map(|(idx, name)| (name.clone(), CategoryId::new((idx + 1) as u64)))
        .collect();

    let images = images_by_name
        .into_values()
        .map(|raw| {
            let mut image = Image::new(
                image_id_by_name[&raw.file_name],
                raw.file_name,
                raw.width,
                raw.height,
            );
            image.attributes = raw.attributes;
            image
        })
        .collect();

    let categories = category_meta
        .into_iter()
        .map(|(name, supercategory)| {
            let mut cat = Category::new(category_id_by_name[&name], name);
            cat.supercategory = supercategory;
            cat
        })
        .collect();

    let annotations = anns
        .into_iter()
        .enumerate()
        .map(|(idx, raw)| {
            let mut ann = Annotation::new(
                AnnotationId::new((idx + 1) as u64),
                image_id_by_name[&raw.image],
                category_id_by_name[&raw.category],
                raw.bbox,
            );
            ann.confidence = raw.confidence;
            ann.attributes = raw.attributes;
            ann
        })
        .collect();

    Dataset {
        info,
        licenses: vec![],
        images,
        categories,
        annotations,
    }
}

fn inferred_width_for(image: &str, anns: &[RawAnn]) -> u32 {
    anns.iter()
        .filter(|ann| ann.image == image)
        .map(|ann| ann.bbox.xmax().ceil().max(1.0) as u32)
        .max()
        .unwrap_or(1)
}

fn inferred_height_for(image: &str, anns: &[RawAnn]) -> u32 {
    anns.iter()
        .filter(|ann| ann.image == image)
        .map(|ann| ann.bbox.ymax().ceil().max(1.0) as u32)
        .max()
        .unwrap_or(1)
}

pub(crate) fn image_dimensions_if_found(base_dir: &Path, image_ref: &str) -> Option<(u32, u32)> {
    for candidate in image_candidates(base_dir, image_ref) {
        if candidate.is_file() {
            if let Ok(size) = imagesize::size(&candidate) {
                return Some((size.width as u32, size.height as u32));
            }
        }
    }
    None
}

pub(crate) fn image_dimensions_or_error<E>(
    base_dir: &Path,
    image_ref: &str,
    not_found: impl FnOnce() -> E,
    dim_read: impl FnOnce(PathBuf, imagesize::ImageError) -> E,
) -> Result<(u32, u32), E> {
    let candidates = image_candidates(base_dir, image_ref);
    for candidate in candidates {
        if candidate.is_file() {
            return imagesize::size(&candidate)
                .map(|size| (size.width as u32, size.height as u32))
                .map_err(|source| dim_read(candidate, source));
        }
    }
    Err(not_found())
}

pub(crate) fn image_candidates(base_dir: &Path, image_ref: &str) -> Vec<PathBuf> {
    let ref_path = Path::new(image_ref);
    let mut out = Vec::new();
    if ref_path.is_absolute() {
        out.push(ref_path.to_path_buf());
        return out;
    }
    if ref_path.extension().is_some() {
        out.push(base_dir.join(image_ref));
        out.push(base_dir.join("images").join(image_ref));
    } else {
        for ext in IMAGE_EXTENSIONS {
            out.push(base_dir.join(format!("{image_ref}{ext}")));
            out.push(base_dir.join("images").join(format!("{image_ref}{ext}")));
        }
    }
    out
}

pub(crate) fn f64_field(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

pub(crate) fn u32_field(value: &Value, key: &str) -> Option<u32> {
    value.get(key).and_then(Value::as_u64).map(|v| v as u32)
}

pub(crate) fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

pub(crate) fn scalar_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

pub(crate) fn annotations_by_image(dataset: &Dataset) -> BTreeMap<ImageId, Vec<&Annotation>> {
    let mut out: BTreeMap<ImageId, Vec<&Annotation>> = BTreeMap::new();
    for ann in &dataset.annotations {
        out.entry(ann.image_id).or_default().push(ann);
    }
    for anns in out.values_mut() {
        anns.sort_by_key(|ann| ann.id);
    }
    out
}
