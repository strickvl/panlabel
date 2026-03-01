#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use panlabel::ir::{
    Annotation, AnnotationId, BBoxXYXY, Category, CategoryId, Dataset, DatasetInfo, Image, ImageId,
    Pixel,
};
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use proptest::test_runner::{Config as ProptestConfig, FileFailurePersistence};

pub const EPS_COCO: f64 = 1e-10;
pub const EPS_TFOD: f64 = 1e-2;
pub const EPS_LABEL_STUDIO: f64 = 1e-4;
pub const EPS_VOC: f64 = 1e-9;

pub fn eps_yolo(image_w: u32, image_h: u32) -> f64 {
    image_w.max(image_h) as f64 * 1e-6
}

pub fn eps_yolo_for_dataset(dataset: &Dataset) -> f64 {
    dataset
        .images
        .iter()
        .map(|img| eps_yolo(img.width, img.height))
        .fold(1e-9, f64::max)
}

pub fn proptest_config() -> ProptestConfig {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(64);

    let mut config = ProptestConfig::with_failure_persistence(FileFailurePersistence::WithSource(
        "proptest-regressions",
    ));
    config.cases = cases;
    config.max_shrink_iters = 1024;
    config
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnnSem {
    pub image_file: String,
    pub category: String,
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
}

pub fn ann_semantics(dataset: &Dataset) -> Result<Vec<AnnSem>, String> {
    let image_by_id: BTreeMap<ImageId, String> = dataset
        .images
        .iter()
        .map(|img| (img.id, img.file_name.clone()))
        .collect();
    let category_by_id: BTreeMap<CategoryId, String> = dataset
        .categories
        .iter()
        .map(|cat| (cat.id, cat.name.clone()))
        .collect();

    let mut out = Vec::with_capacity(dataset.annotations.len());
    for ann in &dataset.annotations {
        let image_file = image_by_id.get(&ann.image_id).ok_or_else(|| {
            format!(
                "annotation {} references missing image_id {}",
                ann.id.as_u64(),
                ann.image_id.as_u64()
            )
        })?;
        let category = category_by_id.get(&ann.category_id).ok_or_else(|| {
            format!(
                "annotation {} references missing category_id {}",
                ann.id.as_u64(),
                ann.category_id.as_u64()
            )
        })?;

        out.push(AnnSem {
            image_file: image_file.clone(),
            category: category.clone(),
            xmin: ann.bbox.xmin(),
            ymin: ann.bbox.ymin(),
            xmax: ann.bbox.xmax(),
            ymax: ann.bbox.ymax(),
        });
    }

    out.sort_by(ann_sem_cmp);
    Ok(out)
}

pub fn assert_annotations_equivalent(a: &Dataset, b: &Dataset, eps: f64) -> Result<(), String> {
    let left = ann_semantics(a)?;
    let right = ann_semantics(b)?;

    if left.len() != right.len() {
        return Err(format!(
            "annotation count mismatch: left={} right={}",
            left.len(),
            right.len()
        ));
    }

    assert_semantics_subset(&left, &right, eps)?;
    assert_semantics_subset(&right, &left, eps)?;
    Ok(())
}

pub fn assert_annotations_subset(sub: &Dataset, sup: &Dataset, eps: f64) -> Result<(), String> {
    let sub_sem = ann_semantics(sub)?;
    let sup_sem = ann_semantics(sup)?;
    assert_semantics_subset(&sub_sem, &sup_sem, eps)
}

pub fn image_dims_by_file_name(dataset: &Dataset) -> BTreeMap<String, (u32, u32)> {
    dataset
        .images
        .iter()
        .map(|img| (img.file_name.clone(), (img.width, img.height)))
        .collect()
}

pub fn used_image_file_names(dataset: &Dataset) -> Result<BTreeSet<String>, String> {
    Ok(ann_semantics(dataset)?
        .into_iter()
        .map(|ann| ann.image_file)
        .collect())
}

pub fn used_category_names(dataset: &Dataset) -> Result<BTreeSet<String>, String> {
    Ok(ann_semantics(dataset)?
        .into_iter()
        .map(|ann| ann.category)
        .collect())
}

pub fn assert_valid_references(dataset: &Dataset) -> Result<(), String> {
    let image_ids: BTreeSet<ImageId> = dataset.images.iter().map(|img| img.id).collect();
    let category_ids: BTreeSet<CategoryId> = dataset.categories.iter().map(|cat| cat.id).collect();

    for ann in &dataset.annotations {
        if !image_ids.contains(&ann.image_id) {
            return Err(format!(
                "annotation {} references missing image_id {}",
                ann.id.as_u64(),
                ann.image_id.as_u64()
            ));
        }
        if !category_ids.contains(&ann.category_id) {
            return Err(format!(
                "annotation {} references missing category_id {}",
                ann.id.as_u64(),
                ann.category_id.as_u64()
            ));
        }
    }

    Ok(())
}

pub fn arb_bbox_within(width: u32, height: u32) -> BoxedStrategy<BBoxXYXY<Pixel>> {
    prop::num::u32::ANY
        .prop_map(move |seed| {
            bbox_from_seed(
                width,
                height,
                seed,
                seed.rotate_left(3),
                seed.rotate_left(7),
                seed.rotate_left(11),
            )
        })
        .boxed()
}

pub fn arb_dataset_full(
    max_images: usize,
    max_cats: usize,
    max_anns: usize,
) -> BoxedStrategy<Dataset> {
    assert!(max_images > 0, "max_images must be > 0");
    assert!(max_cats > 0, "max_cats must be > 0");

    (1usize..=max_images, 1usize..=max_cats, 0usize..=max_anns)
        .prop_flat_map(|(image_count, category_count, ann_count)| {
            (
                proptest::collection::hash_map(
                    image_file_name_strategy(),
                    (2u32..=4096, 2u32..=4096),
                    image_count..=image_count,
                ),
                proptest::collection::hash_set(
                    category_name_strategy(),
                    category_count..=category_count,
                ),
                proptest::collection::vec(ann_seed_strategy(), ann_count..=ann_count),
            )
                .prop_map(|(images, categories, ann_seeds)| {
                    build_dataset(images, categories, ann_seeds, false)
                })
        })
        .boxed()
}

pub fn arb_dataset_annotated(
    max_images: usize,
    max_cats: usize,
    max_anns: usize,
) -> BoxedStrategy<Dataset> {
    assert!(max_images > 0, "max_images must be > 0");
    assert!(max_cats > 0, "max_cats must be > 0");
    assert!(
        max_anns >= max_images.max(max_cats),
        "max_anns must be >= max(max_images, max_cats)"
    );

    (1usize..=max_images, 1usize..=max_cats)
        .prop_flat_map(move |(image_count, category_count)| {
            let min_anns = image_count.max(category_count);
            (Just(image_count), Just(category_count), min_anns..=max_anns).prop_flat_map(
                |(image_count, category_count, ann_count)| {
                    (
                        proptest::collection::hash_map(
                            image_file_name_strategy(),
                            (2u32..=4096, 2u32..=4096),
                            image_count..=image_count,
                        ),
                        proptest::collection::hash_set(
                            category_name_strategy(),
                            category_count..=category_count,
                        ),
                        proptest::collection::vec(ann_seed_strategy(), ann_count..=ann_count),
                    )
                        .prop_map(|(images, categories, ann_seeds)| {
                            build_dataset(images, categories, ann_seeds, true)
                        })
                },
            )
        })
        .boxed()
}

pub fn arb_dataset_with_confidence(
    max_images: usize,
    max_cats: usize,
    max_anns: usize,
) -> BoxedStrategy<Dataset> {
    arb_dataset_annotated(max_images, max_cats, max_anns)
        .prop_flat_map(|dataset| {
            let ann_count = dataset.annotations.len();
            proptest::collection::vec((any::<bool>(), 0u16..=1000u16), ann_count..=ann_count)
                .prop_map(move |choices| {
                    let mut with_confidence = dataset.clone();
                    for (idx, (include_confidence, raw_confidence)) in
                        choices.into_iter().enumerate()
                    {
                        let conf = raw_confidence as f64 / 1000.0;
                        if idx == 0 || include_confidence {
                            with_confidence.annotations[idx].confidence = Some(conf);
                        }
                    }
                    with_confidence
                })
        })
        .boxed()
}

type AnnSeed = (u16, u16, u32, u32, u32, u32);

fn ann_seed_strategy() -> impl Strategy<Value = AnnSeed> {
    (
        any::<u16>(),
        any::<u16>(),
        any::<u32>(),
        any::<u32>(),
        any::<u32>(),
        any::<u32>(),
    )
}

fn image_file_name_strategy() -> BoxedStrategy<String> {
    proptest::string::string_regex("[a-z0-9_]{1,12}\\.jpg")
        .expect("valid filename regex")
        .boxed()
}

fn category_name_strategy() -> BoxedStrategy<String> {
    proptest::string::string_regex("[a-z]{1,20}")
        .expect("valid category name regex")
        .boxed()
}

fn build_dataset(
    image_data: HashMap<String, (u32, u32)>,
    category_names: HashSet<String>,
    ann_seeds: Vec<AnnSeed>,
    ensure_coverage: bool,
) -> Dataset {
    let mut image_rows: Vec<(String, (u32, u32))> = image_data.into_iter().collect();
    image_rows.sort_by(|a, b| a.0.cmp(&b.0));

    let mut category_rows: Vec<String> = category_names.into_iter().collect();
    category_rows.sort();

    let images: Vec<Image> = image_rows
        .iter()
        .enumerate()
        .map(|(idx, (file_name, (width, height)))| {
            Image::new((idx + 1) as u64, file_name.clone(), *width, *height)
        })
        .collect();

    let categories: Vec<Category> = category_rows
        .iter()
        .enumerate()
        .map(|(idx, name)| Category::new((idx + 1) as u64, name.clone()))
        .collect();

    let coverage_len = if ensure_coverage {
        images.len().max(categories.len())
    } else {
        0
    };

    let annotations: Vec<Annotation> = ann_seeds
        .into_iter()
        .enumerate()
        .map(|(idx, seed)| {
            let (image_seed, category_seed, sx, sy, sw, sh) = seed;
            let image_idx = if idx < coverage_len {
                idx % images.len()
            } else {
                image_seed as usize % images.len()
            };
            let category_idx = if idx < coverage_len {
                idx % categories.len()
            } else {
                category_seed as usize % categories.len()
            };

            let image = &images[image_idx];
            let category = &categories[category_idx];
            let bbox = bbox_from_seed(image.width, image.height, sx, sy, sw, sh);

            Annotation::new(
                AnnotationId::new((idx + 1) as u64),
                image.id,
                category.id,
                bbox,
            )
        })
        .collect();

    Dataset {
        info: DatasetInfo::default(),
        licenses: vec![],
        images,
        categories,
        annotations,
    }
}

fn bbox_from_seed(width: u32, height: u32, sx: u32, sy: u32, sw: u32, sh: u32) -> BBoxXYXY<Pixel> {
    let xmin = sx % (width - 1);
    let ymin = sy % (height - 1);
    let xmax = xmin + 1 + (sw % (width - xmin));
    let ymax = ymin + 1 + (sh % (height - ymin));

    BBoxXYXY::from_xyxy(xmin as f64, ymin as f64, xmax as f64, ymax as f64)
}

fn assert_semantics_subset(sub: &[AnnSem], sup: &[AnnSem], eps: f64) -> Result<(), String> {
    let mut used = vec![false; sup.len()];

    for wanted in sub {
        let mut found_match = None;
        for (idx, candidate) in sup.iter().enumerate() {
            if used[idx] {
                continue;
            }
            if approx_ann_sem(wanted, candidate, eps) {
                found_match = Some(idx);
                break;
            }
        }

        match found_match {
            Some(idx) => used[idx] = true,
            None => {
                return Err(format!(
                    "missing annotation semantic match for image='{}', category='{}', bbox=({}, {}, {}, {}), eps={}",
                    wanted.image_file,
                    wanted.category,
                    wanted.xmin,
                    wanted.ymin,
                    wanted.xmax,
                    wanted.ymax,
                    eps
                ));
            }
        }
    }

    Ok(())
}

fn approx_ann_sem(left: &AnnSem, right: &AnnSem, eps: f64) -> bool {
    left.image_file == right.image_file
        && left.category == right.category
        && (left.xmin - right.xmin).abs() <= eps
        && (left.ymin - right.ymin).abs() <= eps
        && (left.xmax - right.xmax).abs() <= eps
        && (left.ymax - right.ymax).abs() <= eps
}

fn ann_sem_cmp(a: &AnnSem, b: &AnnSem) -> std::cmp::Ordering {
    a.image_file
        .cmp(&b.image_file)
        .then_with(|| a.category.cmp(&b.category))
        .then_with(|| a.xmin.total_cmp(&b.xmin))
        .then_with(|| a.ymin.total_cmp(&b.ymin))
        .then_with(|| a.xmax.total_cmp(&b.xmax))
        .then_with(|| a.ymax.total_cmp(&b.ymax))
}
