//! Dataset sampling utilities.

use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{HashMap, HashSet};

use crate::error::PanlabelError;
use crate::ir::{CategoryId, Dataset, ImageId};

/// Image sampling strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleStrategy {
    /// Uniform random sampling.
    Random,
    /// Category-aware weighted sampling.
    Stratified,
}

/// Category filtering mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CategoryMode {
    /// Keep whole images that contain at least one selected category.
    Images,
    /// Keep only matching annotations; drop images with no remaining annotations.
    Annotations,
}

/// Sampling options.
#[derive(Clone, Debug)]
pub struct SampleOptions {
    pub n: Option<usize>,
    pub fraction: Option<f64>,
    pub seed: Option<u64>,
    pub strategy: SampleStrategy,
    pub categories: Vec<String>,
    pub category_mode: CategoryMode,
}

/// Validate sampling options before running.
pub fn validate_sample_options(opts: &SampleOptions) -> Result<(), PanlabelError> {
    match (opts.n, opts.fraction) {
        (Some(_), Some(_)) => {
            return Err(PanlabelError::InvalidSampleParams {
                message: "-n and --fraction are mutually exclusive".to_string(),
            });
        }
        (None, None) => {
            return Err(PanlabelError::InvalidSampleParams {
                message: "set exactly one of -n or --fraction".to_string(),
            });
        }
        _ => {}
    }

    if let Some(n) = opts.n {
        if n == 0 {
            return Err(PanlabelError::InvalidSampleParams {
                message: "-n must be greater than 0".to_string(),
            });
        }
    }

    if let Some(fraction) = opts.fraction {
        if !(0.0 < fraction && fraction <= 1.0) {
            return Err(PanlabelError::InvalidSampleParams {
                message: "--fraction must be in the interval (0.0, 1.0]".to_string(),
            });
        }
    }

    Ok(())
}

/// Sample a dataset according to options.
pub fn sample_dataset(dataset: &Dataset, opts: &SampleOptions) -> Result<Dataset, PanlabelError> {
    validate_sample_options(opts)?;

    let filtered = filter_dataset_by_categories(dataset, &opts.categories, opts.category_mode)?;
    if filtered.images.is_empty() {
        return Err(PanlabelError::SampleFailed {
            message: "no images available after category filtering".to_string(),
        });
    }

    let target = target_image_count(filtered.images.len(), opts.n, opts.fraction);
    if target == 0 {
        return Err(PanlabelError::SampleFailed {
            message: "requested sample size is zero".to_string(),
        });
    }

    let selected_ids = match opts.strategy {
        SampleStrategy::Random => select_image_ids_random(&filtered, target, opts.seed),
        SampleStrategy::Stratified => select_image_ids_stratified(&filtered, target, opts.seed),
    };

    let keep: HashSet<ImageId> = selected_ids.into_iter().collect();
    Ok(subset_by_image_ids(&filtered, &keep))
}

/// Filter dataset by categories according to mode.
pub fn filter_dataset_by_categories(
    dataset: &Dataset,
    categories: &[String],
    mode: CategoryMode,
) -> Result<Dataset, PanlabelError> {
    if categories.is_empty() {
        return Ok(dataset.clone());
    }

    let requested: HashSet<String> = categories
        .iter()
        .map(|c| c.trim())
        .filter(|c| !c.is_empty())
        .map(str::to_string)
        .collect();

    if requested.is_empty() {
        return Ok(dataset.clone());
    }

    let mut selected_category_ids: HashSet<CategoryId> = HashSet::new();
    for category in &dataset.categories {
        if requested.contains(&category.name) {
            selected_category_ids.insert(category.id);
        }
    }

    if selected_category_ids.is_empty() {
        return Err(PanlabelError::InvalidSampleParams {
            message: "none of the requested categories were found in the dataset".to_string(),
        });
    }

    let (images, annotations) = match mode {
        CategoryMode::Images => {
            let keep_image_ids: HashSet<ImageId> = dataset
                .annotations
                .iter()
                .filter(|ann| selected_category_ids.contains(&ann.category_id))
                .map(|ann| ann.image_id)
                .collect();

            let images = dataset
                .images
                .iter()
                .filter(|image| keep_image_ids.contains(&image.id))
                .cloned()
                .collect();

            let annotations = dataset
                .annotations
                .iter()
                .filter(|ann| keep_image_ids.contains(&ann.image_id))
                .cloned()
                .collect();

            (images, annotations)
        }
        CategoryMode::Annotations => {
            let annotations: Vec<_> = dataset
                .annotations
                .iter()
                .filter(|ann| selected_category_ids.contains(&ann.category_id))
                .cloned()
                .collect();

            let keep_image_ids: HashSet<ImageId> =
                annotations.iter().map(|ann| ann.image_id).collect();

            let images = dataset
                .images
                .iter()
                .filter(|image| keep_image_ids.contains(&image.id))
                .cloned()
                .collect();

            (images, annotations)
        }
    };

    Ok(Dataset {
        info: dataset.info.clone(),
        licenses: dataset.licenses.clone(),
        images,
        categories: dataset.categories.clone(),
        annotations,
    })
}

/// Compute target image count from n/fraction.
pub fn target_image_count(total: usize, n: Option<usize>, fraction: Option<f64>) -> usize {
    if total == 0 {
        return 0;
    }

    if let Some(n) = n {
        return n.min(total);
    }

    if let Some(frac) = fraction {
        let raw = (total as f64 * frac).ceil() as usize;
        return raw.clamp(1, total);
    }

    0
}

fn sorted_image_ids(dataset: &Dataset) -> Vec<ImageId> {
    let mut rows: Vec<(String, ImageId)> = dataset
        .images
        .iter()
        .map(|image| (image.file_name.clone(), image.id))
        .collect();

    rows.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    rows.into_iter().map(|(_, id)| id).collect()
}

/// Select image IDs uniformly at random.
pub fn select_image_ids_random(dataset: &Dataset, k: usize, seed: Option<u64>) -> Vec<ImageId> {
    let mut ids = sorted_image_ids(dataset);

    if k >= ids.len() {
        return ids;
    }

    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        ids.shuffle(&mut rng);
    } else {
        let mut rng = rand::rng();
        ids.shuffle(&mut rng);
    }

    ids.truncate(k);
    ids.sort();
    ids
}

/// Select image IDs with category-aware weighted sampling without replacement.
pub fn select_image_ids_stratified(dataset: &Dataset, k: usize, seed: Option<u64>) -> Vec<ImageId> {
    let ids = sorted_image_ids(dataset);
    if k >= ids.len() {
        return ids;
    }

    let mut category_freq: HashMap<CategoryId, usize> = HashMap::new();
    for ann in &dataset.annotations {
        *category_freq.entry(ann.category_id).or_insert(0) += 1;
    }

    let mut image_categories: HashMap<ImageId, HashSet<CategoryId>> = HashMap::new();
    for ann in &dataset.annotations {
        image_categories
            .entry(ann.image_id)
            .or_default()
            .insert(ann.category_id);
    }

    let mut candidates: Vec<(ImageId, f64)> = ids
        .iter()
        .map(|id| {
            let weight = image_categories
                .get(id)
                .map(|cats| {
                    cats.iter()
                        .map(|cat_id| {
                            let freq = *category_freq.get(cat_id).unwrap_or(&1) as f64;
                            1.0 / freq
                        })
                        .sum::<f64>()
                })
                .unwrap_or(0.0);
            (*id, weight)
        })
        .collect();

    let mut selected: Vec<ImageId> = Vec::with_capacity(k);

    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        weighted_sample_without_replacement(&mut candidates, k, &mut selected, &mut rng);
    } else {
        let mut rng = rand::rng();
        weighted_sample_without_replacement(&mut candidates, k, &mut selected, &mut rng);
    }

    selected.sort();
    selected
}

fn weighted_sample_without_replacement<R: Rng + ?Sized>(
    candidates: &mut Vec<(ImageId, f64)>,
    k: usize,
    selected: &mut Vec<ImageId>,
    rng: &mut R,
) {
    while selected.len() < k && !candidates.is_empty() {
        let total_weight: f64 = candidates.iter().map(|(_, w)| w.max(0.0)).sum();

        let pick_index = if total_weight <= 0.0 {
            rng.random_range(0..candidates.len())
        } else {
            let mut draw = rng.random::<f64>() * total_weight;
            let mut index = candidates.len() - 1;

            for (i, (_, weight)) in candidates.iter().enumerate() {
                draw -= weight.max(0.0);
                if draw <= 0.0 {
                    index = i;
                    break;
                }
            }

            index
        };

        let (image_id, _) = candidates.swap_remove(pick_index);
        selected.push(image_id);
    }
}

/// Create a subset dataset by selected image IDs, preserving original IDs.
pub fn subset_by_image_ids(dataset: &Dataset, keep: &HashSet<ImageId>) -> Dataset {
    let images = dataset
        .images
        .iter()
        .filter(|image| keep.contains(&image.id))
        .cloned()
        .collect();

    let annotations = dataset
        .annotations
        .iter()
        .filter(|ann| keep.contains(&ann.image_id))
        .cloned()
        .collect();

    Dataset {
        info: dataset.info.clone(),
        licenses: dataset.licenses.clone(),
        images,
        categories: dataset.categories.clone(),
        annotations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Annotation, BBoxXYXY, Category, Image, Pixel};

    fn make_dataset() -> Dataset {
        Dataset {
            images: vec![
                Image::new(1u64, "a.jpg", 100, 100),
                Image::new(2u64, "b.jpg", 100, 100),
                Image::new(3u64, "c.jpg", 100, 100),
            ],
            categories: vec![Category::new(1u64, "person"), Category::new(2u64, "dog")],
            annotations: vec![
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(1.0, 1.0, 10.0, 10.0),
                ),
                Annotation::new(
                    2u64,
                    1u64,
                    2u64,
                    BBoxXYXY::<Pixel>::from_xyxy(2.0, 2.0, 12.0, 12.0),
                ),
                Annotation::new(
                    3u64,
                    2u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(3.0, 3.0, 14.0, 14.0),
                ),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn validate_opts_rejects_invalid_combinations() {
        let both = SampleOptions {
            n: Some(1),
            fraction: Some(0.5),
            seed: None,
            strategy: SampleStrategy::Random,
            categories: Vec::new(),
            category_mode: CategoryMode::Images,
        };
        assert!(validate_sample_options(&both).is_err());

        let none = SampleOptions {
            n: None,
            fraction: None,
            seed: None,
            strategy: SampleStrategy::Random,
            categories: Vec::new(),
            category_mode: CategoryMode::Images,
        };
        assert!(validate_sample_options(&none).is_err());
    }

    #[test]
    fn random_sampling_is_deterministic_with_seed() {
        let dataset = make_dataset();
        let a = select_image_ids_random(&dataset, 2, Some(42));
        let b = select_image_ids_random(&dataset, 2, Some(42));
        assert_eq!(a, b);
    }

    #[test]
    fn annotations_mode_keeps_all_categories() {
        let dataset = make_dataset();
        let filtered = filter_dataset_by_categories(
            &dataset,
            &["person".to_string()],
            CategoryMode::Annotations,
        )
        .expect("filter ok");

        assert_eq!(filtered.categories.len(), 2);
        assert!(filtered
            .annotations
            .iter()
            .all(|ann| ann.category_id == 1u64.into()));
    }
}
