//! Dataset statistics.
//!
//! This module analyzes datasets and produces structured statistics reports.

pub mod html;
mod report;

pub use report::{
    AnnotationDensityStats, AreaDistribution, AspectRatioBucket, AspectRatioDistribution,
    BBoxStats, CooccurrencePair, CooccurrenceTopPairs, ImageResolutionStats, LabelCount,
    LabelsSection, PerCategoryBBoxStats, StatsReport, SummarySection,
};

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::ir::{CategoryId, Dataset, ImageId};

/// Options for dataset statistics.
#[derive(Clone, Debug)]
pub struct StatsOptions {
    /// Number of top labels to show in the histogram.
    pub top_labels: usize,
    /// Number of top co-occurrence pairs to show.
    pub top_pairs: usize,
    /// Tolerance in pixels for out-of-bounds checks.
    pub oob_tolerance_px: f64,
    /// Width of histogram bars (in characters).
    pub bar_width: usize,
}

impl Default for StatsOptions {
    fn default() -> Self {
        Self {
            top_labels: 10,
            top_pairs: 10,
            oob_tolerance_px: 0.5,
            bar_width: 20,
        }
    }
}

/// Compute a full statistics report for a dataset.
pub fn stats_dataset(dataset: &Dataset, opts: &StatsOptions) -> StatsReport {
    let image_dims: HashMap<ImageId, (u32, u32)> = dataset
        .images
        .iter()
        .map(|img| (img.id, (img.width, img.height)))
        .collect();

    let category_names: HashMap<CategoryId, String> = dataset
        .categories
        .iter()
        .map(|cat| (cat.id, cat.name.clone()))
        .collect();

    let summary = compute_summary(dataset);
    let labels = compute_labels(dataset, &category_names, opts.top_labels);
    let bboxes = compute_bbox_stats(dataset, &image_dims, opts.oob_tolerance_px);
    let image_resolutions = compute_image_resolution_stats(dataset);
    let annotation_density = compute_annotation_density(dataset);
    let area_distribution = compute_area_distribution(dataset);
    let aspect_ratios = compute_aspect_ratio_distribution(dataset);
    let per_category_bbox =
        compute_per_category_bbox_stats(dataset, &category_names, opts.top_labels);
    let cooccurrence_top_pairs =
        compute_cooccurrence_top_pairs(dataset, &category_names, opts.top_pairs);

    StatsReport {
        summary,
        labels,
        bboxes,
        image_resolutions,
        annotation_density,
        area_distribution,
        aspect_ratios,
        per_category_bbox,
        cooccurrence_top_pairs,
        bar_width: opts.bar_width,
    }
}

/// Compute summary section counts.
fn compute_summary(dataset: &Dataset) -> SummarySection {
    let annotated_image_ids: HashSet<ImageId> =
        dataset.annotations.iter().map(|ann| ann.image_id).collect();

    SummarySection {
        images: dataset.images.len(),
        categories: dataset.categories.len(),
        annotations: dataset.annotations.len(),
        licenses: dataset.licenses.len(),
        annotated_images: annotated_image_ids.len(),
    }
}

/// Compute label distribution histogram.
fn compute_labels(
    dataset: &Dataset,
    category_names: &HashMap<CategoryId, String>,
    top_n: usize,
) -> LabelsSection {
    let mut counts: HashMap<String, usize> = HashMap::new();

    for ann in &dataset.annotations {
        let label = category_names
            .get(&ann.category_id)
            .cloned()
            .unwrap_or_else(|| format!("<missing cat {}>", ann.category_id));

        *counts.entry(label).or_insert(0) += 1;
    }

    let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let total_distinct = sorted.len();
    let total_annotations = dataset.annotations.len();

    let (top_entries, rest): (Vec<_>, Vec<_>) = sorted
        .into_iter()
        .enumerate()
        .partition(|(i, _)| *i < top_n);

    let entries: Vec<LabelCount> = top_entries
        .into_iter()
        .map(|(_, (label, count))| LabelCount { label, count })
        .collect();

    let other_count: usize = rest.into_iter().map(|(_, (_, count))| count).sum();

    LabelsSection {
        top_n,
        total_distinct,
        total_annotations,
        entries,
        other_count,
    }
}

/// Compute bounding box statistics.
fn compute_bbox_stats(
    dataset: &Dataset,
    image_dims: &HashMap<ImageId, (u32, u32)>,
    tolerance: f64,
) -> BBoxStats {
    let mut stats = BBoxStats {
        total: dataset.annotations.len(),
        ..Default::default()
    };

    let mut min_width: Option<f64> = None;
    let mut max_width: Option<f64> = None;
    let mut min_height: Option<f64> = None;
    let mut max_height: Option<f64> = None;

    for ann in &dataset.annotations {
        let bbox = &ann.bbox;

        let xmin = bbox.min.x;
        let ymin = bbox.min.y;
        let xmax = bbox.max.x;
        let ymax = bbox.max.y;

        let is_finite =
            xmin.is_finite() && ymin.is_finite() && xmax.is_finite() && ymax.is_finite();

        if is_finite {
            stats.finite += 1;

            let is_ordered = xmin <= xmax && ymin <= ymax;
            if is_ordered {
                stats.ordered += 1;

                let width = xmax - xmin;
                let height = ymax - ymin;

                min_width = Some(min_width.map_or(width, |m| m.min(width)));
                max_width = Some(max_width.map_or(width, |m| m.max(width)));
                min_height = Some(min_height.map_or(height, |m| m.min(height)));
                max_height = Some(max_height.map_or(height, |m| m.max(height)));

                let area = width * height;
                if area <= 0.0 {
                    stats.degenerate_area += 1;
                }
            }

            if let Some(&(img_w, img_h)) = image_dims.get(&ann.image_id) {
                stats.oob_checked += 1;

                let img_w = img_w as f64;
                let img_h = img_h as f64;

                let is_oob = xmin < -tolerance
                    || ymin < -tolerance
                    || xmax > img_w + tolerance
                    || ymax > img_h + tolerance;

                if is_oob {
                    stats.out_of_bounds += 1;
                }
            } else {
                stats.missing_image_ref += 1;
            }
        } else if !image_dims.contains_key(&ann.image_id) {
            stats.missing_image_ref += 1;
        }
    }

    stats.min_width = min_width;
    stats.max_width = max_width;
    stats.min_height = min_height;
    stats.max_height = max_height;

    stats
}

/// Compute image resolution spread statistics.
fn compute_image_resolution_stats(dataset: &Dataset) -> ImageResolutionStats {
    if dataset.images.is_empty() {
        return ImageResolutionStats::default();
    }

    let mut min_w = u32::MAX;
    let mut max_w = 0u32;
    let mut sum_w = 0u64;

    let mut min_h = u32::MAX;
    let mut max_h = 0u32;
    let mut sum_h = 0u64;

    for image in &dataset.images {
        min_w = min_w.min(image.width);
        max_w = max_w.max(image.width);
        sum_w += image.width as u64;

        min_h = min_h.min(image.height);
        max_h = max_h.max(image.height);
        sum_h += image.height as u64;
    }

    let count = dataset.images.len() as f64;
    ImageResolutionStats {
        min_w,
        max_w,
        mean_w: sum_w as f64 / count,
        min_h,
        max_h,
        mean_h: sum_h as f64 / count,
    }
}

/// Compute annotation density statistics (per image).
fn compute_annotation_density(dataset: &Dataset) -> AnnotationDensityStats {
    if dataset.images.is_empty() {
        return AnnotationDensityStats::default();
    }

    let mut counts: HashMap<ImageId, usize> = dataset
        .images
        .iter()
        .map(|image| (image.id, 0usize))
        .collect();

    for ann in &dataset.annotations {
        if let Some(count) = counts.get_mut(&ann.image_id) {
            *count += 1;
        }
    }

    let values: Vec<usize> = counts.values().copied().collect();
    let min_per_image = *values.iter().min().unwrap_or(&0);
    let max_per_image = *values.iter().max().unwrap_or(&0);
    let sum: usize = values.iter().sum();
    let mean_per_image = sum as f64 / values.len() as f64;
    let zero_annotation_images = values.iter().filter(|&&v| v == 0).count();

    AnnotationDensityStats {
        min_per_image,
        max_per_image,
        mean_per_image,
        zero_annotation_images,
    }
}

/// Compute area distribution using COCO thresholds.
fn compute_area_distribution(dataset: &Dataset) -> AreaDistribution {
    let mut stats = AreaDistribution::default();

    for ann in &dataset.annotations {
        let bbox = &ann.bbox;
        if !bbox.is_finite() || !bbox.is_ordered() {
            stats.invalid += 1;
            continue;
        }

        let area = bbox.area();
        if !area.is_finite() || area <= 0.0 {
            stats.invalid += 1;
            continue;
        }

        if area < 1024.0 {
            stats.small += 1;
        } else if area < 9216.0 {
            stats.medium += 1;
        } else {
            stats.large += 1;
        }
    }

    stats
}

/// Compute aspect-ratio distribution across fixed buckets.
fn compute_aspect_ratio_distribution(dataset: &Dataset) -> AspectRatioDistribution {
    let names = ["<0.5", "0.5-1", "1-2", "2-5", ">=5"];
    let mut counts = [0usize; 5];
    let mut invalid = 0usize;

    for ann in &dataset.annotations {
        let bbox = &ann.bbox;
        if !bbox.is_finite() || !bbox.is_ordered() {
            invalid += 1;
            continue;
        }

        let width = bbox.width();
        let height = bbox.height();
        let area = bbox.area();

        if !width.is_finite()
            || !height.is_finite()
            || !area.is_finite()
            || width <= 0.0
            || height <= 0.0
            || area <= 0.0
        {
            invalid += 1;
            continue;
        }

        let ratio = width / height;
        if !ratio.is_finite() {
            invalid += 1;
            continue;
        }

        let idx = if ratio < 0.5 {
            0
        } else if ratio < 1.0 {
            1
        } else if ratio < 2.0 {
            2
        } else if ratio < 5.0 {
            3
        } else {
            4
        };

        counts[idx] += 1;
    }

    AspectRatioDistribution {
        buckets: names
            .iter()
            .zip(counts)
            .map(|(name, count)| AspectRatioBucket {
                name: (*name).to_string(),
                count,
            })
            .collect(),
        invalid,
    }
}

/// Compute per-category bbox area stats, sorted by annotation count desc.
fn compute_per_category_bbox_stats(
    dataset: &Dataset,
    category_names: &HashMap<CategoryId, String>,
    top_n: usize,
) -> Vec<PerCategoryBBoxStats> {
    #[derive(Default)]
    struct Agg {
        annotations: usize,
        valid_count: usize,
        min_area: f64,
        max_area: f64,
        sum_area: f64,
    }

    let mut per_category: BTreeMap<String, Agg> = BTreeMap::new();

    for ann in &dataset.annotations {
        let category = category_names
            .get(&ann.category_id)
            .cloned()
            .unwrap_or_else(|| format!("<missing cat {}>", ann.category_id));

        let entry = per_category.entry(category).or_default();
        entry.annotations += 1;

        let bbox = &ann.bbox;
        if !bbox.is_finite() || !bbox.is_ordered() {
            continue;
        }

        let area = bbox.area();
        if !area.is_finite() || area <= 0.0 {
            continue;
        }

        if entry.valid_count == 0 {
            entry.min_area = area;
            entry.max_area = area;
        } else {
            entry.min_area = entry.min_area.min(area);
            entry.max_area = entry.max_area.max(area);
        }
        entry.valid_count += 1;
        entry.sum_area += area;
    }

    let mut rows: Vec<PerCategoryBBoxStats> = per_category
        .into_iter()
        .map(|(category, agg)| PerCategoryBBoxStats {
            category,
            annotations: agg.annotations,
            min_area: if agg.valid_count > 0 {
                Some(agg.min_area)
            } else {
                None
            },
            max_area: if agg.valid_count > 0 {
                Some(agg.max_area)
            } else {
                None
            },
            mean_area: if agg.valid_count > 0 {
                Some(agg.sum_area / agg.valid_count as f64)
            } else {
                None
            },
        })
        .collect();

    rows.sort_by(|a, b| {
        b.annotations
            .cmp(&a.annotations)
            .then_with(|| a.category.cmp(&b.category))
    });

    if top_n < rows.len() {
        rows.truncate(top_n);
    }

    rows
}

/// Compute top category co-occurrence pairs.
fn compute_cooccurrence_top_pairs(
    dataset: &Dataset,
    category_names: &HashMap<CategoryId, String>,
    top_n: usize,
) -> CooccurrenceTopPairs {
    if top_n == 0 {
        return CooccurrenceTopPairs {
            top_n,
            pairs: Vec::new(),
        };
    }

    let mut per_image_categories: HashMap<ImageId, BTreeSet<String>> = HashMap::new();

    for ann in &dataset.annotations {
        let category = category_names
            .get(&ann.category_id)
            .cloned()
            .unwrap_or_else(|| format!("<missing cat {}>", ann.category_id));

        per_image_categories
            .entry(ann.image_id)
            .or_default()
            .insert(category);
    }

    let mut pair_counts: BTreeMap<(String, String), usize> = BTreeMap::new();

    for categories in per_image_categories.values() {
        let labels: Vec<&String> = categories.iter().collect();
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                let key = (labels[i].clone(), labels[j].clone());
                *pair_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    let mut pairs: Vec<CooccurrencePair> = pair_counts
        .into_iter()
        .map(|((a, b), count)| CooccurrencePair { a, b, count })
        .collect();

    pairs.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.a.cmp(&b.a))
            .then_with(|| a.b.cmp(&b.b))
    });

    if top_n < pairs.len() {
        pairs.truncate(top_n);
    }

    CooccurrenceTopPairs { top_n, pairs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Annotation, BBoxXYXY, Category, Image, Pixel};

    fn make_test_dataset() -> Dataset {
        Dataset {
            images: vec![
                Image::new(1u64, "img1.jpg", 640, 480),
                Image::new(2u64, "img2.jpg", 800, 600),
                Image::new(3u64, "img3.jpg", 1920, 1080),
            ],
            categories: vec![
                Category::new(1u64, "person"),
                Category::new(2u64, "car"),
                Category::new(3u64, "dog"),
            ],
            annotations: vec![
                Annotation::new(
                    1u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(10.0, 10.0, 100.0, 100.0),
                ),
                Annotation::new(
                    2u64,
                    1u64,
                    1u64,
                    BBoxXYXY::<Pixel>::from_xyxy(200.0, 200.0, 300.0, 300.0),
                ),
                Annotation::new(
                    3u64,
                    2u64,
                    2u64,
                    BBoxXYXY::<Pixel>::from_xyxy(50.0, 50.0, 150.0, 150.0),
                ),
                Annotation::new(
                    4u64,
                    2u64,
                    3u64,
                    BBoxXYXY::<Pixel>::from_xyxy(100.0, 100.0, 200.0, 200.0),
                ),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn test_summary_counts() {
        let dataset = make_test_dataset();
        let opts = StatsOptions::default();
        let report = stats_dataset(&dataset, &opts);

        assert_eq!(report.summary.images, 3);
        assert_eq!(report.summary.categories, 3);
        assert_eq!(report.summary.annotations, 4);
        assert_eq!(report.summary.annotated_images, 2);
    }

    #[test]
    fn test_label_histogram() {
        let dataset = make_test_dataset();
        let opts = StatsOptions::default();
        let report = stats_dataset(&dataset, &opts);

        assert_eq!(report.labels.total_distinct, 3);
        assert_eq!(report.labels.entries.len(), 3);
        assert_eq!(report.labels.entries[0].label, "person");
        assert_eq!(report.labels.entries[0].count, 2);
    }

    #[test]
    fn test_bbox_stats() {
        let dataset = make_test_dataset();
        let opts = StatsOptions::default();
        let report = stats_dataset(&dataset, &opts);

        assert_eq!(report.bboxes.total, 4);
        assert_eq!(report.bboxes.finite, 4);
        assert_eq!(report.bboxes.ordered, 4);
        assert_eq!(report.bboxes.out_of_bounds, 0);
        assert_eq!(report.bboxes.degenerate_area, 0);
        assert_eq!(report.bboxes.min_width, Some(90.0));
        assert_eq!(report.bboxes.max_width, Some(100.0));
    }

    #[test]
    fn test_extra_sections() {
        let dataset = make_test_dataset();
        let opts = StatsOptions::default();
        let report = stats_dataset(&dataset, &opts);

        assert_eq!(report.image_resolutions.min_w, 640);
        assert_eq!(report.image_resolutions.max_w, 1920);
        assert_eq!(report.annotation_density.min_per_image, 0);
        assert_eq!(report.annotation_density.max_per_image, 2);
        assert_eq!(report.area_distribution.small, 0);
        assert_eq!(report.area_distribution.medium, 1);
        assert_eq!(report.area_distribution.large, 3);
        assert_eq!(report.cooccurrence_top_pairs.pairs.len(), 1);
        assert_eq!(report.cooccurrence_top_pairs.pairs[0].a, "car");
        assert_eq!(report.cooccurrence_top_pairs.pairs[0].b, "dog");
        assert_eq!(report.cooccurrence_top_pairs.pairs[0].count, 1);
    }

    #[test]
    fn test_display_output() {
        let dataset = make_test_dataset();
        let opts = StatsOptions::default();
        let report = stats_dataset(&dataset, &opts);

        let output = format!("{}", report);
        assert!(output.contains("Dataset Stats Report"));
        assert!(output.contains("Summary"));
        assert!(output.contains("Labels"));
        assert!(output.contains("Bounding Boxes"));
        assert!(output.contains("person"));
    }
}
