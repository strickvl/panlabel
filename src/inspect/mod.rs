//! Dataset inspection and statistics.
//!
//! This module provides functionality to analyze datasets and produce
//! structured reports with summary statistics, label distributions,
//! and bounding box quality metrics.

mod report;

pub use report::{BBoxStats, InspectReport, LabelCount, LabelsSection, SummarySection};

use std::collections::{HashMap, HashSet};

use crate::ir::{CategoryId, Dataset, ImageId};

/// Options for dataset inspection.
#[derive(Clone, Debug)]
pub struct InspectOptions {
    /// Number of top labels to show in the histogram.
    pub top_labels: usize,
    /// Tolerance in pixels for out-of-bounds checks.
    pub oob_tolerance_px: f64,
    /// Width of histogram bars (in characters).
    pub bar_width: usize,
}

impl Default for InspectOptions {
    fn default() -> Self {
        Self {
            top_labels: 10,
            oob_tolerance_px: 0.5,
            bar_width: 20,
        }
    }
}

/// Inspect a dataset and produce a detailed report.
///
/// This analyzes the dataset to compute:
/// - Summary counts (images, categories, annotations, licenses)
/// - Label distribution histogram (top N categories)
/// - Bounding box statistics (dimensions, quality metrics)
pub fn inspect_dataset(dataset: &Dataset, opts: &InspectOptions) -> InspectReport {
    // Build lookup tables
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

    // Compute summary
    let summary = compute_summary(dataset);

    // Compute label histogram
    let labels = compute_labels(dataset, &category_names, opts.top_labels);

    // Compute bbox stats
    let bboxes = compute_bbox_stats(dataset, &image_dims, opts.oob_tolerance_px);

    InspectReport {
        summary,
        labels,
        bboxes,
        bar_width: opts.bar_width,
    }
}

/// Compute summary section counts.
fn compute_summary(dataset: &Dataset) -> SummarySection {
    // Count images that have at least one annotation
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
    // Count annotations per category
    let mut counts: HashMap<String, usize> = HashMap::new();

    for ann in &dataset.annotations {
        let label = category_names
            .get(&ann.category_id)
            .cloned()
            .unwrap_or_else(|| format!("<missing cat {}>", ann.category_id));

        *counts.entry(label).or_insert(0) += 1;
    }

    // Sort by count descending, then by name ascending for deterministic output
    let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| {
        b.1.cmp(&a.1) // Primary: count descending
            .then_with(|| a.0.cmp(&b.0)) // Secondary: name ascending
    });

    let total_distinct = sorted.len();
    let total_annotations = dataset.annotations.len();

    // Split into top N and "other"
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

    // Track min/max for width/height
    let mut min_width: Option<f64> = None;
    let mut max_width: Option<f64> = None;
    let mut min_height: Option<f64> = None;
    let mut max_height: Option<f64> = None;

    for ann in &dataset.annotations {
        let bbox = &ann.bbox;

        // Check if coordinates are finite
        let xmin = bbox.min.x;
        let ymin = bbox.min.y;
        let xmax = bbox.max.x;
        let ymax = bbox.max.y;

        let is_finite =
            xmin.is_finite() && ymin.is_finite() && xmax.is_finite() && ymax.is_finite();

        if is_finite {
            stats.finite += 1;

            // Check ordering
            let is_ordered = xmin <= xmax && ymin <= ymax;
            if is_ordered {
                stats.ordered += 1;

                // Compute dimensions
                let width = xmax - xmin;
                let height = ymax - ymin;

                // Update min/max
                min_width = Some(min_width.map_or(width, |m| m.min(width)));
                max_width = Some(max_width.map_or(width, |m| m.max(width)));
                min_height = Some(min_height.map_or(height, |m| m.min(height)));
                max_height = Some(max_height.map_or(height, |m| m.max(height)));

                // Check for degenerate area
                let area = width * height;
                if area <= 0.0 {
                    stats.degenerate_area += 1;
                }
            }

            // Check out-of-bounds (requires image dimensions)
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
        } else {
            // Non-finite coordinates also count as missing image ref check
            if !image_dims.contains_key(&ann.image_id) {
                stats.missing_image_ref += 1;
            }
        }
    }

    stats.min_width = min_width;
    stats.max_width = max_width;
    stats.min_height = min_height;
    stats.max_height = max_height;

    stats
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
        let opts = InspectOptions::default();
        let report = inspect_dataset(&dataset, &opts);

        assert_eq!(report.summary.images, 3);
        assert_eq!(report.summary.categories, 3);
        assert_eq!(report.summary.annotations, 4);
        assert_eq!(report.summary.annotated_images, 2); // img1 and img2 have annotations
    }

    #[test]
    fn test_label_histogram() {
        let dataset = make_test_dataset();
        let opts = InspectOptions::default();
        let report = inspect_dataset(&dataset, &opts);

        assert_eq!(report.labels.total_distinct, 3);
        assert_eq!(report.labels.entries.len(), 3);

        // "person" has 2 annotations, should be first
        assert_eq!(report.labels.entries[0].label, "person");
        assert_eq!(report.labels.entries[0].count, 2);
    }

    #[test]
    fn test_bbox_stats() {
        let dataset = make_test_dataset();
        let opts = InspectOptions::default();
        let report = inspect_dataset(&dataset, &opts);

        assert_eq!(report.bboxes.total, 4);
        assert_eq!(report.bboxes.finite, 4);
        assert_eq!(report.bboxes.ordered, 4);
        assert_eq!(report.bboxes.out_of_bounds, 0);
        assert_eq!(report.bboxes.degenerate_area, 0);

        // All boxes are 90x90 pixels
        assert_eq!(report.bboxes.min_width, Some(90.0));
        assert_eq!(report.bboxes.max_width, Some(100.0));
    }

    #[test]
    fn test_display_output() {
        let dataset = make_test_dataset();
        let opts = InspectOptions::default();
        let report = inspect_dataset(&dataset, &opts);

        let output = format!("{}", report);

        // Check that key sections are present
        assert!(output.contains("Dataset Inspection Report"));
        assert!(output.contains("Summary"));
        assert!(output.contains("Labels"));
        assert!(output.contains("Bounding Boxes"));
        assert!(output.contains("person"));
    }
}
