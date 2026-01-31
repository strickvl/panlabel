//! Dataset validation for panlabel.
//!
//! This module provides comprehensive validation of datasets, checking for:
//! - Structural integrity (unique IDs, valid references)
//! - Data quality (non-empty names, valid dimensions)
//! - Geometric validity (proper bounding boxes, within image bounds)

mod report;

pub use report::{IssueCode, IssueContext, Severity, ValidationIssue, ValidationReport};

use std::collections::{HashMap, HashSet};

use crate::ir::{AnnotationId, CategoryId, Dataset, ImageId};

/// Options for validation behavior.
#[derive(Clone, Debug, Default)]
pub struct ValidateOptions {
    /// If true, treat warnings as errors.
    pub strict: bool,
}

/// Validates a dataset and returns a report of all issues found.
///
/// This function performs comprehensive validation including:
/// - Checking for duplicate IDs (images, annotations, categories)
/// - Verifying all references are valid (image_id, category_id in annotations)
/// - Validating image dimensions are positive
/// - Validating category and file names are non-empty
/// - Checking bounding box validity (finite, ordered, within bounds)
pub fn validate_dataset(dataset: &Dataset, _opts: &ValidateOptions) -> ValidationReport {
    let mut report = ValidationReport::new();

    // Build lookup maps for reference validation
    let image_ids: HashSet<ImageId> = dataset.images.iter().map(|i| i.id).collect();
    let category_ids: HashSet<CategoryId> = dataset.categories.iter().map(|c| c.id).collect();

    // Validate images
    validate_images(dataset, &mut report);

    // Validate categories
    validate_categories(dataset, &mut report);

    // Validate annotations
    validate_annotations(dataset, &image_ids, &category_ids, &mut report);

    report
}

/// Validates all images in the dataset.
fn validate_images(dataset: &Dataset, report: &mut ValidationReport) {
    let mut seen_ids: HashMap<ImageId, usize> = HashMap::new();

    for (idx, image) in dataset.images.iter().enumerate() {
        let id = image.id.as_u64();

        // Check for duplicate IDs
        if let Some(first_idx) = seen_ids.get(&image.id) {
            report.add(ValidationIssue::error(
                IssueCode::DuplicateImageId,
                format!(
                    "Duplicate image ID {} (first seen at index {})",
                    id, first_idx
                ),
                IssueContext::Image { id },
            ));
        } else {
            seen_ids.insert(image.id, idx);
        }

        // Check dimensions
        if image.width == 0 || image.height == 0 {
            report.add(ValidationIssue::error(
                IssueCode::InvalidImageDimensions,
                format!(
                    "Invalid dimensions {}x{} (must be positive)",
                    image.width, image.height
                ),
                IssueContext::Image { id },
            ));
        }

        // Check filename
        if image.file_name.is_empty() {
            report.add(ValidationIssue::warning(
                IssueCode::EmptyFileName,
                "Empty filename",
                IssueContext::Image { id },
            ));
        }
    }
}

/// Validates all categories in the dataset.
fn validate_categories(dataset: &Dataset, report: &mut ValidationReport) {
    let mut seen_ids: HashMap<CategoryId, usize> = HashMap::new();
    let mut seen_names: HashMap<&str, CategoryId> = HashMap::new();

    for (idx, category) in dataset.categories.iter().enumerate() {
        let id = category.id.as_u64();

        // Check for duplicate IDs
        if let Some(first_idx) = seen_ids.get(&category.id) {
            report.add(ValidationIssue::error(
                IssueCode::DuplicateCategoryId,
                format!(
                    "Duplicate category ID {} (first seen at index {})",
                    id, first_idx
                ),
                IssueContext::Category { id },
            ));
        } else {
            seen_ids.insert(category.id, idx);
        }

        // Check for empty name
        if category.name.is_empty() {
            report.add(ValidationIssue::warning(
                IssueCode::EmptyCategoryName,
                "Empty category name",
                IssueContext::Category { id },
            ));
        } else {
            // Check for duplicate names (warning only - may be intentional)
            if let Some(first_id) = seen_names.get(category.name.as_str()) {
                report.add(ValidationIssue::warning(
                    IssueCode::DuplicateCategoryName,
                    format!(
                        "Duplicate category name '{}' (also used by category {})",
                        category.name, first_id
                    ),
                    IssueContext::Category { id },
                ));
            } else {
                seen_names.insert(&category.name, category.id);
            }
        }
    }
}

/// Validates all annotations in the dataset.
fn validate_annotations(
    dataset: &Dataset,
    image_ids: &HashSet<ImageId>,
    category_ids: &HashSet<CategoryId>,
    report: &mut ValidationReport,
) {
    let mut seen_ids: HashMap<AnnotationId, usize> = HashMap::new();

    // Build image dimension lookup for bounds checking
    let image_dims: HashMap<ImageId, (u32, u32)> = dataset
        .images
        .iter()
        .map(|i| (i.id, (i.width, i.height)))
        .collect();

    for (idx, annotation) in dataset.annotations.iter().enumerate() {
        let id = annotation.id.as_u64();

        // Check for duplicate IDs
        if let Some(first_idx) = seen_ids.get(&annotation.id) {
            report.add(ValidationIssue::error(
                IssueCode::DuplicateAnnotationId,
                format!(
                    "Duplicate annotation ID {} (first seen at index {})",
                    id, first_idx
                ),
                IssueContext::Annotation { id },
            ));
        } else {
            seen_ids.insert(annotation.id, idx);
        }

        // Check image reference
        if !image_ids.contains(&annotation.image_id) {
            report.add(ValidationIssue::error(
                IssueCode::MissingImageRef,
                format!("References non-existent image {}", annotation.image_id),
                IssueContext::Annotation { id },
            ));
        }

        // Check category reference
        if !category_ids.contains(&annotation.category_id) {
            report.add(ValidationIssue::error(
                IssueCode::MissingCategoryRef,
                format!(
                    "References non-existent category {}",
                    annotation.category_id
                ),
                IssueContext::Annotation { id },
            ));
        }

        // Validate bounding box
        let bbox = &annotation.bbox;

        // Check for non-finite coordinates
        if !bbox.is_finite() {
            report.add(ValidationIssue::error(
                IssueCode::BBoxNotFinite,
                format!(
                    "Non-finite coordinates ({}, {}, {}, {})",
                    bbox.xmin(),
                    bbox.ymin(),
                    bbox.xmax(),
                    bbox.ymax()
                ),
                IssueContext::Annotation { id },
            ));
            continue; // Skip further bbox checks if coordinates are invalid
        }

        // Check ordering (min <= max)
        if !bbox.is_ordered() {
            report.add(ValidationIssue::error(
                IssueCode::InvalidBBoxOrdering,
                format!(
                    "Invalid ordering: min ({}, {}) should be <= max ({}, {})",
                    bbox.xmin(),
                    bbox.ymin(),
                    bbox.xmax(),
                    bbox.ymax()
                ),
                IssueContext::Annotation { id },
            ));
        }

        // Check area (should be positive)
        let area = bbox.area();
        if area <= 0.0 {
            report.add(ValidationIssue::warning(
                IssueCode::InvalidBBoxArea,
                format!("Zero or negative area: {:.2}", area),
                IssueContext::Annotation { id },
            ));
        }

        // Check bounds (if we have the image dimensions)
        if let Some((width, height)) = image_dims.get(&annotation.image_id) {
            let (w, h) = (*width as f64, *height as f64);

            // Allow small tolerance for floating point
            let tolerance = 0.5;

            if bbox.xmin() < -tolerance
                || bbox.ymin() < -tolerance
                || bbox.xmax() > w + tolerance
                || bbox.ymax() > h + tolerance
            {
                report.add(ValidationIssue::error(
                    IssueCode::BBoxOutOfBounds,
                    format!(
                        "Bounding box ({:.1}, {:.1}, {:.1}, {:.1}) extends outside image bounds (0, 0, {}, {})",
                        bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax(), width, height
                    ),
                    IssueContext::Annotation { id },
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Annotation, BBoxXYXY, Category, Dataset, Image, Pixel};

    fn valid_dataset() -> Dataset {
        Dataset {
            images: vec![Image::new(1u64, "image.jpg", 640, 480)],
            categories: vec![Category::new(1u64, "person")],
            annotations: vec![Annotation::new(
                1u64,
                1u64,
                1u64,
                BBoxXYXY::<Pixel>::from_xyxy(10.0, 20.0, 100.0, 200.0),
            )],
            ..Default::default()
        }
    }

    #[test]
    fn test_valid_dataset() {
        let dataset = valid_dataset();
        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert!(
            report.is_clean(),
            "Expected no issues, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn test_duplicate_image_id() {
        let mut dataset = valid_dataset();
        dataset
            .images
            .push(Image::new(1u64, "duplicate.jpg", 640, 480));

        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert_eq!(report.error_count(), 1);
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::DuplicateImageId));
    }

    #[test]
    fn test_duplicate_annotation_id() {
        let mut dataset = valid_dataset();
        dataset.annotations.push(Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::from_xyxy(50.0, 60.0, 150.0, 160.0),
        ));

        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert_eq!(report.error_count(), 1);
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::DuplicateAnnotationId));
    }

    #[test]
    fn test_missing_image_ref() {
        let mut dataset = valid_dataset();
        dataset.annotations.push(Annotation::new(
            2u64,
            999u64,
            1u64, // non-existent image 999
            BBoxXYXY::<Pixel>::from_xyxy(10.0, 10.0, 50.0, 50.0),
        ));

        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert_eq!(report.error_count(), 1);
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::MissingImageRef));
    }

    #[test]
    fn test_missing_category_ref() {
        let mut dataset = valid_dataset();
        dataset.annotations.push(Annotation::new(
            2u64,
            1u64,
            999u64, // non-existent category 999
            BBoxXYXY::<Pixel>::from_xyxy(10.0, 10.0, 50.0, 50.0),
        ));

        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert_eq!(report.error_count(), 1);
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::MissingCategoryRef));
    }

    #[test]
    fn test_invalid_image_dimensions() {
        // Create a dataset with no annotations so we only test dimension validation
        let dataset = Dataset {
            images: vec![Image::new(1u64, "image.jpg", 0, 480)], // width = 0
            categories: vec![Category::new(1u64, "person")],
            annotations: vec![], // no annotations to avoid bbox-out-of-bounds errors
            ..Default::default()
        };

        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert_eq!(report.error_count(), 1);
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::InvalidImageDimensions));
    }

    #[test]
    fn test_bbox_out_of_bounds() {
        let mut dataset = valid_dataset();
        dataset.annotations[0].bbox = BBoxXYXY::<Pixel>::from_xyxy(600.0, 400.0, 800.0, 600.0);

        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::BBoxOutOfBounds));
    }

    #[test]
    fn test_bbox_invalid_ordering() {
        let mut dataset = valid_dataset();
        // xmax < xmin
        dataset.annotations[0].bbox = BBoxXYXY::<Pixel>::from_xyxy(100.0, 20.0, 10.0, 200.0);

        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::InvalidBBoxOrdering));
    }

    #[test]
    fn test_bbox_not_finite() {
        let mut dataset = valid_dataset();
        dataset.annotations[0].bbox = BBoxXYXY::<Pixel>::from_xyxy(f64::NAN, 20.0, 100.0, 200.0);

        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::BBoxNotFinite));
    }

    #[test]
    fn test_empty_category_name() {
        let mut dataset = valid_dataset();
        dataset.categories[0].name = String::new();

        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert_eq!(report.warning_count(), 1);
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::EmptyCategoryName));
    }

    #[test]
    fn test_duplicate_category_name() {
        let mut dataset = valid_dataset();
        dataset.categories.push(Category::new(2u64, "person")); // same name

        let report = validate_dataset(&dataset, &ValidateOptions::default());
        assert_eq!(report.warning_count(), 1);
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::DuplicateCategoryName));
    }
}
