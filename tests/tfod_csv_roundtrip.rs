//! Integration tests for TFOD CSV format.
//!
//! These tests verify that the TFOD CSV reader and writer work correctly.
//! Tests using fixture files run in CI; tests using large generated assets
//! are marked `#[ignore]` and run locally only.

use std::path::Path;

use panlabel::ir::io_tfod_csv::{from_tfod_csv_str, read_tfod_csv, to_tfod_csv_string};

/// Test that we can read the small TFOD CSV asset file.
#[test]
fn read_tfod_small_asset() {
    let path = Path::new("tests/fixtures/sample_valid.tfod.csv");
    let dataset = read_tfod_csv(path).expect("Failed to read TFOD CSV");

    // The small dataset should have some images, categories, and annotations
    assert!(!dataset.images.is_empty(), "Expected at least one image");
    assert!(
        !dataset.categories.is_empty(),
        "Expected at least one category"
    );
    assert!(
        !dataset.annotations.is_empty(),
        "Expected at least one annotation"
    );

    // Verify all annotations have valid references
    for ann in &dataset.annotations {
        assert!(
            dataset.images.iter().any(|img| img.id == ann.image_id),
            "Annotation {} references non-existent image {}",
            ann.id.as_u64(),
            ann.image_id.as_u64()
        );
        assert!(
            dataset
                .categories
                .iter()
                .any(|cat| cat.id == ann.category_id),
            "Annotation {} references non-existent category {}",
            ann.id.as_u64(),
            ann.category_id.as_u64()
        );
    }
}

/// Test that we can read the large TFOD CSV asset file.
#[test]
#[ignore] // Requires large generated dataset in assets/ (not committed)
fn read_tfod_large_asset() {
    let path = Path::new("assets/tfod_annotations.csv");
    let dataset = read_tfod_csv(path).expect("Failed to read large TFOD CSV");

    // The large dataset should have significantly more data
    assert!(
        dataset.images.len() >= 10,
        "Expected at least 10 images in large dataset"
    );
    assert!(
        dataset.annotations.len() >= 100,
        "Expected at least 100 annotations in large dataset"
    );
}

/// Test roundtrip: read -> write -> read preserves semantic content.
#[test]
fn tfod_roundtrip_small_asset() {
    let path = Path::new("tests/fixtures/sample_valid.tfod.csv");
    let original = read_tfod_csv(path).expect("Failed to read TFOD CSV");

    // Write to string
    let csv_string = to_tfod_csv_string(&original).expect("Failed to write TFOD CSV");

    // Read back
    let restored = from_tfod_csv_str(&csv_string).expect("Failed to parse written CSV");

    // Compare counts
    assert_eq!(
        original.images.len(),
        restored.images.len(),
        "Image count mismatch after roundtrip"
    );
    assert_eq!(
        original.categories.len(),
        restored.categories.len(),
        "Category count mismatch after roundtrip"
    );
    assert_eq!(
        original.annotations.len(),
        restored.annotations.len(),
        "Annotation count mismatch after roundtrip"
    );

    // Compare image filenames (should be same set)
    let orig_filenames: std::collections::HashSet<_> =
        original.images.iter().map(|i| &i.file_name).collect();
    let rest_filenames: std::collections::HashSet<_> =
        restored.images.iter().map(|i| &i.file_name).collect();
    assert_eq!(
        orig_filenames, rest_filenames,
        "Image filenames differ after roundtrip"
    );

    // Compare category names (should be same set)
    let orig_categories: std::collections::HashSet<_> =
        original.categories.iter().map(|c| &c.name).collect();
    let rest_categories: std::collections::HashSet<_> =
        restored.categories.iter().map(|c| &c.name).collect();
    assert_eq!(
        orig_categories, rest_categories,
        "Category names differ after roundtrip"
    );

    // Compare bbox coordinates (within tolerance)
    for (orig_ann, rest_ann) in original.annotations.iter().zip(restored.annotations.iter()) {
        let epsilon = 0.01; // Allow for floating point precision loss
        assert!(
            (orig_ann.bbox.xmin() - rest_ann.bbox.xmin()).abs() < epsilon,
            "xmin mismatch for annotation"
        );
        assert!(
            (orig_ann.bbox.ymin() - rest_ann.bbox.ymin()).abs() < epsilon,
            "ymin mismatch for annotation"
        );
        assert!(
            (orig_ann.bbox.xmax() - rest_ann.bbox.xmax()).abs() < epsilon,
            "xmax mismatch for annotation"
        );
        assert!(
            (orig_ann.bbox.ymax() - rest_ann.bbox.ymax()).abs() < epsilon,
            "ymax mismatch for annotation"
        );
    }
}

/// Test that bbox coordinates are within valid bounds after reading.
#[test]
fn tfod_bbox_bounds_valid() {
    let path = Path::new("tests/fixtures/sample_valid.tfod.csv");
    let dataset = read_tfod_csv(path).expect("Failed to read TFOD CSV");

    for ann in &dataset.annotations {
        // Find the image for this annotation
        let image = dataset
            .images
            .iter()
            .find(|img| img.id == ann.image_id)
            .expect("Annotation references missing image");

        // Pixel coordinates should be non-negative and within image bounds
        assert!(ann.bbox.xmin() >= 0.0, "xmin should be non-negative");
        assert!(ann.bbox.ymin() >= 0.0, "ymin should be non-negative");
        assert!(
            ann.bbox.xmax() <= image.width as f64,
            "xmax should be <= image width"
        );
        assert!(
            ann.bbox.ymax() <= image.height as f64,
            "ymax should be <= image height"
        );

        // Bbox should be properly ordered
        assert!(ann.bbox.xmin() <= ann.bbox.xmax(), "xmin should be <= xmax");
        assert!(ann.bbox.ymin() <= ann.bbox.ymax(), "ymin should be <= ymax");
    }
}
