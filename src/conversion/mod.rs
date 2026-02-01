//! Conversion module for format transformation reporting.
//!
//! This module provides structured reporting for conversions between
//! annotation formats, tracking what information is preserved, lost,
//! or transformed according to deterministic policies.

pub mod report;

pub use report::{
    ConversionCounts, ConversionIssue, ConversionIssueCode, ConversionReport, ConversionSeverity,
};

use crate::ir::Dataset;
use std::collections::HashSet;

/// Format identifier for conversion reporting.
///
/// This mirrors the CLI's ConvertFormat but is decoupled from clap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    IrJson,
    Coco,
    Tfod,
}

/// Classification of how lossy a format is relative to the IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrLossiness {
    /// Format can represent everything in the IR (round-trip safe).
    Lossless,
    /// Format may lose some information depending on dataset content.
    Conditional,
    /// Format always loses some IR information.
    Lossy,
}

impl Format {
    /// Human-readable name for the format.
    pub fn name(&self) -> &'static str {
        match self {
            Format::IrJson => "ir-json",
            Format::Coco => "coco",
            Format::Tfod => "tfod",
        }
    }

    /// How lossy this format is relative to the IR.
    ///
    /// - `IrJson`: Lossless (it IS the IR)
    /// - `Coco`: Conditional (loses dataset name, may lose some attributes)
    /// - `Tfod`: Lossy (loses metadata, licenses, images without annotations, etc.)
    pub fn lossiness_relative_to_ir(&self) -> IrLossiness {
        match self {
            Format::IrJson => IrLossiness::Lossless,
            Format::Coco => IrLossiness::Conditional,
            Format::Tfod => IrLossiness::Lossy,
        }
    }
}

/// Build a conversion report analyzing what will happen during conversion.
///
/// This function examines the dataset and target format to determine:
/// - Input/output counts
/// - What information will be lost (warnings)
/// - What policy decisions apply (info notes)
pub fn build_conversion_report(dataset: &Dataset, from: Format, to: Format) -> ConversionReport {
    let mut report = ConversionReport::new(from.name(), to.name());

    // Set input counts
    report.input = ConversionCounts {
        images: dataset.images.len(),
        categories: dataset.categories.len(),
        annotations: dataset.annotations.len(),
    };

    // Compute output counts and issues based on target format
    match to {
        Format::Tfod => analyze_to_tfod(dataset, &mut report),
        Format::Coco => analyze_to_coco(dataset, &mut report),
        Format::IrJson => analyze_to_ir_json(dataset, &mut report),
    }

    // Add policy notes based on source format
    match from {
        Format::Tfod => add_tfod_reader_policy(&mut report),
        Format::Coco | Format::IrJson => {}
    }

    // Add policy notes based on target format
    if to == Format::Tfod {
        add_tfod_writer_policy(&mut report);
    }

    report
}

/// Analyze conversion to TFOD format.
fn analyze_to_tfod(dataset: &Dataset, report: &mut ConversionReport) {
    // TFOD cannot represent dataset info/metadata
    if !dataset.info.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropDatasetInfo,
            "dataset info/metadata will be dropped".to_string(),
        ));
    }

    // TFOD cannot represent licenses
    if !dataset.licenses.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropLicenses,
            format!("{} license(s) will be dropped", dataset.licenses.len()),
        ));
    }

    // TFOD cannot represent image license_id or date_captured
    let images_with_metadata = dataset
        .images
        .iter()
        .filter(|img| img.license_id.is_some() || img.date_captured.is_some())
        .count();
    if images_with_metadata > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropImageMetadata,
            format!(
                "{} image(s) have license_id/date_captured that will be dropped",
                images_with_metadata
            ),
        ));
    }

    // TFOD cannot represent category supercategory
    let cats_with_supercategory = dataset
        .categories
        .iter()
        .filter(|cat| cat.supercategory.is_some())
        .count();
    if cats_with_supercategory > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropCategorySupercategory,
            format!(
                "{} category(s) have supercategory that will be dropped",
                cats_with_supercategory
            ),
        ));
    }

    // TFOD cannot represent annotation confidence
    let anns_with_confidence = dataset
        .annotations
        .iter()
        .filter(|ann| ann.confidence.is_some())
        .count();
    if anns_with_confidence > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationConfidence,
            format!(
                "{} annotation(s) have confidence scores that will be dropped",
                anns_with_confidence
            ),
        ));
    }

    // TFOD cannot represent annotation attributes
    let anns_with_attributes = dataset
        .annotations
        .iter()
        .filter(|ann| !ann.attributes.is_empty())
        .count();
    if anns_with_attributes > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationAttributes,
            format!(
                "{} annotation(s) have attributes that will be dropped",
                anns_with_attributes
            ),
        ));
    }

    // Images without annotations won't appear in TFOD output
    let image_ids_with_annotations: HashSet<_> =
        dataset.annotations.iter().map(|a| a.image_id).collect();
    let images_without_annotations = dataset
        .images
        .iter()
        .filter(|img| !image_ids_with_annotations.contains(&img.id))
        .count();
    if images_without_annotations > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropImagesWithoutAnnotations,
            format!(
                "{} image(s) have no annotations and will not appear in output",
                images_without_annotations
            ),
        ));
    }

    // Compute output counts for TFOD
    // TFOD only outputs images/categories that have annotations
    let distinct_image_ids: HashSet<_> = dataset.annotations.iter().map(|a| a.image_id).collect();
    let distinct_category_ids: HashSet<_> =
        dataset.annotations.iter().map(|a| a.category_id).collect();

    report.output = ConversionCounts {
        images: distinct_image_ids.len(),
        categories: distinct_category_ids.len(),
        annotations: dataset.annotations.len(),
    };
}

/// Analyze conversion to COCO format.
fn analyze_to_coco(dataset: &Dataset, report: &mut ConversionReport) {
    // COCO doesn't have a dataset name field
    if dataset.info.name.is_some() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropDatasetInfoName,
            "dataset info.name has no COCO equivalent".to_string(),
        ));
    }

    // COCO round-trips area/iscrowd via attributes, but other attributes may be lost
    let anns_with_other_attributes = dataset
        .annotations
        .iter()
        .filter(|ann| ann.attributes.keys().any(|k| k != "area" && k != "iscrowd"))
        .count();
    if anns_with_other_attributes > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::CocoAttributesMayNotBePreserved,
            format!(
                "{} annotation(s) have attributes (other than area/iscrowd) that may not be preserved by COCO tools",
                anns_with_other_attributes
            ),
        ));
    }

    // COCO preserves all images, categories, and annotations
    report.output = report.input.clone();
}

/// Analyze conversion to IR JSON format.
fn analyze_to_ir_json(_dataset: &Dataset, report: &mut ConversionReport) {
    // IR JSON is the canonical format - no lossiness
    report.output = report.input.clone();
}

/// Add policy notes for TFOD reader behavior.
fn add_tfod_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::TfodReaderIdAssignment,
        "TFOD reader assigns IDs deterministically: images by filename (lexicographic), \
         categories by class name (lexicographic), annotations by CSV row order"
            .to_string(),
    ));
}

/// Add policy notes for TFOD writer behavior.
fn add_tfod_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::TfodWriterRowOrder,
        "TFOD writer orders rows by annotation ID for deterministic output".to_string(),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        Annotation, AnnotationId, BBoxXYXY, Category, CategoryId, Coord, DatasetInfo, Image,
        ImageId, License, LicenseId, Pixel,
    };

    fn sample_dataset() -> Dataset {
        Dataset {
            info: DatasetInfo {
                name: Some("Test Dataset".to_string()),
                ..Default::default()
            },
            licenses: vec![License {
                id: LicenseId(1),
                name: "CC0".to_string(),
                url: None,
            }],
            images: vec![
                Image {
                    id: ImageId(1),
                    file_name: "img1.jpg".to_string(),
                    width: 100,
                    height: 100,
                    license_id: Some(LicenseId(1)),
                    date_captured: None,
                },
                Image {
                    id: ImageId(2),
                    file_name: "img2.jpg".to_string(),
                    width: 100,
                    height: 100,
                    license_id: None,
                    date_captured: None,
                },
            ],
            categories: vec![Category {
                id: CategoryId(1),
                name: "cat".to_string(),
                supercategory: Some("animal".to_string()),
            }],
            annotations: vec![Annotation {
                id: AnnotationId(1),
                image_id: ImageId(1),
                category_id: CategoryId(1),
                bbox: BBoxXYXY::<Pixel>::new(Coord::new(10.0, 10.0), Coord::new(50.0, 50.0)),
                confidence: Some(0.95),
                attributes: [("custom".to_string(), "value".to_string())]
                    .into_iter()
                    .collect(),
            }],
        }
    }

    #[test]
    fn to_tfod_detects_all_lossiness() {
        let dataset = sample_dataset();
        let report = build_conversion_report(&dataset, Format::Coco, Format::Tfod);

        assert!(report.is_lossy());
        // Should detect: info, licenses, image metadata, supercategory, confidence, attributes, images without annotations
        assert!(report.warning_count() >= 6);
    }

    #[test]
    fn to_ir_json_is_not_lossy() {
        let dataset = sample_dataset();
        let report = build_conversion_report(&dataset, Format::Coco, Format::IrJson);

        assert!(!report.is_lossy());
        assert_eq!(report.warning_count(), 0);
    }

    #[test]
    fn to_coco_detects_name_lossiness() {
        let dataset = sample_dataset();
        let report = build_conversion_report(&dataset, Format::IrJson, Format::Coco);

        assert!(report.is_lossy());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::DropDatasetInfoName));
    }

    #[test]
    fn tfod_source_adds_policy_note() {
        let dataset = Dataset::default();
        let report = build_conversion_report(&dataset, Format::Tfod, Format::Coco);

        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::TfodReaderIdAssignment));
    }

    #[test]
    fn tfod_target_adds_policy_note() {
        let dataset = Dataset::default();
        let report = build_conversion_report(&dataset, Format::Coco, Format::Tfod);

        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::TfodWriterRowOrder));
    }

    #[test]
    fn output_counts_differ_for_tfod() {
        let dataset = sample_dataset(); // Has 2 images but only 1 with annotations
        let report = build_conversion_report(&dataset, Format::Coco, Format::Tfod);

        assert_eq!(report.input.images, 2);
        assert_eq!(report.output.images, 1); // Only image with annotations
    }
}
