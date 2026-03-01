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
    Cvat,
    LabelStudio,
    Tfod,
    Yolo,
    Voc,
    HfImagefolder,
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
            Format::Cvat => "cvat",
            Format::LabelStudio => "label-studio",
            Format::Tfod => "tfod",
            Format::Yolo => "yolo",
            Format::Voc => "voc",
            Format::HfImagefolder => "hf",
        }
    }

    /// How lossy this format is relative to the IR.
    ///
    /// - `IrJson`: Lossless (it IS the IR)
    /// - `Coco`: Conditional (loses dataset name, may lose some attributes)
    /// - `LabelStudio`: Lossy (drops IR-level metadata fields not representable in task export)
    /// - `Tfod`: Lossy (loses metadata, licenses, images without annotations, etc.)
    /// - `Yolo`: Lossy (loses metadata, licenses, attributes, etc.)
    /// - `Voc`: Lossy (loses metadata, licenses, supercategory, confidence, etc.)
    pub fn lossiness_relative_to_ir(&self) -> IrLossiness {
        match self {
            Format::IrJson => IrLossiness::Lossless,
            Format::Coco => IrLossiness::Conditional,
            Format::Cvat => IrLossiness::Lossy,
            Format::LabelStudio => IrLossiness::Lossy,
            Format::Tfod => IrLossiness::Lossy,
            Format::Yolo => IrLossiness::Lossy,
            Format::Voc => IrLossiness::Lossy,
            Format::HfImagefolder => IrLossiness::Lossy,
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
        Format::Yolo => analyze_to_yolo(dataset, &mut report),
        Format::Voc => analyze_to_voc(dataset, &mut report),
        Format::LabelStudio => analyze_to_label_studio(dataset, &mut report),
        Format::Coco => analyze_to_coco(dataset, &mut report),
        Format::Cvat => analyze_to_cvat(dataset, &mut report),
        Format::IrJson => analyze_to_ir_json(dataset, &mut report),
        Format::HfImagefolder => analyze_to_hf(dataset, &mut report),
    }

    // Add policy notes based on source format
    match from {
        Format::Tfod => add_tfod_reader_policy(&mut report),
        Format::Yolo => add_yolo_reader_policy(&mut report),
        Format::Voc => add_voc_reader_policy(dataset, &mut report),
        Format::LabelStudio => add_label_studio_reader_policy(dataset, &mut report),
        Format::Cvat => add_cvat_reader_policy(dataset, &mut report),
        Format::HfImagefolder => add_hf_reader_policy(&mut report),
        Format::Coco | Format::IrJson => {}
    }

    // Add policy notes based on target format
    match to {
        Format::Tfod => add_tfod_writer_policy(&mut report),
        Format::Yolo => add_yolo_writer_policy(&mut report),
        Format::Voc => add_voc_writer_policy(&mut report),
        Format::LabelStudio => add_label_studio_writer_policy(dataset, &mut report),
        Format::Cvat => add_cvat_writer_policy(&mut report),
        Format::HfImagefolder => add_hf_writer_policy(&mut report),
        Format::Coco | Format::IrJson => {}
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

/// Analyze conversion to YOLO format.
fn analyze_to_yolo(dataset: &Dataset, report: &mut ConversionReport) {
    // YOLO cannot represent dataset info/metadata
    if !dataset.info.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropDatasetInfo,
            "dataset info/metadata will be dropped".to_string(),
        ));
    }

    // YOLO cannot represent licenses
    if !dataset.licenses.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropLicenses,
            format!("{} license(s) will be dropped", dataset.licenses.len()),
        ));
    }

    // YOLO cannot represent image license_id or date_captured
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

    // YOLO cannot represent category supercategory
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

    // YOLO cannot represent annotation confidence
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

    // YOLO cannot represent annotation attributes
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

    // YOLO keeps full counts (including images without annotations via empty label files).
    report.output = report.input.clone();
}

/// Analyze conversion to VOC format.
fn analyze_to_voc(dataset: &Dataset, report: &mut ConversionReport) {
    if !dataset.info.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropDatasetInfo,
            "dataset info/metadata will be dropped".to_string(),
        ));
    }

    if !dataset.licenses.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropLicenses,
            format!("{} license(s) will be dropped", dataset.licenses.len()),
        ));
    }

    let images_with_metadata = dataset
        .images
        .iter()
        .filter(|img| {
            img.license_id.is_some()
                || img.date_captured.is_some()
                || img
                    .attributes
                    .iter()
                    .any(|(key, value)| key != "depth" || value.trim().parse::<u32>().is_err())
        })
        .count();
    if images_with_metadata > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropImageMetadata,
            format!(
                "{} image(s) have metadata that VOC cannot represent (license/date or non-depth image attributes)",
                images_with_metadata
            ),
        ));
    }

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

    let anns_with_unrepresentable_attrs = dataset
        .annotations
        .iter()
        .filter(|ann| {
            ann.attributes.keys().any(|key| {
                !matches!(
                    key.as_str(),
                    "pose" | "truncated" | "difficult" | "occluded"
                )
            })
        })
        .count();
    if anns_with_unrepresentable_attrs > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationAttributes,
            format!(
                "{} annotation(s) have attributes outside VOC's preserved set (pose/truncated/difficult/occluded)",
                anns_with_unrepresentable_attrs
            ),
        ));
    }

    report.output = report.input.clone();
}

/// Analyze conversion to Label Studio format.
fn analyze_to_label_studio(dataset: &Dataset, report: &mut ConversionReport) {
    if !dataset.info.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropDatasetInfo,
            "dataset info/metadata will be dropped".to_string(),
        ));
    }

    if !dataset.licenses.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropLicenses,
            format!("{} license(s) will be dropped", dataset.licenses.len()),
        ));
    }

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

    let anns_with_unrepresentable_attrs = dataset
        .annotations
        .iter()
        .filter(|ann| {
            ann.attributes
                .keys()
                .any(|key| key.as_str() != "ls_rotation_deg")
        })
        .count();
    if anns_with_unrepresentable_attrs > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationAttributes,
            format!(
                "{} annotation(s) have attributes outside Label Studio's preserved set",
                anns_with_unrepresentable_attrs
            ),
        ));
    }

    report.output = report.input.clone();
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

/// Analyze conversion to HF ImageFolder metadata format.
fn analyze_to_hf(dataset: &Dataset, report: &mut ConversionReport) {
    if !dataset.info.is_empty() || !dataset.licenses.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::HfMetadataLost,
            "HF ImageFolder metadata.jsonl cannot represent full IR dataset metadata/licenses"
                .to_string(),
        ));
    }

    let images_with_unrepresentable_attrs = dataset
        .images
        .iter()
        .filter(|img| {
            img.license_id.is_some() || img.date_captured.is_some() || !img.attributes.is_empty()
        })
        .count();
    if images_with_unrepresentable_attrs > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::HfAttributesLost,
            format!(
                "{} image(s) have metadata/attributes that HF metadata.jsonl cannot represent",
                images_with_unrepresentable_attrs
            ),
        ));
    }

    let categories_with_supercategory = dataset
        .categories
        .iter()
        .filter(|category| category.supercategory.is_some())
        .count();
    if categories_with_supercategory > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::HfMetadataLost,
            format!(
                "{} category(s) have supercategory that HF metadata.jsonl cannot represent",
                categories_with_supercategory
            ),
        ));
    }

    let anns_with_confidence = dataset
        .annotations
        .iter()
        .filter(|ann| ann.confidence.is_some())
        .count();
    if anns_with_confidence > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::HfConfidenceLost,
            format!(
                "{} annotation(s) have confidence scores that will be dropped",
                anns_with_confidence
            ),
        ));
    }

    let anns_with_attributes = dataset
        .annotations
        .iter()
        .filter(|ann| !ann.attributes.is_empty())
        .count();
    if anns_with_attributes > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::HfAttributesLost,
            format!(
                "{} annotation(s) have attributes that will be dropped",
                anns_with_attributes
            ),
        ));
    }

    report.output = report.input.clone();
}

/// Analyze conversion to CVAT XML format.
fn analyze_to_cvat(dataset: &Dataset, report: &mut ConversionReport) {
    if !dataset.info.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropDatasetInfo,
            "dataset info/metadata will be dropped".to_string(),
        ));
    }

    if !dataset.licenses.is_empty() {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropLicenses,
            format!("{} license(s) will be dropped", dataset.licenses.len()),
        ));
    }

    let images_with_metadata = dataset
        .images
        .iter()
        .filter(|img| {
            img.license_id.is_some() || img.date_captured.is_some() || !img.attributes.is_empty()
        })
        .count();
    if images_with_metadata > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropImageMetadata,
            format!(
                "{} image(s) have metadata that CVAT cannot represent (license/date/image attributes)",
                images_with_metadata
            ),
        ));
    }

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

    let anns_with_unrepresentable_attrs = dataset
        .annotations
        .iter()
        .filter(|ann| {
            ann.attributes.keys().any(|key| {
                !(key == "occluded"
                    || key == "z_order"
                    || key == "source"
                    || key.starts_with("cvat_attr_"))
            })
        })
        .count();
    if anns_with_unrepresentable_attrs > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationAttributes,
            format!(
                "{} annotation(s) have attributes outside CVAT's preserved set (occluded/z_order/source/cvat_attr_*)",
                anns_with_unrepresentable_attrs
            ),
        ));
    }

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

/// Add policy notes for YOLO reader behavior.
fn add_yolo_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::YoloReaderIdAssignment,
        "YOLO reader assigns IDs deterministically: images by relative path (lexicographic), categories by class index, annotations by label-file order then line number".to_string(),
    ));
    report.add(ConversionIssue::info(
        ConversionIssueCode::YoloReaderClassMapSource,
        "YOLO reader class map source precedence: data.yaml, then classes.txt, then inferred from label files".to_string(),
    ));
}

/// Add policy notes for YOLO writer behavior.
fn add_yolo_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::YoloWriterClassOrder,
        "YOLO writer assigns class indices by CategoryId order (sorted ascending)".to_string(),
    ));
    report.add(ConversionIssue::info(
        ConversionIssueCode::YoloWriterEmptyLabelFiles,
        "YOLO writer creates empty .txt files for images without annotations".to_string(),
    ));
    report.add(ConversionIssue::info(
        ConversionIssueCode::YoloWriterFloatPrecision,
        "YOLO writer outputs normalized coordinates with 6 decimal places".to_string(),
    ));
}

/// Add policy notes for VOC reader behavior.
fn add_voc_reader_policy(dataset: &Dataset, report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::VocReaderIdAssignment,
        "VOC reader assigns IDs deterministically: images by <filename> (lexicographic), categories by class name (lexicographic), annotations by XML file order then <object> order".to_string(),
    ));
    report.add(ConversionIssue::info(
        ConversionIssueCode::VocReaderAttributeMapping,
        "VOC reader maps pose/truncated/difficult/occluded into annotation attributes".to_string(),
    ));
    report.add(ConversionIssue::info(
        ConversionIssueCode::VocReaderCoordinatePolicy,
        "VOC reader keeps bndbox coordinates exactly as provided (no 0/1-based adjustment)"
            .to_string(),
    ));

    let has_non_rgb_depth = dataset
        .images
        .iter()
        .filter_map(|image| image.attributes.get("depth"))
        .filter_map(|depth| depth.parse::<u32>().ok())
        .any(|depth| depth != 3);
    if has_non_rgb_depth {
        report.add(ConversionIssue::info(
            ConversionIssueCode::VocReaderDepthHandling,
            "VOC reader preserves <depth> as image attribute 'depth'; non-3 depth values may indicate non-RGB imagery"
                .to_string(),
        ));
    }
}

/// Add policy notes for VOC writer behavior.
fn add_voc_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::VocWriterFileLayout,
        "VOC writer emits one XML per image under Annotations/, preserving image subdirectory structure"
            .to_string(),
    ));
    report.add(ConversionIssue::info(
        ConversionIssueCode::VocWriterNoImageCopy,
        "VOC writer creates JPEGImages/README.txt but does not copy image binaries".to_string(),
    ));
    report.add(ConversionIssue::info(
        ConversionIssueCode::VocWriterBoolNormalization,
        "VOC writer normalizes truncated/difficult/occluded attributes: true/yes/1 -> 1 and false/no/0 -> 0"
            .to_string(),
    ));
}

/// Add policy notes for Label Studio reader behavior.
fn add_label_studio_reader_policy(dataset: &Dataset, report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::LabelStudioReaderIdAssignment,
        "Label Studio reader assigns IDs deterministically: images by derived file_name (lexicographic), categories by label (lexicographic), annotations by image order then result order".to_string(),
    ));
    report.add(ConversionIssue::info(
        ConversionIssueCode::LabelStudioReaderImageRefPolicy,
        "Label Studio reader derives Image.file_name from data.image basename and preserves full source reference in image attribute ls_image_ref".to_string(),
    ));

    let has_rotation = dataset
        .annotations
        .iter()
        .any(|ann| ann.attributes.contains_key("ls_rotation_deg"));
    if has_rotation {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::LabelStudioRotationDropped,
            "Label Studio rotated bbox converted to axis-aligned envelope (original angle stored in annotation attribute ls_rotation_deg)".to_string(),
        ));
    }
}

/// Add policy notes for Label Studio writer behavior.
fn add_label_studio_writer_policy(dataset: &Dataset, report: &mut ConversionReport) {
    let used_defaults = dataset.images.iter().any(|image| {
        !image.attributes.contains_key("ls_from_name")
            || !image.attributes.contains_key("ls_to_name")
    });

    if used_defaults {
        report.add(ConversionIssue::info(
            ConversionIssueCode::LabelStudioWriterFromToDefaults,
            "Label Studio writer uses from_name='label' and to_name='image' when ls_from_name/ls_to_name attributes are absent".to_string(),
        ));
    }
}

/// Add policy notes for CVAT reader behavior.
fn add_cvat_reader_policy(_dataset: &Dataset, report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::CvatReaderIdAssignment,
        "CVAT reader assigns IDs deterministically: images by <image name> (lexicographic), categories by label name (lexicographic), annotations by image order then <box> order".to_string(),
    ));
    report.add(ConversionIssue::info(
        ConversionIssueCode::CvatReaderAttributePolicy,
        "CVAT reader maps xtl/ytl/xbr/ybr to IR pixel XYXY 1:1; custom <attribute> children are stored as annotation attributes with 'cvat_attr_' prefix".to_string(),
    ));
}

/// Add policy notes for CVAT writer behavior.
fn add_cvat_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::CvatWriterMetaDefaults,
        "CVAT writer emits a minimal <meta><task> block with name='panlabel export' and writes labels only for categories referenced by annotations".to_string(),
    ));
}

/// Add policy notes for HF reader behavior.
fn add_hf_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::HfReaderCategoryResolution,
        "HF reader resolves category names with precedence: ClassLabel/preflight map, then --hf-category-map, then integer fallback"
            .to_string(),
    ));
}

/// Add policy notes for HF writer behavior.
fn add_hf_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::info(
        ConversionIssueCode::HfWriterDeterministicOrder,
        "HF writer orders metadata rows by image file_name and annotation lists by annotation ID"
            .to_string(),
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
                    attributes: std::collections::BTreeMap::new(),
                },
                Image {
                    id: ImageId(2),
                    file_name: "img2.jpg".to_string(),
                    width: 100,
                    height: 100,
                    license_id: None,
                    date_captured: None,
                    attributes: std::collections::BTreeMap::new(),
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

    #[test]
    fn yolo_target_keeps_images_without_annotations() {
        let dataset = sample_dataset();
        let report = build_conversion_report(&dataset, Format::IrJson, Format::Yolo);

        assert!(report
            .issues
            .iter()
            .all(|issue| issue.code != ConversionIssueCode::DropImagesWithoutAnnotations));
        assert_eq!(report.output.images, report.input.images);
    }

    #[test]
    fn yolo_source_adds_policy_notes() {
        let dataset = Dataset::default();
        let report = build_conversion_report(&dataset, Format::Yolo, Format::Coco);

        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::YoloReaderIdAssignment));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::YoloReaderClassMapSource));
    }

    #[test]
    fn yolo_target_adds_policy_notes() {
        let dataset = Dataset::default();
        let report = build_conversion_report(&dataset, Format::Coco, Format::Yolo);

        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::YoloWriterClassOrder));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::YoloWriterEmptyLabelFiles));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::YoloWriterFloatPrecision));
    }

    #[test]
    fn to_voc_detects_lossiness() {
        let dataset = sample_dataset();
        let report = build_conversion_report(&dataset, Format::IrJson, Format::Voc);

        assert!(report.is_lossy());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::DropAnnotationAttributes));
    }

    #[test]
    fn voc_source_adds_policy_notes_and_depth_note() {
        let mut dataset = Dataset::default();
        let mut image = Image::new(1u64, "img1.jpg", 100, 100);
        image
            .attributes
            .insert("depth".to_string(), "1".to_string());
        dataset.images.push(image);

        let report = build_conversion_report(&dataset, Format::Voc, Format::Coco);

        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::VocReaderIdAssignment));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::VocReaderAttributeMapping));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::VocReaderCoordinatePolicy));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::VocReaderDepthHandling));
    }

    #[test]
    fn voc_target_adds_policy_notes() {
        let dataset = Dataset::default();
        let report = build_conversion_report(&dataset, Format::Coco, Format::Voc);

        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::VocWriterFileLayout));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::VocWriterNoImageCopy));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::VocWriterBoolNormalization));
    }

    #[test]
    fn to_label_studio_detects_lossiness() {
        let dataset = sample_dataset();
        let report = build_conversion_report(&dataset, Format::IrJson, Format::LabelStudio);

        assert!(report.is_lossy());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::DropDatasetInfo));
    }

    #[test]
    fn label_studio_source_adds_policy_notes_and_rotation_warning() {
        let mut dataset = Dataset::default();
        dataset.images.push(Image::new(1u64, "img.jpg", 100, 100));
        dataset.categories.push(Category::new(1u64, "cat"));

        let mut ann = Annotation::new(
            1u64,
            1u64,
            1u64,
            BBoxXYXY::<Pixel>::new(Coord::new(10.0, 10.0), Coord::new(20.0, 20.0)),
        );
        ann.attributes
            .insert("ls_rotation_deg".to_string(), "15".to_string());
        dataset.annotations.push(ann);

        let report = build_conversion_report(&dataset, Format::LabelStudio, Format::Coco);

        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::LabelStudioReaderIdAssignment));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::LabelStudioReaderImageRefPolicy));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::LabelStudioRotationDropped));
    }

    #[test]
    fn label_studio_target_adds_default_name_policy_note() {
        let mut dataset = Dataset::default();
        dataset.images.push(Image::new(1u64, "img.jpg", 100, 100));

        let report = build_conversion_report(&dataset, Format::Coco, Format::LabelStudio);

        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::LabelStudioWriterFromToDefaults));
    }

    #[test]
    fn to_cvat_detects_lossiness() {
        let dataset = sample_dataset();
        let report = build_conversion_report(&dataset, Format::IrJson, Format::Cvat);

        assert!(report.is_lossy());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::DropImageMetadata));
    }

    #[test]
    fn cvat_source_and_target_add_policy_notes() {
        let dataset = Dataset::default();
        let from_report = build_conversion_report(&dataset, Format::Cvat, Format::Coco);
        assert!(from_report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::CvatReaderIdAssignment));
        assert!(from_report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::CvatReaderAttributePolicy));

        let to_report = build_conversion_report(&dataset, Format::Coco, Format::Cvat);
        assert!(to_report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::CvatWriterMetaDefaults));
    }
}
