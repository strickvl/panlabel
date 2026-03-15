//! Conversion module for format transformation reporting.
//!
//! This module provides structured reporting for conversions between
//! annotation formats, tracking what information is preserved, lost,
//! or transformed according to deterministic policies.

pub mod report;

pub use report::{
    ConversionCounts, ConversionIssue, ConversionIssueCode, ConversionReport, ConversionSeverity,
    ConversionStage,
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
    LabelMe,
    CreateMl,
    Kitti,
    Via,
    Retinanet,
    OpenImages,
    KaggleWheat,
    AutoMlVision,
    Udacity,
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
            Format::LabelMe => "labelme",
            Format::CreateMl => "create-ml",
            Format::Kitti => "kitti",
            Format::Via => "via",
            Format::Retinanet => "retinanet",
            Format::OpenImages => "openimages",
            Format::KaggleWheat => "kaggle-wheat",
            Format::AutoMlVision => "automl-vision",
            Format::Udacity => "udacity",
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
            Format::LabelMe => IrLossiness::Lossy,
            Format::CreateMl => IrLossiness::Lossy,
            Format::Kitti => IrLossiness::Lossy,
            Format::Via => IrLossiness::Lossy,
            Format::Retinanet => IrLossiness::Lossy,
            Format::OpenImages => IrLossiness::Lossy,
            Format::KaggleWheat => IrLossiness::Lossy,
            Format::AutoMlVision => IrLossiness::Lossy,
            Format::Udacity => IrLossiness::Lossy,
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
        Format::LabelMe => analyze_to_labelme(dataset, &mut report),
        Format::CreateMl => analyze_to_createml(dataset, &mut report),
        Format::Kitti => analyze_to_kitti(dataset, &mut report),
        Format::Via => analyze_to_via(dataset, &mut report),
        Format::Retinanet => analyze_to_retinanet(dataset, &mut report),
        Format::OpenImages => analyze_to_openimages(dataset, &mut report),
        Format::KaggleWheat => analyze_to_kaggle_wheat(dataset, &mut report),
        Format::AutoMlVision => analyze_to_automl_vision(dataset, &mut report),
        Format::Udacity => analyze_to_udacity(dataset, &mut report),
    }

    // Add policy notes based on source format
    match from {
        Format::Tfod => add_tfod_reader_policy(&mut report),
        Format::Yolo => add_yolo_reader_policy(dataset, &mut report),
        Format::Voc => add_voc_reader_policy(dataset, &mut report),
        Format::LabelStudio => add_label_studio_reader_policy(dataset, &mut report),
        Format::Cvat => add_cvat_reader_policy(dataset, &mut report),
        Format::Coco => add_coco_reader_policy(&mut report),
        Format::HfImagefolder => add_hf_reader_policy(&mut report),
        Format::LabelMe => add_labelme_reader_policy(dataset, &mut report),
        Format::CreateMl => add_createml_reader_policy(&mut report),
        Format::Kitti => add_kitti_reader_policy(&mut report),
        Format::Via => add_via_reader_policy(&mut report),
        Format::Retinanet => add_retinanet_reader_policy(&mut report),
        Format::OpenImages => add_openimages_reader_policy(&mut report),
        Format::KaggleWheat => add_kaggle_wheat_reader_policy(&mut report),
        Format::AutoMlVision => add_automl_vision_reader_policy(&mut report),
        Format::Udacity => add_udacity_reader_policy(&mut report),
        Format::IrJson => {}
    }

    // Add policy notes based on target format
    match to {
        Format::Tfod => add_tfod_writer_policy(&mut report),
        Format::Yolo => add_yolo_writer_policy(&mut report),
        Format::Voc => add_voc_writer_policy(&mut report),
        Format::LabelStudio => add_label_studio_writer_policy(dataset, &mut report),
        Format::Cvat => add_cvat_writer_policy(&mut report),
        Format::Coco => add_coco_writer_policy(&mut report),
        Format::HfImagefolder => add_hf_writer_policy(&mut report),
        Format::LabelMe => add_labelme_writer_policy(&mut report),
        Format::CreateMl => add_createml_writer_policy(&mut report),
        Format::Kitti => add_kitti_writer_policy(&mut report),
        Format::Via => add_via_writer_policy(&mut report),
        Format::Retinanet => add_retinanet_writer_policy(&mut report),
        Format::OpenImages => add_openimages_writer_policy(&mut report),
        Format::KaggleWheat => add_kaggle_wheat_writer_policy(&mut report),
        Format::AutoMlVision => add_automl_vision_writer_policy(&mut report),
        Format::Udacity => add_udacity_writer_policy(&mut report),
        Format::IrJson => {}
    }

    report
}

/// Analyze conversion to TFOD format.
fn analyze_to_tfod(dataset: &Dataset, report: &mut ConversionReport) {
    add_common_csv_lossiness_warnings(dataset, report);
    add_annotation_drop_warnings_and_output_counts(dataset, report);
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

    // YOLO now preserves annotation confidence as an optional 6th token.
    // No DropAnnotationConfidence warning needed.

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

    // CVAT writer drops categories not referenced by any annotation.
    let used_category_ids: HashSet<_> = dataset.annotations.iter().map(|a| a.category_id).collect();
    let unused_count = dataset
        .categories
        .iter()
        .filter(|cat| !used_category_ids.contains(&cat.id))
        .count();

    report.output = report.input.clone();
    if unused_count > 0 {
        report.output.categories -= unused_count;
        report.add(ConversionIssue::warning(
            ConversionIssueCode::CvatWriterDropUnusedCategories,
            format!(
                "{} category(s) not referenced by any annotation will be dropped from CVAT <meta><labels>",
                unused_count
            ),
        ));
    }
}

/// Add policy notes for TFOD reader behavior.
fn add_tfod_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::TfodReaderIdAssignment,
        "TFOD reader assigns IDs deterministically: images by filename (lexicographic), \
         categories by class name (lexicographic), annotations by CSV row order"
            .to_string(),
    ));
}

/// Add policy notes for TFOD writer behavior.
fn add_tfod_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::TfodWriterRowOrder,
        "TFOD writer orders rows by annotation ID for deterministic output".to_string(),
    ));
}

/// Add policy notes for YOLO reader behavior.
fn add_yolo_reader_policy(dataset: &Dataset, report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::YoloReaderIdAssignment,
        "YOLO reader assigns IDs deterministically: images by relative path (lexicographic), categories by class index, annotations by label-file order then line number".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::YoloReaderClassMapSource,
        "YOLO reader class map source precedence: data.yaml, then classes.txt, then inferred from label files".to_string(),
    ));

    // Emit split handling note if YOLO split provenance is present
    if let Some(mode) = dataset.info.attributes.get("yolo_layout_mode") {
        if mode == "split_aware" {
            let found = dataset
                .info
                .attributes
                .get("yolo_splits_found")
                .map(|s| s.as_str())
                .unwrap_or("?");
            let read = dataset
                .info
                .attributes
                .get("yolo_splits_read")
                .map(|s| s.as_str())
                .unwrap_or("?");
            let message = if found == read {
                format!(
                    "YOLO reader discovered splits [{}] and merged them into one dataset",
                    found
                )
            } else {
                format!(
                    "YOLO reader discovered splits [{}]; selected split(s): [{}]",
                    found, read
                )
            };
            report.add(ConversionIssue::reader_info(
                ConversionIssueCode::YoloReaderSplitHandling,
                message,
            ));
        }
    }
}

/// Add policy notes for YOLO writer behavior.
fn add_yolo_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::YoloWriterClassOrder,
        "YOLO writer assigns class indices by CategoryId order (sorted ascending)".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::YoloWriterEmptyLabelFiles,
        "YOLO writer creates empty .txt files for images without annotations".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::YoloWriterFloatPrecision,
        "YOLO writer outputs normalized coordinates (and confidence when present) with 6 decimal places".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::YoloWriterDeterministicOrder,
        "YOLO writer orders images and label files by file_name (lexicographic) for deterministic output".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::YoloWriterNoImageCopy,
        "YOLO writer creates only label files and data.yaml; image binaries are not copied to the output directory".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::YoloWriterDataYamlPolicy,
        "YOLO writer emits data.yaml with a names: mapping (sorted by class index); does not emit train/val paths or nc".to_string(),
    ));
}

/// Add policy notes for VOC reader behavior.
fn add_voc_reader_policy(dataset: &Dataset, report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::VocReaderIdAssignment,
        "VOC reader assigns IDs deterministically: images by <filename> (lexicographic), categories by class name (lexicographic), annotations by XML file order then <object> order".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::VocReaderAttributeMapping,
        "VOC reader maps pose/truncated/difficult/occluded into annotation attributes".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
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
        report.add(ConversionIssue::reader_info(
            ConversionIssueCode::VocReaderDepthHandling,
            "VOC reader preserves <depth> as image attribute 'depth'; non-3 depth values may indicate non-RGB imagery"
                .to_string(),
        ));
    }
}

/// Add policy notes for VOC writer behavior.
fn add_voc_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::VocWriterFileLayout,
        "VOC writer emits one XML per image under Annotations/, preserving image subdirectory structure"
            .to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::VocWriterNoImageCopy,
        "VOC writer creates JPEGImages/README.txt but does not copy image binaries".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::VocWriterBoolNormalization,
        "VOC writer normalizes truncated/difficult/occluded attributes: true/yes/1 -> 1 and false/no/0 -> 0"
            .to_string(),
    ));
}

/// Add policy notes for Label Studio reader behavior.
fn add_label_studio_reader_policy(dataset: &Dataset, report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::LabelStudioReaderIdAssignment,
        "Label Studio reader assigns IDs deterministically: images by derived file_name (lexicographic), categories by label (lexicographic), annotations by image order then result order".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
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
        report.add(ConversionIssue::writer_info(
            ConversionIssueCode::LabelStudioWriterFromToDefaults,
            "Label Studio writer uses from_name='label' and to_name='image' when ls_from_name/ls_to_name attributes are absent".to_string(),
        ));
    }

    let has_confidence = dataset
        .annotations
        .iter()
        .any(|ann| ann.confidence.is_some());
    if has_confidence {
        report.add(ConversionIssue::writer_info(
            ConversionIssueCode::LabelStudioWriterConfidenceRouting,
            "Label Studio writer routes annotations with confidence scores to the 'predictions' block instead of 'annotations'".to_string(),
        ));
    }
}

/// Add policy notes for CVAT reader behavior.
fn add_cvat_reader_policy(_dataset: &Dataset, report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::CvatReaderIdAssignment,
        "CVAT reader assigns IDs deterministically: images by <image name> (lexicographic), categories by label name (lexicographic), annotations by image order then <box> order".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::CvatReaderAttributePolicy,
        "CVAT reader maps xtl/ytl/xbr/ybr to IR pixel XYXY 1:1; custom <attribute> children are stored as annotation attributes with 'cvat_attr_' prefix".to_string(),
    ));
}

/// Add policy notes for CVAT writer behavior.
fn add_cvat_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CvatWriterMetaDefaults,
        "CVAT writer emits a minimal <meta><task> block with name='panlabel export', mode='annotation', and size equal to image count".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CvatWriterDeterministicOrder,
        "CVAT writer orders images by file_name (lexicographic) and boxes within each image by annotation ID".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CvatWriterImageIdReassignment,
        "CVAT writer assigns sequential image IDs (0, 1, 2, ...) by sorted order; original cvat_image_id attributes are not preserved in output".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CvatWriterSourceDefault,
        "CVAT writer defaults missing or empty source attribute to 'manual'".to_string(),
    ));
}

/// Add policy notes for HF reader behavior.
fn add_hf_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::HfReaderCategoryResolution,
        "HF reader resolves category names with precedence: ClassLabel/preflight map, then --hf-category-map, then integer fallback"
            .to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::HfReaderObjectContainerPrecedence,
        "HF reader selects the object container with precedence: --hf-objects-column override, then 'objects' column, then 'faces' column"
            .to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::HfReaderBboxFormatDependence,
        "HF reader interprets bounding boxes according to --hf-bbox-format (default: xywh); incorrect setting will produce wrong coordinates"
            .to_string(),
    ));
}

/// Add policy notes for HF writer behavior.
fn add_hf_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::HfWriterDeterministicOrder,
        "HF writer orders metadata rows by image file_name and annotation lists by annotation ID"
            .to_string(),
    ));
}

/// Add policy notes for COCO reader behavior.
fn add_coco_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::CocoReaderAttributeMapping,
        "COCO reader maps score to IR confidence and stores area/iscrowd as annotation attributes"
            .to_string(),
    ));
}

/// Add policy notes for COCO writer behavior.
fn add_coco_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CocoWriterDeterministicOrder,
        "COCO writer sorts licenses, images, categories, and annotations by ID for deterministic output"
            .to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CocoWriterScoreMapping,
        "COCO writer maps IR confidence to the COCO score field".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CocoWriterAreaIscrowdMapping,
        "COCO writer reads area/iscrowd from annotation attributes; defaults to bbox-computed area and iscrowd=0 when absent"
            .to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CocoWriterEmptySegmentation,
        "COCO writer emits an empty segmentation array for detection-only output".to_string(),
    ));
}

// ============================================================================
// LabelMe analysis and policy
// ============================================================================

/// LabelMe-specific attribute key for shape type provenance.
const LABELME_ATTR_SHAPE_TYPE: &str = "labelme_shape_type";

fn analyze_to_labelme(dataset: &Dataset, report: &mut ConversionReport) {
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

    let has_license_or_date = dataset
        .images
        .iter()
        .any(|img| img.license_id.is_some() || img.date_captured.is_some());
    if has_license_or_date {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropImageMetadata,
            "image license_id and/or date_captured will be dropped".to_string(),
        ));
    }

    let has_supercategory = dataset.categories.iter().any(|c| c.supercategory.is_some());
    if has_supercategory {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropCategorySupercategory,
            "category supercategory will be dropped".to_string(),
        ));
    }

    let has_confidence = dataset.annotations.iter().any(|a| a.confidence.is_some());
    if has_confidence {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationConfidence,
            "annotation confidence values will be dropped".to_string(),
        ));
    }

    let has_non_labelme_attrs = dataset
        .annotations
        .iter()
        .any(|a| a.attributes.keys().any(|k| k != LABELME_ATTR_SHAPE_TYPE));
    if has_non_labelme_attrs {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationAttributes,
            "annotation attributes (other than labelme_shape_type) will be dropped".to_string(),
        ));
    }

    // Output counts: all images and annotations survive; categories = only referenced ones
    let referenced_cats: HashSet<_> = dataset.annotations.iter().map(|a| a.category_id).collect();
    let output_categories = referenced_cats.len();

    report.output = ConversionCounts {
        images: dataset.images.len(),
        categories: output_categories,
        annotations: dataset.annotations.len(),
    };
}

fn add_labelme_reader_policy(dataset: &Dataset, report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::LabelmeReaderIdAssignment,
        "LabelMe reader assigns image IDs by sorted file_name, category IDs by sorted label, \
         and annotation IDs sequentially by image then shape order"
            .to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::LabelmeReaderPathPolicy,
        "LabelMe reader derives IR file_name from annotation file path and imagePath extension; \
         raw imagePath is stored in image attributes as labelme_image_path"
            .to_string(),
    ));

    let has_polygons = dataset.annotations.iter().any(|a| {
        a.attributes
            .get(LABELME_ATTR_SHAPE_TYPE)
            .map(|v| v.as_str())
            == Some("polygon")
    });
    if has_polygons {
        report.add(ConversionIssue::reader_info(
            ConversionIssueCode::LabelmePolygonEnvelopeApplied,
            "LabelMe reader converted polygon shapes to axis-aligned bounding box envelopes; \
             original shape type stored as labelme_shape_type=polygon attribute"
                .to_string(),
        ));
    }
}

fn add_labelme_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::LabelmeWriterFileLayout,
        "LabelMe writer emits annotations/<stem>.json files in a canonical directory layout"
            .to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::LabelmeWriterRectanglePolicy,
        "LabelMe writer emits all annotations as rectangle shapes with 2 corner points".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::LabelmeWriterNoImageCopy,
        "LabelMe writer creates only annotation files; images are not copied".to_string(),
    ));
}

// ============================================================================
// CreateML analysis and policy
// ============================================================================

fn analyze_to_createml(dataset: &Dataset, report: &mut ConversionReport) {
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

    let has_license_or_date = dataset
        .images
        .iter()
        .any(|img| img.license_id.is_some() || img.date_captured.is_some());
    if has_license_or_date {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropImageMetadata,
            "image license_id and/or date_captured will be dropped".to_string(),
        ));
    }

    let has_supercategory = dataset.categories.iter().any(|c| c.supercategory.is_some());
    if has_supercategory {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropCategorySupercategory,
            "category supercategory will be dropped".to_string(),
        ));
    }

    let has_confidence = dataset.annotations.iter().any(|a| a.confidence.is_some());
    if has_confidence {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationConfidence,
            "annotation confidence values will be dropped".to_string(),
        ));
    }

    let has_attributes = dataset.annotations.iter().any(|a| !a.attributes.is_empty());
    if has_attributes {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationAttributes,
            "annotation attributes will be dropped".to_string(),
        ));
    }

    let referenced_cats: HashSet<_> = dataset.annotations.iter().map(|a| a.category_id).collect();
    let output_categories = referenced_cats.len();

    report.output = ConversionCounts {
        images: dataset.images.len(),
        categories: output_categories,
        annotations: dataset.annotations.len(),
    };
}

fn add_createml_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::CreatemlReaderIdAssignment,
        "CreateML reader assigns image IDs by sorted filename, category IDs by sorted label, \
         and annotation IDs sequentially by image then annotation order"
            .to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::CreatemlReaderImageResolution,
        "CreateML reader resolves image dimensions from local files: \
         tries <json_dir>/<image>, then <json_dir>/images/<image>"
            .to_string(),
    ));
}

fn add_createml_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CreatemlWriterDeterministicOrder,
        "CreateML writer orders image rows by filename and annotations by ID".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CreatemlWriterCoordinateMapping,
        "CreateML writer converts IR pixel XYXY to center-based absolute coordinates (x, y, width, height)"
            .to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::CreatemlWriterNoImageCopy,
        "CreateML writer creates only the JSON file; images are not copied".to_string(),
    ));
}

/// KITTI attribute keys preserved by the writer.
const KITTI_PRESERVED_ATTRS: &[&str] = &[
    "kitti_truncated",
    "kitti_occluded",
    "kitti_alpha",
    "kitti_dim_height",
    "kitti_dim_width",
    "kitti_dim_length",
    "kitti_loc_x",
    "kitti_loc_y",
    "kitti_loc_z",
    "kitti_rotation_y",
];

fn analyze_to_kitti(dataset: &Dataset, report: &mut ConversionReport) {
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
    // KITTI preserves confidence as optional score field — no warning needed.
    let anns_with_unrepresentable_attrs = dataset
        .annotations
        .iter()
        .filter(|ann| {
            ann.attributes
                .keys()
                .any(|key| !KITTI_PRESERVED_ATTRS.contains(&key.as_str()))
        })
        .count();
    if anns_with_unrepresentable_attrs > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationAttributes,
            format!(
                "{} annotation(s) have attributes outside KITTI's preserved set (kitti_*)",
                anns_with_unrepresentable_attrs
            ),
        ));
    }
    report.output = report.input.clone();
}

fn analyze_to_via(dataset: &Dataset, report: &mut ConversionReport) {
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
                    .keys()
                    .any(|key| key != "via_size_bytes" && !key.starts_with("via_file_attr_"))
        })
        .count();
    if images_with_metadata > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropImageMetadata,
            format!(
                "{} image(s) have metadata that VIA cannot represent",
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
            ann.attributes
                .keys()
                .any(|key| !key.starts_with("via_region_attr_"))
        })
        .count();
    if anns_with_unrepresentable_attrs > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationAttributes,
            format!(
                "{} annotation(s) have attributes outside VIA's preserved set (via_region_attr_*)",
                anns_with_unrepresentable_attrs
            ),
        ));
    }
    report.output = report.input.clone();
}

fn analyze_to_retinanet(dataset: &Dataset, report: &mut ConversionReport) {
    add_common_csv_lossiness_warnings(dataset, report);
    add_annotation_drop_warnings(dataset, report);
    // RetinaNet supports unannotated images (empty rows), so output = input
    report.output = report.input.clone();
}

fn add_kitti_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::KittiReaderIdAssignment,
        "KITTI reader assigns image IDs by filename order, category IDs by class name order, annotation IDs by file/line order".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::KittiReaderFieldMapping,
        "KITTI non-bbox fields (truncated, occluded, alpha, dimensions, location, rotation) are stored as kitti_* annotation attributes".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::KittiReaderImageResolution,
        "KITTI reader resolves image dimensions from image_2/ with extension precedence: .png, .jpg, .jpeg, .bmp, .webp".to_string(),
    ));
}

fn add_kitti_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::KittiWriterFileLayout,
        "KITTI writer creates label_2/ with one .txt per image and image_2/README.txt".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::KittiWriterDefaultFieldValues,
        "KITTI writer uses default values for missing kitti_* attributes (truncated=0, occluded=0, alpha=-10, dims=-1, loc=-1000, rotation=-10)".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::KittiWriterDeterministicOrder,
        "KITTI writer sorts images by filename and annotations within each image by ID".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::KittiWriterNoImageCopy,
        "KITTI writer creates only label files; images are not copied".to_string(),
    ));
}

fn add_via_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::ViaReaderIdAssignment,
        "VIA reader assigns image IDs by filename order, category IDs by label order, annotation IDs by image/region order".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::ViaReaderLabelResolution,
        "VIA reader resolves category labels from region_attributes with precedence: 'label', 'class', then sole scalar attribute".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::ViaReaderImageResolution,
        "VIA reader resolves image dimensions from disk: <json_dir>/<filename> then <json_dir>/images/<filename>".to_string(),
    ));
}

fn add_via_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::ViaWriterDeterministicOrder,
        "VIA writer orders entries by filename and regions by annotation ID".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::ViaWriterLabelAttributeKey,
        "VIA writer uses canonical 'label' key in region_attributes for category names".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::ViaWriterNoImageCopy,
        "VIA writer creates only the JSON file; images are not copied".to_string(),
    ));
}

fn add_retinanet_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::RetinanetReaderIdAssignment,
        "RetinaNet reader assigns image IDs by path order, category IDs by class name order, annotation IDs by row order".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::RetinanetReaderImageResolution,
        "RetinaNet reader resolves image dimensions from disk relative to CSV parent directory"
            .to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::RetinanetReaderEmptyRowHandling,
        "RetinaNet reader treats rows with empty bbox/class fields as unannotated image entries"
            .to_string(),
    ));
}

fn add_retinanet_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::RetinanetWriterDeterministicOrder,
        "RetinaNet writer groups rows by image (sorted by filename) with annotations sorted by ID"
            .to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::RetinanetWriterEmptyRows,
        "RetinaNet writer emits path,,,,, rows for images without annotations".to_string(),
    ));
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::RetinanetWriterNoImageCopy,
        "RetinaNet writer creates only the CSV file; images are not copied".to_string(),
    ));
}

// ============================================================================
// OpenImages CSV
// ============================================================================

fn analyze_to_openimages(dataset: &Dataset, report: &mut ConversionReport) {
    add_common_csv_lossiness_warnings(dataset, report);
    // OpenImages preserves confidence, so no DropAnnotationConfidence
    // But non-openimages_* attributes are dropped
    let anns_with_non_openimages_attrs = dataset
        .annotations
        .iter()
        .filter(|ann| ann.attributes.keys().any(|k| !k.starts_with("openimages_")))
        .count();
    if anns_with_non_openimages_attrs > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropAnnotationAttributes,
            format!(
                "{} annotation(s) have non-OpenImages attributes that will be dropped",
                anns_with_non_openimages_attrs
            ),
        ));
    }
    add_images_without_annotations_warning_and_output_counts(dataset, report);
}

fn add_openimages_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::OpenimagesReaderIdAssignment,
        "OpenImages reader assigns image IDs by ImageID order, category IDs by LabelName order, annotation IDs by row order".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::OpenimagesReaderImageResolution,
        "OpenImages reader resolves image dimensions from local files (base_dir and base_dir/images with extension probing)".to_string(),
    ));
}

fn add_openimages_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::OpenimagesWriterDeterministicOrder,
        "OpenImages writer orders rows by annotation ID".to_string(),
    ));
}

// ============================================================================
// Kaggle Wheat CSV
// ============================================================================

fn analyze_to_kaggle_wheat(dataset: &Dataset, report: &mut ConversionReport) {
    add_common_csv_lossiness_warnings(dataset, report);
    add_annotation_drop_warnings(dataset, report);
    // Single-class collapse warning
    if dataset.categories.len() > 1 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::CollapseMultipleCategoriesToSingleClass,
            format!(
                "{} categories will be collapsed to single class 'wheat_head'",
                dataset.categories.len()
            ),
        ));
    }
    let distinct_image_ids = add_images_without_annotations_warning(dataset, report);
    report.output = ConversionCounts {
        images: distinct_image_ids.len(),
        categories: if dataset.annotations.is_empty() { 0 } else { 1 },
        annotations: dataset.annotations.len(),
    };
}

fn add_kaggle_wheat_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::KaggleWheatReaderIdAssignment,
        "Kaggle Wheat reader assigns image IDs by image_id order, single category 'wheat_head', annotation IDs by row order; source stored as kaggle_wheat_source image attribute".to_string(),
    ));
}

fn add_kaggle_wheat_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::KaggleWheatWriterDeterministicOrder,
        "Kaggle Wheat writer orders rows by annotation ID and emits bbox as [x, y, width, height]"
            .to_string(),
    ));
}

// ============================================================================
// AutoML Vision CSV
// ============================================================================

fn analyze_to_automl_vision(dataset: &Dataset, report: &mut ConversionReport) {
    add_common_csv_lossiness_warnings(dataset, report);
    add_annotation_drop_warnings_and_output_counts(dataset, report);
}

fn add_automl_vision_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::AutomlVisionReaderIdAssignment,
        "AutoML Vision reader assigns image IDs by URI order, category IDs by label order, annotation IDs by row order".to_string(),
    ));
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::AutomlVisionReaderImageResolution,
        "AutoML Vision reader resolves image dimensions from local files; GCS URIs resolved by path suffix then basename".to_string(),
    ));
}

fn add_automl_vision_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::AutomlVisionWriterDeterministicOrder,
        "AutoML Vision writer emits headerless 11-column sparse rows ordered by annotation ID"
            .to_string(),
    ));
}

// ============================================================================
// Udacity CSV
// ============================================================================

fn analyze_to_udacity(dataset: &Dataset, report: &mut ConversionReport) {
    add_common_csv_lossiness_warnings(dataset, report);
    add_annotation_drop_warnings_and_output_counts(dataset, report);
}

fn add_udacity_reader_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::reader_info(
        ConversionIssueCode::UdacityReaderIdAssignment,
        "Udacity reader assigns image IDs by filename order, category IDs by class name order, annotation IDs by row order".to_string(),
    ));
}

fn add_udacity_writer_policy(report: &mut ConversionReport) {
    report.add(ConversionIssue::writer_info(
        ConversionIssueCode::UdacityWriterRowOrder,
        "Udacity writer orders rows by annotation ID".to_string(),
    ));
}

// ============================================================================
// Common CSV lossiness helpers
// ============================================================================

/// Adds common lossiness warnings shared by all simple row-based CSV formats:
/// DropDatasetInfo, DropLicenses, DropImageMetadata, DropCategorySupercategory.
fn add_common_csv_lossiness_warnings(dataset: &Dataset, report: &mut ConversionReport) {
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
}

/// Adds DropAnnotationConfidence and DropAnnotationAttributes warnings.
/// Used by formats that lose both confidence and all attributes.
fn add_annotation_drop_warnings(dataset: &Dataset, report: &mut ConversionReport) {
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
}

/// Adds DropImagesWithoutAnnotations warning and returns the set of annotated image IDs.
/// Reuse the returned set for output counts to avoid recomputing it.
fn add_images_without_annotations_warning(
    dataset: &Dataset,
    report: &mut ConversionReport,
) -> HashSet<crate::ir::ImageId> {
    let distinct_image_ids: HashSet<_> = dataset.annotations.iter().map(|a| a.image_id).collect();
    let images_without = dataset
        .images
        .iter()
        .filter(|img| !distinct_image_ids.contains(&img.id))
        .count();
    if images_without > 0 {
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropImagesWithoutAnnotations,
            format!(
                "{} image(s) have no annotations and will not appear in output",
                images_without
            ),
        ));
    }
    distinct_image_ids
}

/// Adds DropImagesWithoutAnnotations warning and sets output counts
/// (distinct annotated images/categories). Used by most annotation-only CSV formats.
fn add_images_without_annotations_warning_and_output_counts(
    dataset: &Dataset,
    report: &mut ConversionReport,
) {
    let distinct_image_ids = add_images_without_annotations_warning(dataset, report);
    let distinct_category_ids: HashSet<_> =
        dataset.annotations.iter().map(|a| a.category_id).collect();
    report.output = ConversionCounts {
        images: distinct_image_ids.len(),
        categories: distinct_category_ids.len(),
        annotations: dataset.annotations.len(),
    };
}

/// Full annotation-drop lossiness: confidence + attributes + images-without-annotations + output counts.
/// Used by TFOD, Udacity, AutoML Vision, and similar annotation-only CSV formats.
fn add_annotation_drop_warnings_and_output_counts(
    dataset: &Dataset,
    report: &mut ConversionReport,
) {
    add_annotation_drop_warnings(dataset, report);
    add_images_without_annotations_warning_and_output_counts(dataset, report);
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
        assert!(to_report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::CvatWriterDeterministicOrder));
        assert!(to_report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::CvatWriterImageIdReassignment));
        assert!(to_report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::CvatWriterSourceDefault));
    }

    #[test]
    fn cvat_output_counts_reflect_unused_category_drop() {
        let mut dataset = sample_dataset();
        // Add an unused category
        dataset
            .categories
            .push(Category::new(99u64, "unused_label"));

        let report = build_conversion_report(&dataset, Format::IrJson, Format::Cvat);

        // Input has all categories, output should drop the unused one
        assert_eq!(report.input.categories, 2);
        assert_eq!(report.output.categories, 1);
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::CvatWriterDropUnusedCategories));
    }

    #[test]
    fn cvat_no_unused_category_warning_when_all_used() {
        // sample_dataset has 1 category and 1 annotation using it
        let dataset = sample_dataset();
        let report = build_conversion_report(&dataset, Format::IrJson, Format::Cvat);

        assert_eq!(report.input.categories, report.output.categories);
        assert!(!report
            .issues
            .iter()
            .any(|i| i.code == ConversionIssueCode::CvatWriterDropUnusedCategories));
    }
}
