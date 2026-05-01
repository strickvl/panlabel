//! Conversion report types for tracking lossiness and policy decisions.
//!
//! This module provides structured reporting for format conversions,
//! similar to how `validation::ValidationReport` tracks dataset issues.

use serde::Serialize;
use std::fmt;

/// A report generated during format conversion.
///
/// Tracks input/output counts, lossiness warnings, and policy decisions
/// to help users understand exactly what happened during conversion.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ConversionReport {
    /// Source format name.
    pub from: String,
    /// Target format name.
    pub to: String,
    /// Counts from the input dataset.
    pub input: ConversionCounts,
    /// Counts in the output (may differ if images are dropped, etc.).
    pub output: ConversionCounts,
    /// Issues discovered during conversion analysis.
    pub issues: Vec<ConversionIssue>,
}

impl ConversionReport {
    /// Create a new empty report for a conversion between formats.
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            ..Default::default()
        }
    }

    /// Add an issue to the report.
    pub fn add(&mut self, issue: ConversionIssue) {
        self.issues.push(issue);
    }

    /// Count of warning-level issues (true lossiness).
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == ConversionSeverity::Warning)
            .count()
    }

    /// Count of info-level issues (policy decisions, notes).
    pub fn info_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == ConversionSeverity::Info)
            .count()
    }

    /// Returns true if this conversion would lose information.
    ///
    /// A conversion is lossy if it has any warning-level issues.
    pub fn is_lossy(&self) -> bool {
        self.warning_count() > 0
    }
}

impl fmt::Display for ConversionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Always show counts
        writeln!(
            f,
            "  {} images, {} categories, {} annotations",
            self.input.images, self.input.categories, self.input.annotations
        )?;

        // Show output counts if they differ from input
        if self.output != self.input {
            writeln!(
                f,
                "  output: {} images, {} categories, {} annotations",
                self.output.images, self.output.categories, self.output.annotations
            )?;
        }

        // Show issues if any
        if !self.issues.is_empty() {
            let warnings = self.warning_count();
            let infos = self.info_count();

            if warnings > 0 {
                writeln!(f)?;
                writeln!(f, "Warnings ({}):", warnings)?;
                for issue in self
                    .issues
                    .iter()
                    .filter(|i| i.severity == ConversionSeverity::Warning)
                {
                    writeln!(f, "  - [{}] {}", issue.code.as_str(), issue.message)?;
                }
            }

            if infos > 0 {
                writeln!(f)?;
                writeln!(f, "Notes ({}):", infos)?;
                for issue in self
                    .issues
                    .iter()
                    .filter(|i| i.severity == ConversionSeverity::Info)
                {
                    writeln!(f, "  - [{}] {}", issue.code.as_str(), issue.message)?;
                }
            }
        }

        Ok(())
    }
}

/// Counts of dataset elements.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ConversionCounts {
    pub images: usize,
    pub categories: usize,
    pub annotations: usize,
}

/// A single issue discovered during conversion analysis.
#[derive(Clone, Debug, Serialize)]
pub struct ConversionIssue {
    pub severity: ConversionSeverity,
    pub stage: ConversionStage,
    pub code: ConversionIssueCode,
    pub message: String,
}

impl ConversionIssue {
    /// Create a warning-level issue from lossiness analysis.
    pub fn warning(code: ConversionIssueCode, message: impl Into<String>) -> Self {
        Self {
            severity: ConversionSeverity::Warning,
            stage: ConversionStage::Analysis,
            code,
            message: message.into(),
        }
    }

    /// Create an info-level issue for a source reader policy.
    pub fn reader_info(code: ConversionIssueCode, message: impl Into<String>) -> Self {
        Self {
            severity: ConversionSeverity::Info,
            stage: ConversionStage::SourceReader,
            code,
            message: message.into(),
        }
    }

    /// Create an info-level issue for a target writer policy.
    pub fn writer_info(code: ConversionIssueCode, message: impl Into<String>) -> Self {
        Self {
            severity: ConversionSeverity::Info,
            stage: ConversionStage::TargetWriter,
            code,
            message: message.into(),
        }
    }

    /// Create an info-level issue (generic, for backward compatibility).
    pub fn info(code: ConversionIssueCode, message: impl Into<String>) -> Self {
        Self {
            severity: ConversionSeverity::Info,
            stage: ConversionStage::Analysis,
            code,
            message: message.into(),
        }
    }
}

/// The pipeline stage where a conversion issue originates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversionStage {
    /// Lossiness analysis (warnings about data loss).
    Analysis,
    /// Source format reader policy.
    SourceReader,
    /// Target format writer policy.
    TargetWriter,
}

/// Severity level for conversion issues.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversionSeverity {
    /// A warning indicates information loss; requires `--allow-lossy`.
    Warning,
    /// An info note describes policy decisions; does not block conversion.
    Info,
}

/// Stable issue codes for programmatic consumption.
///
/// These codes are part of the JSON schema and should remain stable.
/// Use [`ConversionIssueCode::as_str()`] for the canonical string form
/// shared by both text and JSON output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversionIssueCode {
    // IR -> TFOD lossiness
    /// Dataset info/metadata will be dropped.
    DropDatasetInfo,
    /// Licenses will be dropped.
    DropLicenses,
    /// Image license_id and/or date_captured will be dropped.
    DropImageMetadata,
    /// Category supercategory will be dropped.
    DropCategorySupercategory,
    /// Annotation confidence scores will be dropped.
    DropAnnotationConfidence,
    /// Annotation attributes will be dropped.
    DropAnnotationAttributes,
    /// Images without annotations will not appear in output.
    DropImagesWithoutAnnotations,

    // IR -> COCO lossiness
    /// Dataset info.name has no COCO equivalent.
    DropDatasetInfoName,
    /// Annotation attributes (other than area/iscrowd) may not be preserved by COCO tools.
    CocoAttributesMayNotBePreserved,

    // IR -> COCO policy (Info level — writer behaviors)
    /// COCO writer sorts all lists (licenses, images, categories, annotations) by ID.
    CocoWriterDeterministicOrder,
    /// COCO writer maps IR confidence to COCO score field.
    CocoWriterScoreMapping,
    /// COCO writer maps area/iscrowd via IR annotation attributes.
    CocoWriterAreaIscrowdMapping,
    /// COCO writer emits empty segmentation arrays for detection-only output.
    CocoWriterEmptySegmentation,
    /// COCO reader maps score to IR confidence and area/iscrowd to annotation attributes.
    CocoReaderAttributeMapping,

    // IR -> HF lossiness
    /// HF metadata cannot represent all IR dataset metadata/licensing fields.
    HfMetadataLost,
    /// HF metadata loses annotation/image attributes not in the flat schema.
    HfAttributesLost,
    /// HF metadata does not represent annotation confidence scores.
    HfConfidenceLost,

    // HF policy (Info level)
    /// HF reader object-container column precedence.
    HfReaderObjectContainerPrecedence,
    /// HF reader bbox interpretation depends on --hf-bbox-format.
    HfReaderBboxFormatDependence,

    // Label Studio policy (Info level — writer behaviors)
    /// Label Studio writer routes confident annotations to predictions block.
    LabelStudioWriterConfidenceRouting,

    // YOLO policy (Info level — writer behaviors)
    /// YOLO writer orders images and labels deterministically.
    YoloWriterDeterministicOrder,
    /// YOLO writer does not copy image binaries; only label files are written.
    YoloWriterNoImageCopy,
    /// YOLO writer emits data.yaml with class names and directory paths.
    YoloWriterDataYamlPolicy,

    // Policy decisions (Info level — reader/writer behaviors)
    /// TFOD reader assigns IDs by lexicographic ordering.
    TfodReaderIdAssignment,
    /// TFOD writer orders rows by annotation ID.
    TfodWriterRowOrder,
    /// YOLO reader assigns IDs by deterministic ordering.
    YoloReaderIdAssignment,
    /// YOLO reader class-map precedence/source.
    YoloReaderClassMapSource,
    /// YOLO reader split-aware layout handling (merge or selection).
    YoloReaderSplitHandling,
    /// YOLO writer assigns class indices by category ID order.
    YoloWriterClassOrder,
    /// YOLO writer creates empty label files for images without annotations.
    YoloWriterEmptyLabelFiles,
    /// YOLO writer outputs normalized floats at 6 decimal places.
    YoloWriterFloatPrecision,
    /// VOC reader assigns IDs by deterministic ordering.
    VocReaderIdAssignment,
    /// VOC reader maps pose/truncated/difficult/occluded to IR attributes.
    VocReaderAttributeMapping,
    /// VOC reader keeps bndbox coordinates exactly as provided (no offset adjustment).
    VocReaderCoordinatePolicy,
    /// VOC reader stores depth as an image attribute; depth != 3 may need downstream care.
    VocReaderDepthHandling,
    /// VOC writer file layout and XML naming policy.
    VocWriterFileLayout,
    /// VOC writer creates JPEGImages/README.txt and does not copy images.
    VocWriterNoImageCopy,
    /// VOC writer normalizes boolean fields (truncated/difficult/occluded).
    VocWriterBoolNormalization,
    /// Label Studio reader converted rotated boxes to axis-aligned envelopes.
    LabelStudioRotationDropped,
    /// Label Studio reader ID assignment policy.
    LabelStudioReaderIdAssignment,
    /// Label Studio reader image-reference policy.
    LabelStudioReaderImageRefPolicy,
    /// Label Studio writer default from_name/to_name policy.
    LabelStudioWriterFromToDefaults,
    /// CVAT reader deterministic ID assignment policy.
    CvatReaderIdAssignment,
    /// CVAT reader coordinate + attribute mapping policy.
    CvatReaderAttributePolicy,
    /// CVAT writer default metadata block policy.
    CvatWriterMetaDefaults,
    /// CVAT writer deterministic ordering policy (images by filename, boxes by annotation ID).
    CvatWriterDeterministicOrder,
    /// CVAT writer reassigns image IDs sequentially (original cvat_image_id not preserved).
    CvatWriterImageIdReassignment,
    /// CVAT writer defaults missing source attribute to "manual".
    CvatWriterSourceDefault,
    /// CVAT writer drops unused categories from <meta><labels>.
    CvatWriterDropUnusedCategories,
    /// HF reader category-name resolution precedence.
    HfReaderCategoryResolution,
    /// HF writer deterministic output ordering policy.
    HfWriterDeterministicOrder,

    // SageMaker Ground Truth policy (Info level)
    /// SageMaker reader deterministic ID assignment policy.
    SagemakerReaderIdAssignment,
    /// SageMaker reader label attribute detection policy.
    SagemakerReaderLabelAttributeDetection,
    /// SageMaker reader class-map/category resolution policy.
    SagemakerReaderClassMapResolution,
    /// SageMaker writer deterministic ordering policy.
    SagemakerWriterDeterministicOrder,
    /// SageMaker writer class-map/class_id assignment policy.
    SagemakerWriterClassMapPolicy,
    /// SageMaker writer metadata default policy.
    SagemakerWriterMetadataDefaults,
    /// SageMaker writer does not copy images.
    SagemakerWriterNoImageCopy,

    // LabelMe policy (Info level)
    /// LabelMe reader deterministic ID assignment policy.
    LabelmeReaderIdAssignment,
    /// LabelMe reader file-name/path derivation policy.
    LabelmeReaderPathPolicy,
    /// LabelMe reader polygon envelope behavior.
    LabelmePolygonEnvelopeApplied,
    /// LabelMe writer canonical file layout.
    LabelmeWriterFileLayout,
    /// LabelMe writer emits all annotations as rectangles.
    LabelmeWriterRectanglePolicy,
    /// LabelMe writer does not copy images.
    LabelmeWriterNoImageCopy,

    // CreateML policy (Info level)
    /// CreateML reader deterministic ID assignment policy.
    CreatemlReaderIdAssignment,
    /// CreateML reader image resolution precedence.
    CreatemlReaderImageResolution,
    /// CreateML writer deterministic ordering policy.
    CreatemlWriterDeterministicOrder,
    /// CreateML writer center-based coordinate mapping.
    CreatemlWriterCoordinateMapping,
    /// CreateML writer does not copy images.
    CreatemlWriterNoImageCopy,

    // KITTI policy (Info level)
    /// KITTI reader deterministic ID assignment policy.
    KittiReaderIdAssignment,
    /// KITTI reader maps non-bbox fields to IR attributes.
    KittiReaderFieldMapping,
    /// KITTI reader image resolution precedence.
    KittiReaderImageResolution,
    /// KITTI writer canonical file layout.
    KittiWriterFileLayout,
    /// KITTI writer uses default values for missing KITTI-specific attributes.
    KittiWriterDefaultFieldValues,
    /// KITTI writer deterministic ordering policy.
    KittiWriterDeterministicOrder,
    /// KITTI writer does not copy images.
    KittiWriterNoImageCopy,

    // VIA policy (Info level)
    /// VIA reader deterministic ID assignment policy.
    ViaReaderIdAssignment,
    /// VIA reader label resolution precedence.
    ViaReaderLabelResolution,
    /// VIA reader image resolution precedence.
    ViaReaderImageResolution,
    /// VIA writer deterministic ordering policy.
    ViaWriterDeterministicOrder,
    /// VIA writer canonical label attribute key.
    ViaWriterLabelAttributeKey,
    /// VIA writer does not copy images.
    ViaWriterNoImageCopy,

    // RetinaNet policy (Info level)
    /// RetinaNet reader deterministic ID assignment policy.
    RetinanetReaderIdAssignment,
    /// RetinaNet reader image resolution precedence.
    RetinanetReaderImageResolution,
    /// RetinaNet reader empty-row handling.
    RetinanetReaderEmptyRowHandling,
    /// RetinaNet writer deterministic ordering policy.
    RetinanetWriterDeterministicOrder,
    /// RetinaNet writer empty-row convention for unannotated images.
    RetinanetWriterEmptyRows,
    /// RetinaNet writer does not copy images.
    RetinanetWriterNoImageCopy,

    // OpenImages policy (Info level)
    /// OpenImages reader deterministic ID assignment policy.
    OpenimagesReaderIdAssignment,
    /// OpenImages reader resolves image dimensions from local files.
    OpenimagesReaderImageResolution,
    /// OpenImages writer deterministic ordering policy.
    OpenimagesWriterDeterministicOrder,

    // Kaggle Wheat policy (Info level)
    /// Kaggle Wheat reader deterministic ID assignment policy.
    KaggleWheatReaderIdAssignment,
    /// Kaggle Wheat writer deterministic ordering and bbox formatting.
    KaggleWheatWriterDeterministicOrder,

    // Kaggle Wheat lossiness
    /// Multiple categories collapsed to single class for single-class format.
    CollapseMultipleCategoriesToSingleClass,

    // AutoML Vision policy (Info level)
    /// AutoML Vision reader deterministic ID assignment policy.
    AutomlVisionReaderIdAssignment,
    /// AutoML Vision reader resolves image dimensions from local files.
    AutomlVisionReaderImageResolution,
    /// AutoML Vision writer deterministic ordering policy.
    AutomlVisionWriterDeterministicOrder,

    // Udacity policy (Info level)
    /// Udacity reader deterministic ID assignment policy.
    UdacityReaderIdAssignment,
    /// Udacity writer deterministic row ordering.
    UdacityWriterRowOrder,
}

impl ConversionIssueCode {
    /// All known issue codes, for drift-prevention testing.
    pub const ALL: &'static [ConversionIssueCode] = &[
        Self::DropDatasetInfo,
        Self::DropLicenses,
        Self::DropImageMetadata,
        Self::DropCategorySupercategory,
        Self::DropAnnotationConfidence,
        Self::DropAnnotationAttributes,
        Self::DropImagesWithoutAnnotations,
        Self::DropDatasetInfoName,
        Self::CocoAttributesMayNotBePreserved,
        Self::CocoWriterDeterministicOrder,
        Self::CocoWriterScoreMapping,
        Self::CocoWriterAreaIscrowdMapping,
        Self::CocoWriterEmptySegmentation,
        Self::CocoReaderAttributeMapping,
        Self::HfMetadataLost,
        Self::HfAttributesLost,
        Self::HfConfidenceLost,
        Self::HfReaderObjectContainerPrecedence,
        Self::HfReaderBboxFormatDependence,
        Self::LabelStudioWriterConfidenceRouting,
        Self::YoloWriterDeterministicOrder,
        Self::YoloWriterNoImageCopy,
        Self::YoloWriterDataYamlPolicy,
        Self::TfodReaderIdAssignment,
        Self::TfodWriterRowOrder,
        Self::YoloReaderIdAssignment,
        Self::YoloReaderClassMapSource,
        Self::YoloReaderSplitHandling,
        Self::YoloWriterClassOrder,
        Self::YoloWriterEmptyLabelFiles,
        Self::YoloWriterFloatPrecision,
        Self::VocReaderIdAssignment,
        Self::VocReaderAttributeMapping,
        Self::VocReaderCoordinatePolicy,
        Self::VocReaderDepthHandling,
        Self::VocWriterFileLayout,
        Self::VocWriterNoImageCopy,
        Self::VocWriterBoolNormalization,
        Self::LabelStudioRotationDropped,
        Self::LabelStudioReaderIdAssignment,
        Self::LabelStudioReaderImageRefPolicy,
        Self::LabelStudioWriterFromToDefaults,
        Self::CvatReaderIdAssignment,
        Self::CvatReaderAttributePolicy,
        Self::CvatWriterMetaDefaults,
        Self::CvatWriterDeterministicOrder,
        Self::CvatWriterImageIdReassignment,
        Self::CvatWriterSourceDefault,
        Self::CvatWriterDropUnusedCategories,
        Self::HfReaderCategoryResolution,
        Self::HfWriterDeterministicOrder,
        Self::SagemakerReaderIdAssignment,
        Self::SagemakerReaderLabelAttributeDetection,
        Self::SagemakerReaderClassMapResolution,
        Self::SagemakerWriterDeterministicOrder,
        Self::SagemakerWriterClassMapPolicy,
        Self::SagemakerWriterMetadataDefaults,
        Self::SagemakerWriterNoImageCopy,
        Self::LabelmeReaderIdAssignment,
        Self::LabelmeReaderPathPolicy,
        Self::LabelmePolygonEnvelopeApplied,
        Self::LabelmeWriterFileLayout,
        Self::LabelmeWriterRectanglePolicy,
        Self::LabelmeWriterNoImageCopy,
        Self::CreatemlReaderIdAssignment,
        Self::CreatemlReaderImageResolution,
        Self::CreatemlWriterDeterministicOrder,
        Self::CreatemlWriterCoordinateMapping,
        Self::CreatemlWriterNoImageCopy,
        Self::KittiReaderIdAssignment,
        Self::KittiReaderFieldMapping,
        Self::KittiReaderImageResolution,
        Self::KittiWriterFileLayout,
        Self::KittiWriterDefaultFieldValues,
        Self::KittiWriterDeterministicOrder,
        Self::KittiWriterNoImageCopy,
        Self::ViaReaderIdAssignment,
        Self::ViaReaderLabelResolution,
        Self::ViaReaderImageResolution,
        Self::ViaWriterDeterministicOrder,
        Self::ViaWriterLabelAttributeKey,
        Self::ViaWriterNoImageCopy,
        Self::RetinanetReaderIdAssignment,
        Self::RetinanetReaderImageResolution,
        Self::RetinanetReaderEmptyRowHandling,
        Self::RetinanetWriterDeterministicOrder,
        Self::RetinanetWriterEmptyRows,
        Self::RetinanetWriterNoImageCopy,
        Self::OpenimagesReaderIdAssignment,
        Self::OpenimagesReaderImageResolution,
        Self::OpenimagesWriterDeterministicOrder,
        Self::KaggleWheatReaderIdAssignment,
        Self::KaggleWheatWriterDeterministicOrder,
        Self::CollapseMultipleCategoriesToSingleClass,
        Self::AutomlVisionReaderIdAssignment,
        Self::AutomlVisionReaderImageResolution,
        Self::AutomlVisionWriterDeterministicOrder,
        Self::UdacityReaderIdAssignment,
        Self::UdacityWriterRowOrder,
    ];

    /// Canonical stable string form, shared by text and JSON output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DropDatasetInfo => "drop_dataset_info",
            Self::DropLicenses => "drop_licenses",
            Self::DropImageMetadata => "drop_image_metadata",
            Self::DropCategorySupercategory => "drop_category_supercategory",
            Self::DropAnnotationConfidence => "drop_annotation_confidence",
            Self::DropAnnotationAttributes => "drop_annotation_attributes",
            Self::DropImagesWithoutAnnotations => "drop_images_without_annotations",
            Self::DropDatasetInfoName => "drop_dataset_info_name",
            Self::CocoAttributesMayNotBePreserved => "coco_attributes_may_not_be_preserved",
            Self::CocoWriterDeterministicOrder => "coco_writer_deterministic_order",
            Self::CocoWriterScoreMapping => "coco_writer_score_mapping",
            Self::CocoWriterAreaIscrowdMapping => "coco_writer_area_iscrowd_mapping",
            Self::CocoWriterEmptySegmentation => "coco_writer_empty_segmentation",
            Self::CocoReaderAttributeMapping => "coco_reader_attribute_mapping",
            Self::HfMetadataLost => "hf_metadata_lost",
            Self::HfAttributesLost => "hf_attributes_lost",
            Self::HfConfidenceLost => "hf_confidence_lost",
            Self::HfReaderObjectContainerPrecedence => "hf_reader_object_container_precedence",
            Self::HfReaderBboxFormatDependence => "hf_reader_bbox_format_dependence",
            Self::LabelStudioWriterConfidenceRouting => "label_studio_writer_confidence_routing",
            Self::YoloWriterDeterministicOrder => "yolo_writer_deterministic_order",
            Self::YoloWriterNoImageCopy => "yolo_writer_no_image_copy",
            Self::YoloWriterDataYamlPolicy => "yolo_writer_data_yaml_policy",
            Self::TfodReaderIdAssignment => "tfod_reader_id_assignment",
            Self::TfodWriterRowOrder => "tfod_writer_row_order",
            Self::YoloReaderIdAssignment => "yolo_reader_id_assignment",
            Self::YoloReaderClassMapSource => "yolo_reader_class_map_source",
            Self::YoloReaderSplitHandling => "yolo_reader_split_handling",
            Self::YoloWriterClassOrder => "yolo_writer_class_order",
            Self::YoloWriterEmptyLabelFiles => "yolo_writer_empty_label_files",
            Self::YoloWriterFloatPrecision => "yolo_writer_float_precision",
            Self::VocReaderIdAssignment => "voc_reader_id_assignment",
            Self::VocReaderAttributeMapping => "voc_reader_attribute_mapping",
            Self::VocReaderCoordinatePolicy => "voc_reader_coordinate_policy",
            Self::VocReaderDepthHandling => "voc_reader_depth_handling",
            Self::VocWriterFileLayout => "voc_writer_file_layout",
            Self::VocWriterNoImageCopy => "voc_writer_no_image_copy",
            Self::VocWriterBoolNormalization => "voc_writer_bool_normalization",
            Self::LabelStudioRotationDropped => "label_studio_rotation_dropped",
            Self::LabelStudioReaderIdAssignment => "label_studio_reader_id_assignment",
            Self::LabelStudioReaderImageRefPolicy => "label_studio_reader_image_ref_policy",
            Self::LabelStudioWriterFromToDefaults => "label_studio_writer_from_to_defaults",
            Self::CvatReaderIdAssignment => "cvat_reader_id_assignment",
            Self::CvatReaderAttributePolicy => "cvat_reader_attribute_policy",
            Self::CvatWriterMetaDefaults => "cvat_writer_meta_defaults",
            Self::CvatWriterDeterministicOrder => "cvat_writer_deterministic_order",
            Self::CvatWriterImageIdReassignment => "cvat_writer_image_id_reassignment",
            Self::CvatWriterSourceDefault => "cvat_writer_source_default",
            Self::CvatWriterDropUnusedCategories => "cvat_writer_drop_unused_categories",
            Self::HfReaderCategoryResolution => "hf_reader_category_resolution",
            Self::HfWriterDeterministicOrder => "hf_writer_deterministic_order",
            Self::SagemakerReaderIdAssignment => "sagemaker_reader_id_assignment",
            Self::SagemakerReaderLabelAttributeDetection => {
                "sagemaker_reader_label_attribute_detection"
            }
            Self::SagemakerReaderClassMapResolution => "sagemaker_reader_class_map_resolution",
            Self::SagemakerWriterDeterministicOrder => "sagemaker_writer_deterministic_order",
            Self::SagemakerWriterClassMapPolicy => "sagemaker_writer_class_map_policy",
            Self::SagemakerWriterMetadataDefaults => "sagemaker_writer_metadata_defaults",
            Self::SagemakerWriterNoImageCopy => "sagemaker_writer_no_image_copy",
            Self::LabelmeReaderIdAssignment => "labelme_reader_id_assignment",
            Self::LabelmeReaderPathPolicy => "labelme_reader_path_policy",
            Self::LabelmePolygonEnvelopeApplied => "labelme_polygon_envelope_applied",
            Self::LabelmeWriterFileLayout => "labelme_writer_file_layout",
            Self::LabelmeWriterRectanglePolicy => "labelme_writer_rectangle_policy",
            Self::LabelmeWriterNoImageCopy => "labelme_writer_no_image_copy",
            Self::CreatemlReaderIdAssignment => "createml_reader_id_assignment",
            Self::CreatemlReaderImageResolution => "createml_reader_image_resolution",
            Self::CreatemlWriterDeterministicOrder => "createml_writer_deterministic_order",
            Self::CreatemlWriterCoordinateMapping => "createml_writer_coordinate_mapping",
            Self::CreatemlWriterNoImageCopy => "createml_writer_no_image_copy",
            Self::KittiReaderIdAssignment => "kitti_reader_id_assignment",
            Self::KittiReaderFieldMapping => "kitti_reader_field_mapping",
            Self::KittiReaderImageResolution => "kitti_reader_image_resolution",
            Self::KittiWriterFileLayout => "kitti_writer_file_layout",
            Self::KittiWriterDefaultFieldValues => "kitti_writer_default_field_values",
            Self::KittiWriterDeterministicOrder => "kitti_writer_deterministic_order",
            Self::KittiWriterNoImageCopy => "kitti_writer_no_image_copy",
            Self::ViaReaderIdAssignment => "via_reader_id_assignment",
            Self::ViaReaderLabelResolution => "via_reader_label_resolution",
            Self::ViaReaderImageResolution => "via_reader_image_resolution",
            Self::ViaWriterDeterministicOrder => "via_writer_deterministic_order",
            Self::ViaWriterLabelAttributeKey => "via_writer_label_attribute_key",
            Self::ViaWriterNoImageCopy => "via_writer_no_image_copy",
            Self::RetinanetReaderIdAssignment => "retinanet_reader_id_assignment",
            Self::RetinanetReaderImageResolution => "retinanet_reader_image_resolution",
            Self::RetinanetReaderEmptyRowHandling => "retinanet_reader_empty_row_handling",
            Self::RetinanetWriterDeterministicOrder => "retinanet_writer_deterministic_order",
            Self::RetinanetWriterEmptyRows => "retinanet_writer_empty_rows",
            Self::RetinanetWriterNoImageCopy => "retinanet_writer_no_image_copy",
            Self::OpenimagesReaderIdAssignment => "openimages_reader_id_assignment",
            Self::OpenimagesReaderImageResolution => "openimages_reader_image_resolution",
            Self::OpenimagesWriterDeterministicOrder => "openimages_writer_deterministic_order",
            Self::KaggleWheatReaderIdAssignment => "kaggle_wheat_reader_id_assignment",
            Self::KaggleWheatWriterDeterministicOrder => "kaggle_wheat_writer_deterministic_order",
            Self::CollapseMultipleCategoriesToSingleClass => {
                "collapse_multiple_categories_to_single_class"
            }
            Self::AutomlVisionReaderIdAssignment => "automl_vision_reader_id_assignment",
            Self::AutomlVisionReaderImageResolution => "automl_vision_reader_image_resolution",
            Self::AutomlVisionWriterDeterministicOrder => {
                "automl_vision_writer_deterministic_order"
            }
            Self::UdacityReaderIdAssignment => "udacity_reader_id_assignment",
            Self::UdacityWriterRowOrder => "udacity_writer_row_order",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_report_is_not_lossy() {
        let report = ConversionReport::new("coco", "ir-json");
        assert!(!report.is_lossy());
        assert_eq!(report.warning_count(), 0);
        assert_eq!(report.info_count(), 0);
    }

    #[test]
    fn warning_makes_report_lossy() {
        let mut report = ConversionReport::new("ir-json", "tfod");
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropDatasetInfo,
            "dataset info will be dropped",
        ));
        assert!(report.is_lossy());
        assert_eq!(report.warning_count(), 1);
    }

    #[test]
    fn info_does_not_make_report_lossy() {
        let mut report = ConversionReport::new("tfod", "coco");
        report.add(ConversionIssue::info(
            ConversionIssueCode::TfodReaderIdAssignment,
            "IDs assigned by lexicographic order",
        ));
        assert!(!report.is_lossy());
        assert_eq!(report.info_count(), 1);
    }

    #[test]
    fn report_serializes_to_json() {
        let mut report = ConversionReport::new("coco", "tfod");
        report.input = ConversionCounts {
            images: 10,
            categories: 3,
            annotations: 50,
        };
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropLicenses,
            "2 license(s) will be dropped",
        ));

        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"from\":\"coco\""));
        assert!(json.contains("\"severity\":\"warning\""));
        assert!(json.contains("\"code\":\"drop_licenses\""));
    }

    #[test]
    fn as_str_matches_serde_json_for_all_codes() {
        // Verify that as_str() produces the same string as serde serialization
        // for EVERY code in ALL — catches as_str()/serde drift.
        for code in ConversionIssueCode::ALL {
            let json = serde_json::to_value(code).unwrap();
            assert_eq!(
                json.as_str().unwrap(),
                code.as_str(),
                "as_str() and serde disagree for {:?}",
                code
            );
        }
    }

    #[test]
    fn all_codes_have_unique_str() {
        let mut seen = std::collections::HashSet::new();
        for code in ConversionIssueCode::ALL {
            assert!(
                seen.insert(code.as_str()),
                "duplicate as_str() value: {}",
                code.as_str()
            );
        }
    }

    #[test]
    fn all_codes_documented_in_conversion_md() {
        let docs =
            std::fs::read_to_string("docs/conversion.md").expect("docs/conversion.md should exist");
        let mut missing = Vec::new();
        for code in ConversionIssueCode::ALL {
            if !docs.contains(code.as_str()) {
                missing.push(code.as_str());
            }
        }
        assert!(
            missing.is_empty(),
            "docs/conversion.md is missing these issue codes: {:?}",
            missing
        );
    }

    #[test]
    fn text_display_includes_stable_codes() {
        let mut report = ConversionReport::new("coco", "tfod");
        report.input = ConversionCounts {
            images: 5,
            categories: 2,
            annotations: 10,
        };
        report.output = report.input.clone();
        report.add(ConversionIssue::warning(
            ConversionIssueCode::DropDatasetInfo,
            "dataset info will be dropped",
        ));
        report.add(ConversionIssue::info(
            ConversionIssueCode::TfodWriterRowOrder,
            "rows ordered by annotation ID",
        ));

        let text = format!("{}", report);
        assert!(
            text.contains("[drop_dataset_info]"),
            "text should contain warning code"
        );
        assert!(
            text.contains("[tfod_writer_row_order]"),
            "text should contain info code"
        );
    }
}
