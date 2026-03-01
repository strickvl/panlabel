use std::path::PathBuf;
use thiserror::Error;

use crate::conversion::ConversionReport;
use crate::validation::ValidationReport;

/// The main error type for panlabel operations.
#[derive(Debug, Error)]
pub enum PanlabelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse IR JSON from {path}: {source}")]
    IrJsonParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("Failed to write IR JSON to {path}: {source}")]
    IrJsonWrite {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("Failed to parse COCO JSON from {path}: {source}")]
    CocoJsonParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("Failed to write COCO JSON to {path}: {source}")]
    CocoJsonWrite {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("Failed to parse Label Studio JSON from {path}: {source}")]
    LabelStudioJsonParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("Failed to write Label Studio JSON to {path}: {source}")]
    LabelStudioJsonWrite {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("Invalid Label Studio JSON: {path}: {message}")]
    LabelStudioJsonInvalid { path: PathBuf, message: String },

    #[error("Failed to parse TFOD CSV from {path}: {source}")]
    TfodCsvParse {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },

    #[error("Failed to write TFOD CSV to {path}: {source}")]
    TfodCsvWrite {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },

    #[error("Invalid TFOD CSV: {path}: {message}")]
    TfodCsvInvalid { path: PathBuf, message: String },

    #[error("Invalid YOLO dataset layout at {path}: {message}")]
    YoloLayoutInvalid { path: PathBuf, message: String },

    #[error("Failed to parse YOLO data.yaml at {path}: {source}")]
    YoloDataYamlParse {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },

    #[error("Invalid YOLO classes.txt at {path}: {message}")]
    YoloClassesTxtInvalid { path: PathBuf, message: String },

    #[error("Failed to parse YOLO label row in {path}:{line}: {message}")]
    YoloLabelParse {
        path: PathBuf,
        line: usize,
        message: String,
    },

    #[error(
        "No matching image found for label file {label_path} (expected stem: {expected_stem})"
    )]
    YoloImageNotFound {
        label_path: PathBuf,
        expected_stem: String,
    },

    #[error("Failed to read YOLO image dimensions from {path}: {source}")]
    YoloImageDimensionRead {
        path: PathBuf,
        #[source]
        source: imagesize::ImageError,
    },

    #[error("Failed to write YOLO dataset at {path}: {message}")]
    YoloWriteError { path: PathBuf, message: String },

    #[error("Invalid VOC dataset layout at {path}: {message}")]
    VocLayoutInvalid { path: PathBuf, message: String },

    #[error("Failed to parse VOC XML from {path}: {message}")]
    VocXmlParse { path: PathBuf, message: String },

    #[error("Failed to write VOC dataset at {path}: {message}")]
    VocWriteError { path: PathBuf, message: String },

    #[error("Invalid CVAT XML layout at {path}: {message}")]
    CvatLayoutInvalid { path: PathBuf, message: String },

    #[error("Failed to parse CVAT XML from {path}: {message}")]
    CvatXmlParse { path: PathBuf, message: String },

    #[error("Failed to write CVAT XML at {path}: {message}")]
    CvatWriteError { path: PathBuf, message: String },

    #[error("Invalid HF ImageFolder layout at {path}: {message}")]
    HfLayoutInvalid { path: PathBuf, message: String },

    #[error("Failed to parse HF metadata.jsonl at {path}, line {line}: {message}")]
    HfJsonlParse {
        path: PathBuf,
        line: usize,
        message: String,
    },

    #[error("Failed to write HF metadata.jsonl at {path}: {message}")]
    HfWriteError { path: PathBuf, message: String },

    #[cfg(feature = "hf-parquet")]
    #[error("Failed to parse HF metadata.parquet at {path}: {message}")]
    HfParquetParse { path: PathBuf, message: String },

    #[cfg(feature = "hf-remote")]
    #[error("Invalid HF repo reference '{input}': {message}")]
    HfResolveError { input: String, message: String },

    #[cfg(feature = "hf-remote")]
    #[error("HF Hub API error for {repo_id}: {message}")]
    HfApiError { repo_id: String, message: String },

    #[cfg(feature = "hf-remote")]
    #[error("Failed to download from HF Hub ({repo_id}): {message}")]
    HfAcquireError { repo_id: String, message: String },

    #[cfg(feature = "hf-remote")]
    #[error("Unsupported HF zip payload for {repo_id}: {message}")]
    HfZipLayoutInvalid { repo_id: String, message: String },

    #[error("Validation failed with {error_count} error(s) and {warning_count} warning(s)")]
    ValidationFailed {
        error_count: usize,
        warning_count: usize,
        report: ValidationReport,
    },

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Failed to detect format for {path}: {reason}")]
    FormatDetectionFailed { path: PathBuf, reason: String },

    #[error("Failed to parse JSON while detecting format for {path}: {source}")]
    FormatDetectionJsonParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("Lossy conversion from {from} to {to} would drop information (use --allow-lossy to proceed):\n{}", format_lossy_messages(report))]
    LossyConversionBlocked {
        from: String,
        to: String,
        report: Box<ConversionReport>,
    },

    #[error("Diff failed: {message}")]
    DiffFailed { message: String },

    #[error("Sample failed: {message}")]
    SampleFailed { message: String },

    #[error("Invalid sample parameters: {message}")]
    InvalidSampleParams { message: String },

    #[error("Failed to write report as JSON: {source}")]
    ReportJsonWrite {
        #[source]
        source: serde_json::Error,
    },
}

/// Format lossy warning messages for error display.
fn format_lossy_messages(report: &ConversionReport) -> String {
    report
        .lossy_messages()
        .map(|msg| format!("  - {}", msg))
        .collect::<Vec<_>>()
        .join("\n")
}
