use std::path::PathBuf;
use thiserror::Error;

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

    #[error("Validation failed with {error_count} error(s) and {warning_count} warning(s)")]
    ValidationFailed {
        error_count: usize,
        warning_count: usize,
        report: ValidationReport,
    },

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}
