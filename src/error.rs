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

    #[error("Validation failed with {error_count} error(s) and {warning_count} warning(s)")]
    ValidationFailed {
        error_count: usize,
        warning_count: usize,
        report: ValidationReport,
    },

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}
