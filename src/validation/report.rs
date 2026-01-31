//! Validation report types for structured error reporting.
//!
//! This module provides rich, structured validation results that can be
//! displayed to users, written to files, or processed programmatically.

use std::fmt;

/// The result of validating a dataset.
///
/// Contains all issues found during validation, categorized by severity.
#[derive(Clone, Debug, Default)]
pub struct ValidationReport {
    /// All issues found during validation.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Creates a new empty report.
    pub fn new() -> Self {
        Self { issues: Vec::new() }
    }

    /// Adds an issue to the report.
    pub fn add(&mut self, issue: ValidationIssue) {
        self.issues.push(issue);
    }

    /// Returns the number of errors in the report.
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .count()
    }

    /// Returns the number of warnings in the report.
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .count()
    }

    /// Returns true if there are no errors.
    pub fn is_ok(&self) -> bool {
        self.error_count() == 0
    }

    /// Returns true if there are no issues at all.
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// Returns true if validation passed in strict mode (no errors or warnings).
    pub fn is_ok_strict(&self) -> bool {
        self.issues.is_empty()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.issues.is_empty() {
            return writeln!(f, "Validation passed: no issues found");
        }

        writeln!(
            f,
            "Validation completed with {} error(s) and {} warning(s):",
            self.error_count(),
            self.warning_count()
        )?;
        writeln!(f)?;

        for issue in &self.issues {
            writeln!(f, "  {}", issue)?;
        }

        Ok(())
    }
}

/// A single validation issue (error or warning).
#[derive(Clone, Debug)]
pub struct ValidationIssue {
    /// The severity of the issue.
    pub severity: Severity,

    /// A stable code for the issue type.
    pub code: IssueCode,

    /// A human-readable description of the issue.
    pub message: String,

    /// Context about where the issue occurred.
    pub context: IssueContext,
}

impl ValidationIssue {
    /// Creates a new validation issue.
    pub fn new(
        severity: Severity,
        code: IssueCode,
        message: impl Into<String>,
        context: IssueContext,
    ) -> Self {
        Self {
            severity,
            code,
            message: message.into(),
            context,
        }
    }

    /// Creates a new error.
    pub fn error(code: IssueCode, message: impl Into<String>, context: IssueContext) -> Self {
        Self::new(Severity::Error, code, message, context)
    }

    /// Creates a new warning.
    pub fn warning(code: IssueCode, message: impl Into<String>, context: IssueContext) -> Self {
        Self::new(Severity::Warning, code, message, context)
    }
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let severity = match self.severity {
            Severity::Error => "ERROR",
            Severity::Warning => "WARN ",
        };
        write!(
            f,
            "[{}] {:?} in {}: {}",
            severity, self.code, self.context, self.message
        )
    }
}

/// The severity of a validation issue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// A warning that doesn't prevent conversion but may indicate problems.
    Warning,
    /// An error that indicates invalid or corrupt data.
    Error,
}

/// A stable code identifying the type of validation issue.
///
/// These codes can be used for filtering, ignoring specific issues,
/// or programmatic handling of validation results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IssueCode {
    // ID uniqueness issues
    /// Multiple images have the same ID.
    DuplicateImageId,
    /// Multiple annotations have the same ID.
    DuplicateAnnotationId,
    /// Multiple categories have the same ID.
    DuplicateCategoryId,

    // Reference issues
    /// An annotation references a non-existent image.
    MissingImageRef,
    /// An annotation references a non-existent category.
    MissingCategoryRef,

    // Image issues
    /// An image has invalid dimensions (zero or negative).
    InvalidImageDimensions,
    /// An image has an empty filename.
    EmptyFileName,

    // Category issues
    /// A category has an empty name.
    EmptyCategoryName,
    /// Multiple categories have the same name (potential confusion).
    DuplicateCategoryName,

    // Bounding box issues
    /// A bounding box has non-finite coordinates (NaN or Infinity).
    BBoxNotFinite,
    /// A bounding box has incorrect ordering (min > max).
    InvalidBBoxOrdering,
    /// A bounding box extends outside the image bounds.
    BBoxOutOfBounds,
    /// A bounding box has zero or negative area.
    InvalidBBoxArea,
}

/// Context about where a validation issue occurred.
#[derive(Clone, Debug)]
pub enum IssueContext {
    /// Issue with the dataset as a whole.
    Dataset,
    /// Issue with a specific image.
    Image { id: u64 },
    /// Issue with a specific annotation.
    Annotation { id: u64 },
    /// Issue with a specific category.
    Category { id: u64 },
}

impl fmt::Display for IssueContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IssueContext::Dataset => write!(f, "dataset"),
            IssueContext::Image { id } => write!(f, "image {}", id),
            IssueContext::Annotation { id } => write!(f, "annotation {}", id),
            IssueContext::Category { id } => write!(f, "category {}", id),
        }
    }
}
