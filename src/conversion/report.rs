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

    /// Iterate over warning messages (for error display compatibility).
    pub fn lossy_messages(&self) -> impl Iterator<Item = &str> {
        self.issues
            .iter()
            .filter(|i| i.severity == ConversionSeverity::Warning)
            .map(|i| i.message.as_str())
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
                    writeln!(f, "  - {}", issue.message)?;
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
                    writeln!(f, "  - {}", issue.message)?;
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
    pub code: ConversionIssueCode,
    pub message: String,
}

impl ConversionIssue {
    /// Create a warning-level issue (indicates lossiness).
    pub fn warning(code: ConversionIssueCode, message: impl Into<String>) -> Self {
        Self {
            severity: ConversionSeverity::Warning,
            code,
            message: message.into(),
        }
    }

    /// Create an info-level issue (policy note, does not block).
    pub fn info(code: ConversionIssueCode, message: impl Into<String>) -> Self {
        Self {
            severity: ConversionSeverity::Info,
            code,
            message: message.into(),
        }
    }
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

    // Policy decisions (Info level)
    /// TFOD reader assigns IDs by lexicographic ordering.
    TfodReaderIdAssignment,
    /// TFOD writer orders rows by annotation ID.
    TfodWriterRowOrder,
    /// YOLO reader assigns IDs by deterministic ordering.
    YoloReaderIdAssignment,
    /// YOLO reader class-map precedence/source.
    YoloReaderClassMapSource,
    /// YOLO writer assigns class indices by category ID order.
    YoloWriterClassOrder,
    /// YOLO writer creates empty label files for images without annotations.
    YoloWriterEmptyLabelFiles,
    /// YOLO writer outputs normalized floats at 6 decimal places.
    YoloWriterFloatPrecision,
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
}
